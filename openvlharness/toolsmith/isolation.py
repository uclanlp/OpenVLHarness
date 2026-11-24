"""Confine a command with Linux namespaces: it sees only the paths it needs, read-only, and optionally no network.

Usage: python -m openvlharness.toolsmith.isolation SPEC -- COMMAND [ARGUMENT...]

Without root, the launcher creates user, mount, PID and IPC namespaces, and a network
namespace with no interfaces up unless SPEC allows network access. The caller keeps
its uid and gid. In the new namespaces a small init (PID 1) runs the command as PID 2
on a fresh root filesystem built in SPEC's root directory: read-only bind mounts of
the read paths, writable bind mounts of the write paths, private tmpfs directories of at
most 1 GiB each, a fresh /proc that shows only the namespace's processes, and a /dev
holding null, zero, full, random and urandom. The host filesystem is detached, the command starts without
capabilities and cannot gain privileges, and everything in the namespace dies with the
launcher. SPEC is JSON with root, read, write, mask (paths hidden inside read paths),
tmp, cwd, network and cleanup (a directory the launcher removes when the command ends).
"""
import ctypes
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys

CLONE_NEWNS, CLONE_NEWIPC, CLONE_NEWUSER, CLONE_NEWPID, CLONE_NEWNET = 0x20000, 0x8000000, 0x10000000, 0x20000000, 0x40000000
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_REMOUNT = 0x1, 0x2, 0x4, 0x8, 0x20
MS_NOATIME, MS_NODIRATIME, MS_BIND, MS_REC, MS_PRIVATE, MS_RELATIME = 0x400, 0x800, 0x1000, 0x4000, 0x40000, 0x200000
MNT_DETACH = 2
PR_SET_PDEATHSIG, PR_SET_NO_NEW_PRIVS = 1, 38
PIVOT_ROOT = {"x86_64": 155, "aarch64": 41}
DEVICES = ("null", "zero", "full", "random", "urandom")
# What Python, OpenSSL and the resolver read from /etc: name resolution, certificates, the dynamic
# linker cache, the time zone and MIME types. Nothing else under /etc (or /opt) is mounted.
ETC = ("resolv.conf", "hosts", "nsswitch.conf", "host.conf", "gai.conf", "services", "protocols", "ssl",
       "ca-certificates", "ca-certificates.conf", "pki", "ld.so.cache", "ld.so.conf", "ld.so.conf.d", "localtime",
       "timezone", "mime.types")
SETUP_FAILED = 125
# Caps of the tmpfs mounts, whose memory the address-space limit does not count: each private
# temporary directory and /dev/shm, and the root and /dev, which hold only mount points and devices.
TMPFS = "size=1g,nr_inodes=65536"
SKELETON = "size=16m,nr_inodes=65536"
# statvfs flags of a mount that a read-only remount inside a user namespace must keep.
LOCKED = tuple((getattr(os, name), flag) for name, flag in (
    ("ST_NOSUID", MS_NOSUID), ("ST_NODEV", MS_NODEV), ("ST_NOEXEC", MS_NOEXEC), ("ST_NOATIME", MS_NOATIME),
    ("ST_NODIRATIME", MS_NODIRATIME), ("ST_RELATIME", MS_RELATIME)) if hasattr(os, name))
_probe = None


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]
    libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
    libc.unshare.argtypes = [ctypes.c_int]
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    return libc


def _check(result, what):
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"{what}: {os.strerror(error)}")


def _bytes(value):
    return os.fsencode(value) if value is not None else None


def _mount(libc, source, target, fstype, flags, data=None):
    _check(libc.mount(_bytes(source), _bytes(target), _bytes(fstype), flags, _bytes(data)), f"mount {target}")


def _target(root, path, *, directory):
    """Create the mount point for path inside root and return it."""
    target = root + path
    if directory:
        os.makedirs(target, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if not os.path.exists(target):
            open(target, "w").close()
    return target


def _bind(libc, root, path, *, writable):
    """Bind-mount a host path at the same place inside root, read-only unless writable."""
    if os.path.islink(path) and os.path.dirname(path) == "/":
        # Top-level links such as /bin -> usr/bin are recreated rather than mounted.
        if not os.path.lexists(root + path):
            os.symlink(os.readlink(path), root + path)
        return
    target = _target(root, path, directory=os.path.isdir(path))
    _mount(libc, path, target, None, MS_BIND | MS_REC)
    if not writable:
        _read_only(libc, target)


def _read_only(libc, target):
    """Remount target and every mount below it read-only, keeping the flags the kernel locks."""
    with open("/proc/self/mountinfo") as stream:
        points = [re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), line.split()[4])
                  for line in stream]
    for point in [target] + sorted(point for point in points if point.startswith(target.rstrip("/") + "/")):
        flags = os.statvfs(point).f_flag
        if flags & os.ST_RDONLY:
            continue
        try:
            _mount(libc, None, point, None, MS_REMOUNT | MS_BIND | MS_RDONLY | sum(ms for st, ms in LOCKED if flags & st))
        except OSError:
            # A kernel file system below /sys may refuse; that is harmless only where we could not write anyway.
            if point == target or os.access(point, os.W_OK):
                raise


def _build_root(libc, spec):
    """Assemble the new root in spec["root"] and pivot into it."""
    root = os.path.realpath(spec["root"])
    _mount(libc, None, "/", None, MS_REC | MS_PRIVATE)
    _mount(libc, "tmpfs", root, "tmpfs", MS_NOSUID | MS_NODEV, f"mode=0755,{SKELETON}")
    entries = [(path, "read") for path in spec.get("read", [])] + [(path, "write") for path in spec.get("write", [])]
    entries += [(path, "tmp") for path in spec.get("tmp", [])]
    for path, kind in sorted(entries, key=lambda entry: (entry[0].rstrip("/").count("/"), entry[0])):
        if kind == "tmp":
            _mount(libc, "tmpfs", _target(root, path, directory=True), "tmpfs", MS_NOSUID | MS_NODEV,
                   f"mode=1777,{TMPFS}")
        elif os.path.lexists(path):
            _bind(libc, root, path, writable=kind == "write")
    for path in spec.get("mask", []):
        if os.path.isdir(root + path):
            _mount(libc, "tmpfs", root + path, "tmpfs", MS_NOSUID | MS_NODEV | MS_RDONLY, "mode=0755")
        elif os.path.exists(root + path):
            _mount(libc, "/dev/null", root + path, None, MS_BIND)
    _mount(libc, "proc", _target(root, "/proc", directory=True), "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    dev = _target(root, "/dev", directory=True)
    _mount(libc, "tmpfs", dev, "tmpfs", MS_NOSUID | MS_NOEXEC, f"mode=0755,{SKELETON}")
    for name in DEVICES:
        _mount(libc, f"/dev/{name}", _target(root, f"/dev/{name}", directory=False), None, MS_BIND)
    _mount(libc, "tmpfs", _target(root, "/dev/shm", directory=True), "tmpfs", MS_NOSUID | MS_NODEV,
           f"mode=1777,{TMPFS}")
    for name, link in (("fd", "/proc/self/fd"), ("stdin", "/proc/self/fd/0"), ("stdout", "/proc/self/fd/1"),
                       ("stderr", "/proc/self/fd/2")):
        os.symlink(link, f"{dev}/{name}")
    os.chdir(root)
    os.mkdir(".oldroot")
    _check(libc.syscall(PIVOT_ROOT[platform.machine()], b".", b".oldroot"), "pivot_root")
    os.chdir("/")
    _check(libc.umount2(b"/.oldroot", MNT_DETACH), "umount the host root")
    os.rmdir("/.oldroot")
    _mount(libc, None, "/", None, MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV)
    os.chdir(spec.get("cwd") or "/")


ROOT_REFUSED = ("isolation refuses to run as root: a uid-0 command keeps its capabilities in the namespace "
                "and could undo the mounts; run as an unprivileged user")


def launch(spec, command):
    """Run command confined as described by spec and exit with its status (the module's entry point)."""
    if platform.machine() not in PIVOT_ROOT:
        sys.exit(f"isolation: unsupported architecture {platform.machine()}")
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        # A non-root command loses every capability at exec; root would keep them.
        print(f"isolation: {ROOT_REFUSED}", file=sys.stderr)
        os._exit(SETUP_FAILED)
    libc = _libc()
    # The child makes its own mount namespace below: pivot_root moves the root of every process
    # in a mount namespace, and the launcher keeps the host view to clean up afterwards.
    flags = CLONE_NEWUSER | CLONE_NEWPID | CLONE_NEWIPC | (0 if spec.get("network") else CLONE_NEWNET)
    try:
        _check(libc.unshare(flags), "unshare")
        for name, value in (("setgroups", "deny"), ("uid_map", f"{uid} {uid} 1"), ("gid_map", f"{gid} {gid} 1")):
            with open(f"/proc/self/{name}", "w") as stream:
                stream.write(value)
    except OSError as exc:
        print(f"isolation: {exc}", file=sys.stderr)
        os._exit(SETUP_FAILED)
    # The child reads end of file on `alive` if the launcher is gone before the death signal is
    # armed; it reports the command's wait status on `report`.
    alive, keep = os.pipe()
    report, reported = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(keep)
            os.close(report)
            _check(libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0), "prctl")
            os.set_blocking(alive, False)
            try:
                if os.read(alive, 1) == b"":
                    os._exit(SETUP_FAILED)
            except BlockingIOError:
                pass
            os.close(alive)
            _check(libc.unshare(CLONE_NEWNS), "unshare the mount namespace")
            _build_root(libc, spec)
            _check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "prctl")
            _init(command, reported)
        except BaseException as exc:
            print(f"isolation: {exc}", file=sys.stderr)
        os._exit(SETUP_FAILED)
    # The command alone holds the pipes, so the caller sees end of file when it ends.
    os.close(alive)
    os.close(reported)
    for descriptor in (0, 1):
        os.close(descriptor)
    _, status = os.waitpid(child, 0)
    answer = os.read(report, 32)
    status = int(answer) if answer else status
    cleanup = spec.get("cleanup")
    if cleanup and os.path.basename(cleanup.rstrip("/")).startswith("openvlharness-"):
        shutil.rmtree(cleanup, ignore_errors=True)
    if os.WIFSIGNALED(status):
        signal.signal(os.WTERMSIG(status), signal.SIG_DFL)
        os.kill(os.getpid(), os.WTERMSIG(status))
    os._exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else SETUP_FAILED)


def _init(command, reported):
    """Run as PID 1 of the namespace: start the command as PID 2, reap every child, report its status.

    The command gets ordinary signal handling, which PID 1 lacks; once it ends, the
    kernel stops whatever it left in the namespace.
    """
    process = os.fork()
    if process == 0:
        os.close(reported)
        os.execvp(command[0], command)
    for descriptor in (0, 1):
        os.close(descriptor)
    while True:
        pid, status = os.waitpid(-1, 0)
        if pid == process:
            os.write(reported, str(status).encode())
            os._exit(0)


def command(spec, argv):
    """Return the argument list that runs argv under the launcher."""
    return [sys.executable, "-m", "openvlharness.toolsmith.isolation", json.dumps(spec), "--", *argv]


def available():
    """Return (True, "") when this host can confine processes, else (False, the reason); cached per process."""
    global _probe
    if _probe is None:
        if sys.platform != "linux" or platform.machine() not in PIVOT_ROOT:
            _probe = (False, f"needs Linux on {' or '.join(PIVOT_ROOT)}, not {sys.platform} on {platform.machine()}")
        elif os.getuid() == 0:
            _probe = (False, ROOT_REFUSED)
        else:
            import tempfile

            with tempfile.TemporaryDirectory(prefix="openvlharness-probe-") as root:
                spec = {"root": root, "read": system_paths(), "tmp": ["/tmp"], "network": False}
                try:
                    done = subprocess.run(command(spec, [sys.executable, "-c", "pass"]), capture_output=True,
                                          text=True, timeout=60, env=probe_environment(), check=False)
                    reason = (done.stderr.strip().splitlines() or [f"exit status {done.returncode}"])[-1]
                    _probe = (True, "") if done.returncode == 0 else (False, reason)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    _probe = (False, str(exc))
    return _probe


def probe_environment():
    """Environment for the availability probe: enough to start Python and import this module."""
    from .worker import PACKAGE_ROOT

    return {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(PACKAGE_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}


def system_paths():
    """Host paths every confined Python process reads: system directories, the interpreter and this package."""
    from .worker import PACKAGE_ROOT

    paths = ["/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/sys", "/run/systemd/resolve",
             *(f"/etc/{name}" for name in ETC), sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
             os.path.dirname(os.path.realpath(sys.executable)), str(PACKAGE_ROOT / "openvlharness")]
    package_root = os.path.realpath(PACKAGE_ROOT)
    for entry in sys.path + os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        # The package root is mounted only as its openvlharness directory, not the repository around it.
        if entry and os.path.isdir(entry) and os.path.realpath(entry) not in (os.getcwd(), package_root):
            paths.append(os.path.realpath(entry))
    return list(dict.fromkeys(path for path in paths if os.path.lexists(path)))


def python_masks():
    """Paths inside the interpreter's prefix that can hold secrets: conda activation scripts."""
    return [os.path.join(prefix, "etc", "conda") for prefix in dict.fromkeys((sys.prefix, sys.base_prefix))
            if os.path.isdir(os.path.join(prefix, "etc", "conda"))]


if __name__ == "__main__":
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    if separator != 2 or separator == len(sys.argv) - 1:
        sys.exit("usage: python -m openvlharness.toolsmith.isolation SPEC -- COMMAND [ARGUMENT...]")
    launch(json.loads(sys.argv[1]), sys.argv[separator + 1:])
