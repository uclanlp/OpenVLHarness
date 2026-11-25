"""Run the generated tools of a catalog in a separate, resource-limited worker process.

The tool router serves generated tools through a CatalogWorker and never imports
them. The worker process loads a private copy of the catalog, taken when the router
started, and runs each call in its own thread. It inherits the router's environment
without credential-like variables, unless toolsmith.worker.environment grants them,
starts in a private directory, and runs with its address space capped and, unless
toolsmith.worker.write_files is set, without file writes. On Linux a watchdog in the router
measures the worker's processes every WATCH_SECONDS and stops the worker when they use more
memory together than toolsmith.worker.memory_gib, or burn more than IDLE_CPU_SECONDS of CPU
in a window with no call in flight. With toolsmith.worker.isolation
(the default) it runs confined by Linux namespaces (see isolation): it sees only the
interpreter, the packages, the catalog copy and the model caches, read-only (with
write_files, the caches take writes in a private layer), and none of the router's other
files or processes. It has a network namespace of its own with only loopback; with
toolsmith.worker.network (the default) its HTTP and HTTPS clients reach the ports in
toolsmith.worker.network_ports through the router's outbound proxy (see proxy), which
refuses this host, its local networks, private addresses and toolsmith.worker.deny_networks,
and toolsmith.worker.network: false leaves it no network at all.

A generated tool reaches other tools only through call_tool, which the worker passes
to the router process: calls to the other generated tools and to the base tools in
CALLABLE_BASE_TOOLS go through the router, and calls to the other router
tools are refused. A call that fails, crashes the worker or runs past the time limit
returns an error envelope. Only a crash, a call past the limit, a stop by the watchdog or
a message that breaks the protocol (one past the size limit, or one the router cannot
read) makes the router start a new worker for later calls: a worker with a call past the
limit takes no new calls and is stopped once each of its other calls has finished or
passed its own limit (RETIRE_GRACE_SECONDS later at the latest). A worker dies with the router: the
router starts every worker from one long-lived thread, and the worker (or, confined,
its launcher) asks Linux to kill it when that parent ends.

The router starts the worker as: python -m openvlharness.toolsmith.worker CATALOG LIMITS
"""
import asyncio
import atexit
import codecs
import concurrent.futures
import ctypes
import errno
import itertools
import json
import logging
import os
import platform
import queue
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from ..contracts import ServiceResult
from ..services.envelope import _build_error_payload, _build_success_payload, _normalize_result_payload
from ..services.tools.base import ToolArgumentError
from .catalog import DEFAULT_CATALOG, catalog_modules, fingerprint_files, load_catalog, tool_problem
from .errors import CatalogLoadError, WorkerStartError, brief, describe
from .proxy import PORTS as NETWORK_PORTS

logger = logging.getLogger(__name__)

# The base tools the generator is told it may call (generation.BASE_RESULTS).
CALLABLE_BASE_TOOLS = ("Text_Search_Tool", "Image_Search_Tool", "Webpage_Visit_Tool", "OCR_Tool", "Zoom_In_Tool")
MEMORY_GIB = 128
LOAD_SECONDS = 120
CLIENT_SHARE = 0.9  # default worker deadline, as a share of the agent's HTTP timeout
EXIT_SECONDS = 5
RETIRE_GRACE_SECONDS = 5  # how long a retired worker outlives the deadline of its last call in flight
MAX_MESSAGE_BYTES = 32 << 20  # a message from the worker; its JSON takes many times that once parsed
STDERR_BYTES = 8 << 20  # what one worker's tools may print to the router's stderr
STDERR_TOTAL_BYTES = 64 << 20  # what all the workers one CatalogWorker starts may print, together
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")  # control characters but tab and newline
CACHE_LAYER_GIB = 1  # default size of the private layer that takes writes to the model caches
WATCH_SECONDS = 0.5  # how often the router measures the processes of each worker
# A sample of a large process tree takes long; the watchdog then waits this many times as long
# before the next, about a tenth of a core per worker, but never more than WATCH_MAX_SECONDS: a tool
# must not stretch the time it goes unmeasured by starting processes. Trees whose samples take more
# than a tenth of that cost the router a larger share.
WATCH_BACKOFF = 10
WATCH_MAX_SECONDS = 5
# Work a tool leaves running after its call returns: when no call has been in flight for a whole
# window, the worker's processes may use at most IDLE_CPU_SECONDS of CPU time in it, or the worker
# is stopped. A window starts IDLE_SETTLE_SECONDS after the last call ended, once the thread pools
# of numerical libraries have stopped spinning.
IDLE_CPU_WINDOW_SECONDS = 10
IDLE_CPU_SECONDS = 2
IDLE_SETTLE_SECONDS = 1
MAX_FORWARDED = 256  # call_tool calls of one worker's tools the router runs at once; each holds a thread
OPEN_FILES = 4096  # file descriptors of each worker process
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# Settings that name a file or directory, and lists of directories, the worker resolves like the router.
PATH_SETTINGS = {"SERPER_API_KEY_FILE", "GOOGLE_APPLICATION_CREDENTIALS", "IMAGE_UPLOAD_LOCAL_DIR", "HF_HOME",
                 "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
                 "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "TMPDIR"}
PATH_LISTS = {"PYTHONPATH", "LD_LIBRARY_PATH"}
HOME_PREFIX = "openvlharness-toolsmith-"


def memory_limit_enforced():
    """Whether this platform enforces an address-space limit; macOS rejects RLIMIT_AS outright."""
    return sys.platform.startswith("linux")


def limit_resources(*, memory_bytes=None, file_bytes=None):
    """Cap the address space of the calling process and the size of the files it writes.

    None leaves a limit unchanged; a cap never exceeds the current hard limit, and
    the process cannot raise it again. The address-space cap applies only where
    memory_limit_enforced().
    """
    if not memory_limit_enforced():
        memory_bytes = None
    for kind, value in ((resource.RLIMIT_AS, memory_bytes), (resource.RLIMIT_FSIZE, file_bytes)):
        if value is not None:
            hard = resource.getrlimit(kind)[1]
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(kind, (value, value))


def worker_environment(granted=()):
    """Return the router's environment without credential-like variables, except granted names.

    A name looks like a credential under the local Python executor's rule (KEY, TOKEN,
    SECRET, PASSWORD or CREDENTIAL, or an AWS_ or GOOGLE_APPLICATION prefix); the rule
    goes by name only, so secrets under other names (DATABASE_URL, a password in a proxy
    URL) are inherited. The
    package the router runs comes first on PYTHONPATH, so the worker imports the same code.
    Relative paths in path settings, path lists and granted variables are resolved
    against the router's working directory, since the worker starts elsewhere.
    """
    from ..services.tools.python_executor_local.executor import _SECRET_ENV

    granted = set(granted)
    environment = {name: value for name, value in os.environ.items()
                   if name in granted or not _SECRET_ENV.search(name)}
    for name, value in environment.items():
        if name in PATH_LISTS:
            environment[name] = os.pathsep.join(os.path.abspath(entry) if entry else entry
                                                for entry in value.split(os.pathsep))
        elif value and not os.path.isabs(value) and (name in PATH_SETTINGS
                                                     or name in granted and os.path.exists(value)):
            environment[name] = os.path.abspath(value)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PACKAGE_ROOT), environment.get("PYTHONPATH"))))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


class MessageTooLarge(ValueError):
    """A message from the worker longer than the router accepts."""


def _size(count):
    """Describe a byte count, in MiB when it is whole MiB."""
    return f"{count >> 20} MiB" if count and not count % (1 << 20) else f"{count} bytes"


class _Channel:
    """Newline-delimited JSON messages over a pair of pipes.

    limit bounds the length of a received message, send_limit that of a sent one.
    """

    def __init__(self, reader, writer, *, limit=None, send_limit=None):
        self.reader, self.writer, self.limit, self.send_limit = reader, writer, limit, send_limit
        self.lock = threading.Lock()

    def send(self, message, **options):
        """Write one message; options go to json.dumps.

        Raises:
            MessageTooLarge: The message is longer than send_limit, which the router would refuse.
        """
        line = json.dumps(message, **options).encode("ascii") + b"\n"
        if self.send_limit and len(line) > self.send_limit:
            raise MessageTooLarge(f"the message is larger than the {_size(self.send_limit)} limit")
        with self.lock:
            self.writer.write(line)
            self.writer.flush()

    def receive(self):
        """Return the next message, or None when the other side has closed the pipe.

        Raises:
            ValueError: The message is longer than the limit, cut off or not JSON.
        """
        line = self.reader.readline(self.limit + 1) if self.limit else self.reader.readline()
        if not line:
            return None
        if not line.endswith(b"\n"):
            raise ValueError("message longer than the limit or cut off")
        return json.loads(line)


def _text(exc):
    """Return str(exc), or a placeholder when the exception cannot be printed."""
    try:
        return str(exc)
    except Exception:
        return f"<unprintable {type(exc).__name__}>"


def _valid_result(message):
    """Return whether a result message from the worker has the fields its outcome needs."""
    if message.get("ok") is True:
        return "result" in message and isinstance(message.get("meta"), dict)
    return (message.get("ok") is False and isinstance(message.get("message"), str)
            and isinstance(message.get("exception_type"), str) and isinstance(message.get("argument_error"), bool))


def _valid_call(message):
    """Return whether a call_tool message from the worker names a tool, arguments and a call ID."""
    return (type(message.get("id")) is int and isinstance(message.get("tool"), str)
            and isinstance(message.get("args"), dict) and isinstance(message.get("tool_call_id"), str))


def _failure(tool_name, tool_call_id, message, kind):
    """Return the 500 envelope for a call the worker could not complete."""
    return _build_error_payload(tool_name, tool_call_id, "tool_execution_error", message, {"exception_type": kind}), 500


def _exit_reason(returncode):
    """Describe how a worker process ended."""
    if returncode < 0:
        try:
            return f"the worker was stopped by {signal.Signals(-returncode).name}"
        except ValueError:
            return f"the worker was stopped by signal {-returncode}"
    return f"the worker exited with status {returncode}"


def _line(message):
    """Encode one protocol message.

    Raises:
        TypeError, ValueError: The message cannot be sent as JSON.
    """
    return json.dumps(message).encode("ascii") + b"\n"


_spawns = None
_spawns_lock = threading.Lock()


def _spawn(argv, **options):
    """Start a process from one long-lived thread and return its Popen.

    Linux sends the death signal a worker asks for when the thread that started it
    ends, so every worker starts from a thread that lasts as long as the router.
    """
    global _spawns
    with _spawns_lock:
        if _spawns is None:
            _spawns = queue.SimpleQueue()
            threading.Thread(target=_serve_spawns, args=(_spawns,), name="toolsmith-spawner", daemon=True).start()
        started = concurrent.futures.Future()
        _spawns.put((started, argv, options))
    return started.result()


def _serve_spawns(requests):
    """Start the processes _spawn asks for, forever."""
    while True:
        started, argv, options = requests.get()
        try:
            started.set_result(subprocess.Popen(argv, **options))
        except BaseException as exc:
            started.set_exception(exc)


def _die_with(parent):
    """Ask Linux to kill this process when parent, the process that started it, ends; exit if it already has."""
    if parent is None or not sys.platform.startswith("linux"):
        return
    import ctypes

    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        # Never run unwatched: a worker the router's death would not stop could outlive it.
        print(f"the ToolSmith worker cannot ask to die with the router: {os.strerror(ctypes.get_errno())}",
              file=sys.stderr, flush=True)
        os._exit(1)
    if os.getppid() != parent:
        os._exit(1)


class _OutputBudget:
    """What the workers of one CatalogWorker, of every generation, may still print to the router's stderr."""

    def __init__(self, total):
        self.total, self.left, self.lock, self.exhausted = total, total, threading.Lock(), False

    def take(self, wanted):
        """Grant up to wanted bytes; return (granted, whether this request found the budget used up first)."""
        with self.lock:
            granted = min(wanted, self.left)
            self.left -= granted
            first = granted < wanted and not self.exhausted
            self.exhausted = self.exhausted or granted < wanted
            return granted, first


def _copy_to_stderr(stream, pid, budget=None):
    """Copy a worker's stderr (what its tools print) to the router's stderr until the worker ends.

    Control characters other than tab and newline are dropped, so escape sequences cannot
    rewrite the log or the terminal. After STDERR_BYTES from this worker, or once the
    workers of its CatalogWorker have printed STDERR_TOTAL_BYTES together (budget), the
    rest is drained, not copied, and one line in the log says so.
    """
    decoder, copied = codecs.getincrementaldecoder("utf-8")("replace"), 0
    with stream:
        while chunk := stream.read1(1 << 16):
            if copied >= STDERR_BYTES:
                continue  # keep draining so the worker never blocks on a full pipe
            data = CONTROL.sub("", decoder.decode(chunk)).encode()[:STDERR_BYTES - copied]
            copied += len(data)
            granted, used_up = budget.take(len(data)) if budget else (len(data), False)
            if used_up:
                data = data[:granted] + (f"\n[ToolSmith: generated tools printed {budget.total} bytes; "
                                         "what they print from now on is dropped]\n").encode()
            elif granted < len(data):
                data = data[:granted]  # the router-wide budget ran out before; its note is in the log
            elif copied >= STDERR_BYTES:
                data += f"\n[ToolSmith worker {pid}: output past {STDERR_BYTES} bytes dropped]\n".encode()
            view = memoryview(data)
            try:
                while view:
                    view = view[os.write(2, view):]
            except OSError:
                pass  # the router's stderr is gone


_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


# A process whose shmem pages (tmpfs files it maps, memfd, SysV and shared anonymous memory) come
# to less than this is not read in detail: a tmpfs file it maps may count twice, by this much at most.
SHMEM_DETAIL_BYTES = 16 << 20
_HEX = b"0123456789abcdef"
KCMP_VM = 1
# kcmp's system call number where it is known for sure; elsewhere the vfork heuristic stands in.
_KCMP_SYSCALL = {"x86_64": 312, "amd64": 312, "aarch64": 272, "arm64": 272, "riscv64": 272}
# Where kcmp cannot tell, a child counts as vforked (its memory as its parent's) only this young: a
# vforked child execs within milliseconds, while a forked child that looks like one may hold memory of
# its own for as long as it likes.
VFORK_SECONDS = 1


# How often the threads of a process whose main thread has exited are listed, and how many of them
# are tried, until one lives long enough to be read.
TASK_LISTINGS = 50
TASKS_TRIED = 64


class _NoTask(OSError):
    """A process has threads, but none lived long enough to show its memory."""


def _task_ids(pid):
    """Yield IDs of threads that may show process pid's memory in /proc; nothing when it has none (a zombie).

    The first is pid itself. When its main thread has exited while others run on, the kernel
    shows the process's own entries as a zombie's, without memory, and other threads follow:
    the newest first, as the likeliest to live on, from fresh listings while the process
    still has any.

    Raises:
        OSError: The process is gone (_NoTask: not one of its threads served the caller).
    """
    for _ in range(TASK_LISTINGS):
        with open(f"/proc/{pid}/stat", "rb") as stream:
            fields = stream.read().rsplit(b")", 1)[1].split()
        if fields[0] not in (b"Z", b"X", b"x"):
            yield pid
        elif int(fields[17]) <= 1:  # num_threads: nothing but the zombie
            return
        else:
            yield from [task for task in sorted(map(int, os.listdir(f"/proc/{pid}/task")), reverse=True)
                        if task != pid][:TASKS_TRIED]
    raise _NoTask(f"no thread of process {pid} could be read")


def _live_task(pid):
    """Return the ID of a thread through which /proc shows the memory of process pid, or None when it has none.

    Raises:
        OSError, IndexError, ValueError: The process is gone, or none of its threads could be read.
    """
    for task in _task_ids(pid):
        if task == pid:
            return pid
        try:
            with open(f"/proc/{pid}/task/{task}/stat", "rb") as stream:
                if stream.read().rsplit(b")", 1)[1].split()[0] not in (b"Z", b"X", b"x"):
                    return task
        except (FileNotFoundError, ProcessLookupError):
            pass
    return None


def _read_memory(pid):
    """Return (memory, shmem) of a process as _process_usage describes them, from a thread that shows them.

    Raises:
        OSError, IndexError, ValueError: The process is gone, or none of its threads could be read.
    """
    rollups = os.path.exists("/proc/self/smaps_rollup")
    for task in _task_ids(pid):
        path = f"/proc/{pid}/task/{task}"
        try:
            if rollups:
                try:
                    with open(f"{path}/smaps_rollup", "rb") as stream:
                        rollup = dict(line.split()[:2] for line in stream if line.startswith((b"Pss:", b"Pss_Shmem:")))
                    if b"Pss:" in rollup:
                        shmem = int(rollup[b"Pss_Shmem:"]) * 1024 if b"Pss_Shmem:" in rollup else None
                        return int(rollup[b"Pss:"]) * 1024, shmem
                    continue
                except (FileNotFoundError, ProcessLookupError):
                    continue  # the thread has exited
                except OSError:
                    pass  # not readable to this process
            with open(f"{path}/statm", "rb") as stream:
                memory = int(stream.read().split()[1]) * _PAGE
            if memory:  # an exited thread shows none
                return memory, None
        except (FileNotFoundError, ProcessLookupError):
            continue
    return 0, None


def _process_usage(pid, memory=True, last=None):
    """Return (parent PID, memory in bytes, CPU seconds, shmem bytes) of one process, from /proc.

    Memory is the proportional set size (shared pages split among the processes that map
    them), and shmem bytes its part in shmem pages (files on a tmpfs, memfd, SysV and
    shared anonymous memory), or None where smaps_rollup has no Pss_Shmem line (older
    kernels). Where smaps_rollup is missing (Linux before 4.14), memory is the resident set
    size, which counts a page that processes share after fork once in each of them. A
    process whose main thread has exited is read through another thread. CPU time includes
    that of the children the process has reaped. With memory false, only the parent and
    the CPU time are read, and memory is 0. last keeps the memory read for each process
    (by PID and start time), which stands in when none of a process's threads lives long
    enough to be read.

    Raises:
        OSError, ValueError, IndexError: The process is gone or its files cannot be read.
    """
    with open(f"/proc/{pid}/stat", "rb") as stream:
        fields = stream.read().rsplit(b")", 1)[1].split()
    parent, ticks = int(fields[1]), sum(int(value) for value in fields[11:15])  # utime, stime, cutime, cstime
    if not memory:
        return parent, 0, ticks / _TICKS, None
    last = {} if last is None else last
    try:
        last[pid] = (fields[19], *_read_memory(pid))  # fields[19]: the process's start
    except _NoTask:
        if last.get(pid, (None,))[0] != fields[19]:
            last[pid] = (fields[19], 0, None)
    _, memory, shmem = last[pid]
    return parent, memory, ticks / _TICKS, shmem


_internal = None


def _internal_shmem():
    """The (major, minor) device of the kernel's own shmem mount, or None where it cannot be told.

    memfd files, SysV shared memory and shared anonymous mappings live there, not on any
    tmpfs that was mounted; a memfd shows which device it is.
    """
    global _internal
    if _internal is None:
        try:
            descriptor = os.memfd_create("toolsmith-meter")
            try:
                device = os.fstat(descriptor).st_dev
            finally:
                os.close(descriptor)
            _internal = (os.major(device), os.minor(device))
        except (AttributeError, OSError):
            _internal = False
    return _internal or None


def _host_tmpfs():
    """The (major, minor) devices of this process's own tmpfs mounts, or None where they cannot be read.

    A sandbox sees the host's files only through bind mounts, which keep their devices; its
    own tmpfs mounts are new file systems, never among these.
    """
    devices = set()
    try:
        with open("/proc/self/mountinfo", "rb") as stream:
            for line in stream:
                fields = line.split()
                if fields[fields.index(b"-") + 1] in (b"tmpfs", b"devtmpfs"):
                    major, minor = fields[2].split(b":")  # decimal here, hexadecimal in smaps
                    devices.add((int(major), int(minor)))
    except (OSError, ValueError, IndexError):
        return None
    return devices


def _not_sandbox_shmem():
    """The devices whose shmem pages are not files of a sandbox's own file systems, or None where unknown.

    They are the kernel's own shmem mount (memfd, SysV, shared anonymous memory) and the
    host's tmpfs mounts, whose files a sandbox may map read-only.
    """
    internal, host = _internal_shmem(), _host_tmpfs()
    return None if internal is None or host is None else host | {internal}


def _file_shmem(pid, shmem, excluded):
    """Return the part of a process's shmem Pss that maps files of a sandbox's own tmpfs mounts.

    That is all of it but the mappings on the devices in excluded (see _not_sandbox_shmem),
    read from the smaps of a live thread; with excluded None, nothing is.
    """
    if excluded is None:
        return 0
    other, current = 0, False
    try:
        task = _live_task(pid)
        if task is None:
            return 0
        with open(f"/proc/{pid}/task/{task}/smaps", "rb") as stream:
            for line in stream:
                if line[:1] in _HEX:  # a mapping's first line: addresses, permissions, offset, device, ...
                    major, minor = line.split()[3].split(b":")
                    current = (int(major, 16), int(minor, 16)) in excluded
                elif current and line.startswith(b"Pss:"):
                    other += int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return max(0, shmem - other)


_libc_handle = None
_kcmp_works = None  # False once the kernel, or a seccomp filter, refuses kcmp outright


def _same_memory(first, second):
    """Whether two processes share one address space (kcmp KCMP_VM), or None when this host will not tell."""
    global _libc_handle, _kcmp_works
    number = _KCMP_SYSCALL.get(platform.machine().lower())
    if number is None or _kcmp_works is False:
        return None
    if _libc_handle is None:
        _libc_handle = ctypes.CDLL(None, use_errno=True)
    result = _libc_handle.syscall(number, first, second, KCMP_VM, 0, 0)
    if result >= 0:
        _kcmp_works = True
        return result == 0
    failure = ctypes.get_errno()
    if failure == errno.ENOSYS or (failure == errno.EPERM and _kcmp_works is None):
        _kcmp_works = False  # no kcmp in this kernel, or a filter that refuses it from the first call on
    return None


def _waiting_on_vfork(parent, child):
    """Heuristic where kcmp cannot tell: child came from vfork or posix_spawn and has not exec'd yet.

    Its calling thread waits, in state D, until the child execs or exits, and the child's
    code and stack addresses are the parent's, which an exec changes. A forked child has
    them too, beside a thread that waits on another vfork, so a child older than
    VFORK_SECONDS is not taken for a vforked one. Neither is one whose addresses /proc
    hides (as 1, 1 and 0, for a process this one may not trace).
    """
    try:
        layout = []
        for pid in (parent, child):
            with open(f"/proc/{_live_task(pid)}/stat", "rb") as stream:
                fields = stream.read().rsplit(b")", 1)[1].split()
            layout.append(fields[23:26])  # startcode, endcode, startstack
        if layout[0] != layout[1] or layout[1][2] == b"0":
            return False
        with open(f"/proc/{child}/stat", "rb") as stream:  # the process's start, not a later thread's
            started = int(stream.read().rsplit(b")", 1)[1].split()[19]) / _TICKS
        if time.clock_gettime(time.CLOCK_BOOTTIME) - started > VFORK_SECONDS:
            return False
        for task in os.listdir(f"/proc/{parent}/task"):
            with open(f"/proc/{parent}/task/{task}/stat", "rb") as stream:
                if stream.read().rsplit(b")", 1)[1].split()[0] == b"D":
                    return True
    except (OSError, ValueError, IndexError, AttributeError):  # AttributeError: no CLOCK_BOOTTIME
        pass
    return False


def _task_lives(pid, task):
    """Whether a thread of process pid has not exited."""
    with open(f"/proc/{pid}/task/{task}/stat", "rb") as stream:
        return stream.read().rsplit(b")", 1)[1].split()[0] not in (b"Z", b"X", b"x")


def _shares_parent_memory(parent, child):
    """Whether child still uses parent's address space, as a vforked child does until it execs."""
    try:
        tasks = [_live_task(parent), _live_task(child)]
        if None in tasks:
            return False
        same = _same_memory(*tasks)
        if same is None:
            return _waiting_on_vfork(parent, child)
        # Threads that have exited since have no memory, which compares as the same.
        return same and all(_task_lives(pid, task) for pid, task in zip((parent, child), tasks))
    except (OSError, IndexError, ValueError):
        return False


def _child_pids(pid):
    """Return the PIDs of a process's children, from the children file of each of its threads."""
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return []
    found = []
    for task in tasks:
        try:
            with open(f"/proc/{pid}/task/{task}/children", "rb") as stream:
                found += [int(child) for child in stream.read().split()]
        except OSError:
            pass
    return found


def _children_map():
    """Map each PID to its children's PIDs, from one scan of /proc, for kernels without children files."""
    children = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat", "rb") as stream:
                    parent = int(stream.read().rsplit(b")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(parent, []).append(int(entry))
    return children


_has_children_files = None


def _children_files():
    """Whether this kernel lists a thread's children in /proc (CONFIG_PROC_CHILDREN); cached per process."""
    global _has_children_files
    if _has_children_files is None:
        _has_children_files = os.path.exists(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    return _has_children_files


def tree_usage(root, parent, *, mapped_files=False, last=None):
    """Return (memory in bytes, CPU seconds, mapped file bytes) of a process and all its descendants.

    Returns None when the process is gone. A process is counted only while its parent is
    the one the walk came from, so a PID reused by an unrelated process is not. A child
    that still uses its parent's address space (after vfork or posix_spawn, until it
    execs) adds its CPU time but not that memory again. With mapped_files, the third value
    is how much of the memory is tmpfs files the processes map (see _file_shmem), which
    a caller that counts those file systems itself takes off; otherwise it is 0. last is a
    dictionary that _process_usage keeps between walks; processes no longer met leave it. Confined,
    every process of the worker stays a descendant of its launcher (orphans are adopted
    by the namespace's init); without isolation, a process that detaches from the worker's
    tree is not counted.
    """
    memory = cpu = mapped = 0
    excluded = False  # read once a process maps enough shmem (see _not_sandbox_shmem)
    stack, seen = [(root, parent)], set()
    scanned = None if _children_files() else _children_map()
    while stack:
        pid, expected = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        # Asked before the memory is read: a vforked child that execs meanwhile has its own by then.
        own = pid == root or not _shares_parent_memory(expected, pid)
        try:
            actual, used, seconds, shmem = _process_usage(pid, memory=own, last=last)
        except (OSError, ValueError, IndexError):
            if pid == root:
                return None
            continue
        if actual != expected:
            if pid == root:
                return None
            continue
        cpu, memory = cpu + seconds, memory + used
        if mapped_files and shmem is not None and shmem >= SHMEM_DETAIL_BYTES:
            if excluded is False:
                excluded = _not_sandbox_shmem()
            mapped += _file_shmem(pid, shmem, excluded)
        children = _child_pids(pid) if scanned is None else scanned.get(pid, [])
        stack += [(child, pid) for child in children]
    for pid in set(last or ()) - seen:
        del last[pid]
    return memory, cpu, mapped


class TreeMeter:
    """Measure a process and its descendants, with what its sandbox's own memory-backed file systems hold.

    mounts are groups of paths, inside the sandbox whose init is root's first child (see
    isolation.private_mounts), each standing for one memory-backed file system: no process's
    memory shows a file written there, so its use counts too, read through the init's root
    once that is the sandbox's, at the group's first path that is not the host's own file
    system (an overlay that could not be mounted is a bind of the host's directory), decided
    when that root first appears. A file there that processes map is in both, and counts
    once. Shared memory whose pages no process currently has mapped in, and memory that only
    the kernel holds for a process, such as the data queued in its sockets and pipes, are
    among what is not counted.
    """

    def __init__(self, root, mounts=()):
        self.root, self.mounts, self.init = root, list(mounts), None
        self.paths = None  # the sandbox's own file systems, once its root has appeared
        self.last = {}  # see tree_usage

    def measure(self):
        """Return (memory in bytes, CPU seconds), or None once the root process is gone."""
        private = self.private_bytes()
        usage = tree_usage(self.root, os.getpid(), mapped_files=private > 0, last=self.last)
        if usage is None:
            return None
        memory, cpu, mapped = usage
        return memory - min(mapped, private) + private, cpu

    def private_bytes(self):
        """Bytes held in the sandbox's own memory-backed file systems; 0 without mounts or before the sandbox's root."""
        if not self.mounts:
            return 0
        if self.init is None:
            children = _child_pids(self.root) if _children_files() else _children_map().get(self.root)
            if not children:
                return 0
            self.init = children[0]
        try:
            if os.stat(f"/proc/{self.init}/root") == os.stat("/"):
                return 0  # not the sandbox's root (yet)
        except OSError:
            return 0
        used, seen = 0, set()
        if self.paths is None:
            self.paths = [path for group in self.mounts for path in self._own(group)]
        for path in self.paths:
            try:
                device = os.stat(f"/proc/{self.init}/root{path}").st_dev
                stats = os.statvfs(f"/proc/{self.init}/root{path}")
            except OSError:
                continue
            if device not in seen:  # a path that is part of another mount counts once
                seen.add(device)
                used += (stats.f_blocks - stats.f_bfree) * stats.f_frsize
        return used

    def _own(self, group):
        """The first path of group, as a list, whose file system in the sandbox is not the host's at that path."""
        for path in group:
            try:
                device = os.stat(f"/proc/{self.init}/root{path}").st_dev
            except OSError:
                continue
            try:
                if device == os.stat(path).st_dev:
                    continue
            except OSError:
                pass  # the host has no such path
            return [path]
        return []


def _gib(count):
    """Describe a byte count in GiB."""
    return f"{count / (1 << 30):.2f} GiB"


class _WorkerGone(RuntimeError):
    """The worker process has stopped and takes no calls."""


class _Worker:
    """One worker process: its pipes, the calls it is running and the thread that reads its messages."""

    def __init__(self, owner, files, environment, limits):
        """Write the catalog copy to a private directory and start the process."""
        self.owner = owner
        self.home = Path(tempfile.mkdtemp(prefix=HOME_PREFIX))
        self.catalog = self.home / "catalog"
        try:
            self.catalog.mkdir()
            (self.home / "root").mkdir()
            for name, data in files.items():
                (self.catalog / name).write_bytes(data)
            argv = [sys.executable, "-m", "openvlharness.toolsmith.worker", str(self.catalog), json.dumps(limits)]
            mounts = []  # the sandbox's own memory-backed file systems, which the watchdog counts
            if owner.isolation:
                from .isolation import command, private_mounts

                spec = owner.confinement(self.home)
                mounts = private_mounts(spec)
                argv = command(spec, argv)
            # stderr is a pipe the router copies to its own: a log file there would fall under the
            # worker's file-size limit, which drops prints and fails tools that flush them.
            self.process = _spawn(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  cwd=self.home, env=environment, start_new_session=True)
        except BaseException:
            shutil.rmtree(self.home, ignore_errors=True)
            raise
        threading.Thread(target=_copy_to_stderr, args=(self.process.stderr, self.process.pid, owner.stderr_budget),
                         daemon=True).start()
        self.meter = TreeMeter(self.process.pid, mounts)
        self.channel = _Channel(self.process.stdout, None, limit=limits["message_bytes"])
        # One thread writes to the worker, so a worker that stops reading blocks no caller.
        self.outbox = queue.SimpleQueue()
        threading.Thread(target=self._write, daemon=True).start()
        self.ready = concurrent.futures.Future()
        self.calls = {}
        self.deadlines = {}  # when each call in flight reaches its time limit (time.monotonic)
        self.abandoned = set()
        self.ids = itertools.count(1)
        self.lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.retiring = self.stopped = False
        self.reason = None
        self.exceeded = None  # (reason, error kind) once the watchdog stopped the worker for a limit
        self.started = 0  # calls submitted so far, which tells the watchdog that a call came and went
        self.nested = set()  # calls in flight that this worker's own call_tool made (see CatalogWorker._forward)
        self.checked = None  # self.started when the watchdog last found the worker idle
        self.quiet_since = time.monotonic()  # when the last call in flight ended
        self.tools = []
        self.forwarding = 0
        self.ended = threading.Event()  # set once the process has ended, which stops the watchdog
        self.reader = threading.Thread(target=self._read, name=f"toolsmith-worker-{self.process.pid}", daemon=True)
        self.reader.start()
        if os.path.isdir(f"/proc/{self.process.pid}"):  # the watchdog reads /proc, which Linux has
            threading.Thread(target=self._watch, name=f"toolsmith-watchdog-{self.process.pid}", daemon=True).start()

    def submit(self, tool_name, args, *, nested=False):
        """Send one call and return its ID and the future of the worker's answer.

        A nested call, one this worker's own tools made through call_tool, does not count
        as work for the idle-CPU rule: otherwise a tool could keep the worker looking busy
        by calling another generated tool in a loop.

        Raises:
            _WorkerGone: The worker has stopped.
        """
        with self.lock:
            if self.stopped or self.exceeded:
                raise _WorkerGone(self.reason or self.exceeded[0])
            request = next(self.ids)
            answer = self.calls[request] = concurrent.futures.Future()
            self.deadlines[request] = time.monotonic() + self.owner.timeout
            if nested:
                self.nested.add(request)
            else:
                self.started += 1
        try:
            self.send({"type": "execute", "id": request, "tool": tool_name, "args": args})
        except (TypeError, ValueError):
            with self.lock:
                self.calls.pop(request, None)
                self.deadlines.pop(request, None)
                self._settle(request)
            raise
        return request, answer

    def _busy(self):
        """Whether a call other than a nested one is in flight; the caller holds self.lock."""
        return any(request not in self.nested for request in self.calls)

    def _settle(self, request):
        """Note that request ended, and when the last call in flight ended; the caller holds self.lock."""
        if request in self.nested:
            self.nested.discard(request)
        elif not self._busy():
            self.quiet_since = time.monotonic()

    def send(self, message):
        """Queue a message for the worker; it is dropped if the worker has stopped.

        Raises:
            TypeError, ValueError: The message cannot be sent as JSON.
        """
        self.outbox.put(_line(message))

    def _write(self):
        """Write queued messages to the worker until it stops."""
        while (line := self.outbox.get()) is not None:
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except (OSError, ValueError):
                return  # the worker has stopped; its calls fail when the reader sees it end

    def abandon(self, request):
        """Stop waiting for a call past its time limit; return False when its answer has already arrived."""
        with self.lock:
            self.deadlines.pop(request, None)
            if self.calls.pop(request, None) is None:
                return False
            self.abandoned.add(request)
            self._settle(request)
            return True

    def retire(self):
        """Take no new calls; stop the process once each call in flight has finished or passed its deadline."""
        with self.lock:
            self.retiring = True
        self._stop_when_done()

    def _stop_when_done(self):
        """Kill a retired worker now if it runs no call, else check again RETIRE_GRACE_SECONDS after the last deadline.

        The calls past their deadline were abandoned by their callers, and each answer
        or abandonment of the others kills an idle retired worker at once; the timer
        only bounds a worker whose remaining calls neither answer nor time out.
        """
        with self.lock:
            latest = max(self.deadlines.values(), default=None) if self.calls else None
        wait = 0 if latest is None else latest + RETIRE_GRACE_SECONDS - time.monotonic()
        if wait <= 0:
            self.kill()
        else:
            timer = threading.Timer(wait, self._stop_when_done)
            timer.daemon = True
            timer.start()

    def kill(self):
        """Kill the process and anything it started, unless it has already been reaped."""
        with self.process_lock:
            if self.process.returncode is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    self.process.kill()
                except OSError:
                    pass

    def _forward(self, message):
        """Run one call_tool through the router, counting it while it runs."""
        try:
            self.owner._forward(self, message)
        finally:
            with self.lock:
                self.forwarding -= 1

    def _read(self):
        """Handle the worker's messages until it stops, then fail the calls still waiting."""
        reason = None
        try:
            while True:
                message = self.channel.receive()
                if message is None:
                    break
                self._handle(message)
        except Exception as exc:
            reason = f"the worker sent an invalid message ({type(exc).__name__}: {brief(_text(exc))})"
            logger.error("ToolSmith worker %s: %s", self.process.pid, reason)
        if reason is None:
            # The isolation launcher outlives the worker briefly while it cleans up; keep its exit status.
            try:
                self.process.wait(timeout=EXIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        self.kill()
        with self.process_lock:
            returncode = self.process.wait()
        self.ended.set()
        with self.lock:
            self.stopped = True
            kind = "WorkerError"
            if reason is None and self.exceeded:
                reason, kind = self.exceeded  # the watchdog stopped it: say which limit it passed
            self.reason = reason or _exit_reason(returncode)
            calls, self.calls = self.calls, {}
            self.deadlines.clear()
            self.abandoned.clear()
            self.nested.clear()
        for answer in calls.values():
            answer.set_result({"failure": self.reason, "kind": kind})
        if not self.ready.done():
            self.ready.set_result({"type": "ready", "error": {"type": "RuntimeError", "message": self.reason}})
        self.outbox.put(None)
        shutil.rmtree(self.home, ignore_errors=True)
        self.owner._stopped(self)

    def _watch(self):
        """Measure the worker's processes every WATCH_SECONDS and stop the worker past a limit.

        The memory limit holds for the launcher and all its descendants together, with what
        the sandbox's own memory-backed file systems hold (see TreeMeter); the CPU limit
        applies only to a whole IDLE_CPU_WINDOW_SECONDS window with no call in flight but
        nested ones, which catches threads or processes that a call left running. A sample
        that takes long (thousands of processes) spaces out the next ones (WATCH_BACKOFF):
        for very large trees the wait between samples reaches WATCH_MAX_SECONDS.
        """
        window, wait = None, WATCH_SECONDS
        while not self.ended.wait(wait):
            began = time.monotonic()
            usage = self.meter.measure()
            if usage is None:
                continue  # the process has ended; the reader is about to notice
            memory, cpu = usage
            wait = min(WATCH_MAX_SECONDS, max(WATCH_SECONDS, WATCH_BACKOFF * (time.monotonic() - began)))
            limit = self.owner.memory_limit
            if limit and memory > limit:
                self._exceed(f"its processes and in-memory files used {_gib(memory)} of memory together, more than "
                             f"toolsmith.worker.memory_gib allows ({_gib(limit)})", "MemoryLimitExceeded")
                return
            now = time.monotonic()
            with self.lock:
                idle = self.ready.done() and not self._busy() and not self.retiring and not self.stopped
                settled = idle and now - self.quiet_since >= IDLE_SETTLE_SECONDS
                started = self.started
            if not settled:
                window = None
            elif window is None or window[2] != started:  # a call came and went between two samples
                window = (now, cpu, started)
            elif now - window[0] >= IDLE_CPU_WINDOW_SECONDS:
                used = cpu - window[1]
                if used > IDLE_CPU_SECONDS:
                    self.checked = started
                    if self._exceed(f"its processes used {used:.1f} CPU-seconds in {IDLE_CPU_WINDOW_SECONDS:g} seconds "
                                    f"with no call in flight, more than {IDLE_CPU_SECONDS:g}: a tool left work running "
                                    "after its call returned", "CPULimitExceeded"):
                        return
                window = (now, cpu, started)

    def _exceed(self, reason, kind):
        """Stop a worker that passed a limit: its calls in flight fail with reason, and the next call starts another.

        For the idle-CPU rule the worker must still be idle as the watchdog found it: a call
        that came in since means the work may be its. Returns whether the worker is stopped.
        """
        with self.lock:
            if self.stopped or self.exceeded:
                return False
            if kind == "CPULimitExceeded" and (self._busy() or self.started != self.checked):
                return False
            self.exceeded, self.retiring = (reason, kind), True
        logger.warning("ToolSmith worker %s: %s; stopping it", self.process.pid, reason)
        self.owner._replace(self)
        self.kill()
        return True

    def _handle(self, message):
        """Dispatch one message from the worker; a message that breaks the protocol raises ValueError."""
        kind = message.get("type") if isinstance(message, dict) else None
        if kind == "ready" and not self.ready.done():
            with self.lock:
                self.quiet_since = time.monotonic()  # loading the catalog is not idle work
            self.ready.set_result(message)
        elif kind == "result":
            if not _valid_result(message):
                raise ValueError("malformed result")
            with self.lock:
                answer = self.calls.pop(message.get("id"), None)
                self.deadlines.pop(message.get("id"), None)
                late = answer is None and message.get("id") in self.abandoned
                self.abandoned.discard(message.get("id"))
                idle = self.retiring and not self.calls
                if answer is not None:
                    self._settle(message.get("id"))
            if answer is None and not late:
                raise ValueError(f"result for unknown call {brief(repr(message.get('id')), 40)}")
            if answer is not None:
                answer.set_result(message)
            if idle:
                self.kill()
        elif kind == "call":
            if not _valid_call(message):
                raise ValueError("malformed call")
            with self.lock:
                admitted = self.forwarding < MAX_FORWARDED
                self.forwarding += admitted
            if admitted:
                threading.Thread(target=self._forward, args=(message,), daemon=True).start()
            else:
                self.send({"type": "reply", "id": message["id"], "payload": _build_error_payload(
                    message["tool"], message["tool_call_id"], "too_many_calls",
                    f"more than {MAX_FORWARDED} calls from generated tools are running")})
        else:
            raise ValueError(f"unexpected message type {brief(repr(kind), 40)}")


class CatalogWorker:
    """Serve the generated tools of a catalog from a worker process; the router's handle on them."""

    def __init__(self, directory=None, *, client, base_tools=(), environment=(), timeout_seconds=600,
                 memory_gib=MEMORY_GIB, write_files=False, isolation=True, network=True, network_ports=NETWORK_PORTS,
                 deny_networks=(), report_environment=True, cache_layer_gib=CACHE_LAYER_GIB):
        """Copy the catalog, start the first worker and read the tools it loaded.

        Args:
            directory: Catalog directory; defaults to the packaged catalog.
            client: ToolClient for calls from generated tools; it reaches the router itself.
            base_tools: Names of the router's base tools.
            environment: Credential-like variables the worker may inherit.
            timeout_seconds: Wall-clock limit of one call.
            memory_gib: Memory limit of a worker's processes together, with its private
                memory-backed file systems, in GiB, which a watchdog checks every
                WATCH_SECONDS on Linux; each process's address space is capped at it as
                well. It stops runaway tools, not code written to evade it.
            write_files: Let the worker write files, for example a local image upload;
                each file it writes is capped at memory_gib.
            isolation: Confine the worker with Linux namespaces; without it, generated
                tools can read every file the router can.
            network: Let a confined worker use the network, through the outbound proxy.
            network_ports: The destination ports the proxy serves.
            deny_networks: Networks (CIDR strings) the proxy refuses besides this host's own.
            report_environment: Warn about the credential-like variables the worker does not inherit.
            cache_layer_gib: With write_files and isolation, the size in GiB of the private
                in-memory layer that takes writes to the model caches.

        Raises:
            CatalogLoadError: The catalog cannot be loaded (a ValueError).
            WorkerStartError: The worker or its outbound proxy does not start, or isolation
                is requested on a host that does not allow it (a RuntimeError).
        """
        self.directory = Path(directory or DEFAULT_CATALOG)
        self.files = {path.name: path.read_bytes() for path in catalog_modules(self.directory)}
        self.fingerprint = fingerprint_files(self.files.items())  # what evaluations check the router serves
        self.client = client
        self.refused = set(base_tools) - set(CALLABLE_BASE_TOOLS)
        self.timeout = timeout_seconds
        self.memory_limit = int(memory_gib * (1 << 30))  # of all a worker's processes, which the watchdog checks
        self.cache_layer_gib = cache_layer_gib
        self.stderr_budget = _OutputBudget(STDERR_TOTAL_BYTES)  # shared by every worker this object starts
        # The router tells the worker its message limit, so the two never disagree.
        # With file writes, no one file (a memfd included, whose unmapped pages no process's memory shows) outgrows
        # the memory limit; without them nothing can be written.
        self.write_files = write_files
        self.limits = {"memory_bytes": int(memory_gib * (1 << 30)),
                       "file_bytes": int(memory_gib * (1 << 30)) if write_files else 0,
                       "message_bytes": MAX_MESSAGE_BYTES}
        if not memory_limit_enforced():
            self.limits["memory_bytes"] = None
            logger.warning("%s does not enforce address-space limits: the generated-tool worker runs without "
                           "toolsmith.worker.memory_gib.", sys.platform)
        self.environment = worker_environment(environment)
        withheld = sorted(set(os.environ) - set(self.environment))
        if isolation:
            # The router's proxies are unreachable from the sandbox, and their URLs can hold passwords.
            self.environment = {name: value for name, value in self.environment.items()
                                if not name.lower().endswith("_proxy")}
        if isolation and not network:
            # Without a network, Hugging Face libraries load cached models at once instead of retrying the Hub.
            self.environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.granted = set(environment)
        self.isolation, self.network = isolation, network
        # A confined worker dies with its launcher's namespace; an unconfined one watches the router itself.
        self.limits["parent"] = None if isolation else os.getpid()
        self.proxy = None
        if isolation:
            from .isolation import available

            confinable, reason = available()
            if not confinable:
                raise WorkerStartError(f"Generated tools run confined by Linux user namespaces, which this host does "
                                       f"not allow ({reason}). Set toolsmith.worker.isolation: false to run them "
                                       "without it.")
            if network:
                from .proxy import OutboundProxy

                # The confined worker's only way out: a socket in a directory only the router's user can open.
                try:
                    self.proxy = OutboundProxy(ports=network_ports, deny=deny_networks)
                except (OSError, RuntimeError) as exc:  # no socket directory or socket, or no thread for it
                    raise WorkerStartError(f"the ToolSmith outbound proxy cannot start: {exc}") from exc
                self.limits["proxy"] = self.proxy.path
        else:
            logger.warning("toolsmith.worker.isolation is off: generated tools can read every file the router can "
                           "read and reach the network.")
        if withheld and report_environment:
            logger.warning("Generated tools run without these router environment variables: %s. Grant any they "
                           "read with toolsmith.worker.environment.", ", ".join(withheld))
        self.lock = threading.Lock()
        self.start_lock = threading.Lock()
        self.workers = set()
        self.nested = {}  # tool_call_id -> count of the call_tool calls to generated tools now forwarded
        self.closed = False
        self.tools = self.current = None
        try:
            self.current = self._start()
        except BaseException:
            self.close()  # and its outbound proxy
            raise
        self.tools = {tool["tool_name"]: tool for tool in self.current.tools}
        atexit.register(self.close)

    @classmethod
    def from_config(cls, config, *, client, base_tools):
        """Build the worker for an agent-system configuration with a toolsmith section."""
        toolsmith = config["toolsmith"]
        settings = toolsmith.get("worker") or {}
        client_timeout = config["services"]["client"].get("timeout_seconds", 600)
        # The worker deadline starts after dispatch; ending it earlier than the agent's HTTP read timeout
        # lets the structured timeout arrive before the client gives up and retries the call.
        timeout = settings.get("timeout_seconds", round(CLIENT_SHARE * client_timeout, 3))
        if timeout >= client_timeout:
            logger.warning("toolsmith.worker.timeout_seconds (%s) is not below services.client.timeout_seconds (%s): "
                           "the agent may time out and retry a generated tool before its error arrives.",
                           timeout, client_timeout)
        return cls(toolsmith.get("catalog"), client=client, base_tools=base_tools,
                   environment=settings.get("environment", ()), timeout_seconds=timeout,
                   memory_gib=settings.get("memory_gib", MEMORY_GIB), write_files=settings.get("write_files", False),
                   isolation=settings.get("isolation", True), network=settings.get("network", True),
                   cache_layer_gib=settings.get("cache_layer_gib", CACHE_LAYER_GIB),
                   network_ports=settings.get("network_ports", NETWORK_PORTS),
                   deny_networks=settings.get("deny_networks", ()))

    def execute(self, tool_name, args, *, tool_call_id):
        """Run one call in the worker and return the router envelope and its HTTP status."""
        try:
            worker, request, answer = self._submit(tool_name, args, tool_call_id)
        except (ValueError, RuntimeError, OSError) as exc:
            return _failure(tool_name, tool_call_id, f"the generated-tool worker is unavailable: {exc}", "WorkerError")
        try:
            outcome = answer.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            outcome = None
        return self._finish(worker, request, answer, outcome, tool_name, tool_call_id)

    async def execute_async(self, tool_name, args, *, tool_call_id):
        """Run one call like execute, awaiting the answer on the event loop instead of in a thread.

        The router uses this, so a generated tool waiting for the calls it makes back
        into the router holds none of the thread-pool workers those calls need.
        """
        loop = asyncio.get_running_loop()
        try:
            worker, request, answer = await loop.run_in_executor(None, self._submit, tool_name, args, tool_call_id)
        except (ValueError, RuntimeError, OSError) as exc:
            return _failure(tool_name, tool_call_id, f"the generated-tool worker is unavailable: {exc}", "WorkerError")
        # asyncio.wait, unlike wait_for, does not cancel the future the worker's reader resolves.
        done, _ = await asyncio.wait({asyncio.wrap_future(answer)}, timeout=self.timeout)
        return self._finish(worker, request, answer, answer.result() if done else None, tool_name, tool_call_id)

    def close(self):
        """Stop every worker and the outbound proxy; later calls fail."""
        atexit.unregister(self.close)  # the exit hook would keep a closed router, and its catalog bytes, alive
        with self.lock:
            self.closed = True
            workers, self.current = list(self.workers), None
        for worker in workers:
            worker.kill()
        for worker in workers:
            worker.reader.join(timeout=5)
        if self.proxy is not None:
            self.proxy.close()

    def confinement(self, home):
        """The isolation spec of a worker whose private directory is home."""
        from .isolation import python_masks, system_paths

        environment, expand = self.environment, os.path.expanduser
        # Where Hugging Face and torch put their caches unless HF_HOME or TORCH_HOME says otherwise.
        xdg = environment.get("XDG_CACHE_HOME") or expand("~/.cache")
        caches = [environment.get("HF_HOME") or os.path.join(xdg, "huggingface"), environment.get("HF_HUB_CACHE"),
                  environment.get("HUGGINGFACE_HUB_CACHE"), environment.get("TRANSFORMERS_CACHE"),
                  environment.get("TORCH_HOME") or os.path.join(xdg, "torch")]
        caches = list(dict.fromkeys(path for path in caches if path and os.path.isdir(path)))
        masks = [os.path.join(cache, name) for cache in caches for name in ("token", "stored_tokens")]
        # A cache inside another is covered by the outer mount.
        caches = [path for path in caches if not any(path.startswith(other.rstrip("/") + "/") for other in caches)]
        package_root = os.path.realpath(PACKAGE_ROOT)
        read = system_paths() + [str(home / "catalog")]
        if self.proxy is not None:
            read.append(os.path.dirname(self.proxy.path))  # connecting to a socket needs no write access
        read += [entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
                 if entry and os.path.isdir(entry) and os.path.realpath(entry) != package_root]
        # Certificate bundles, and files that granted variables name (a key file, service-account credentials).
        read += [environment[name] for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
                                                *sorted(self.granted))
                 if environment.get(name) and os.path.isabs(environment[name]) and os.path.exists(environment[name])]
        write, overlay = [], []
        if self.write_files:
            # The host's caches stay read-only; writes and downloads go to the worker's private layer.
            overlay = caches
            if environment.get("IMAGE_UPLOAD_PROVIDER", "").lower() == "local":
                # The local image host serves this directory, so uploads must land in the host's copy.
                uploads = environment.get("IMAGE_UPLOAD_LOCAL_DIR") or os.path.join(tempfile.gettempdir(),
                                                                                     "openvlharness-uploads")
                os.makedirs(uploads, mode=0o700, exist_ok=True)
                write.append(uploads)
        else:
            read += caches
        temporary = ["/tmp", "/var/tmp", expand("~")] + ([environment["TMPDIR"]] if environment.get("TMPDIR") else [])
        return {"root": str(home / "root"), "read": read, "write": write, "overlay": overlay, "tmp": temporary,
                "mask": masks + python_masks(), "cwd": str(home), "cleanup": str(home),
                "parent": os.getpid(), "layer_bytes": int(self.cache_layer_gib * (1 << 30))}

    def _submit(self, tool_name, args, tool_call_id=None):
        """Send one call to the worker taking new calls; return the worker, the call ID and its answer.

        A worker that stopped, or that the watchdog stopped, between being chosen and
        taking the call is replaced once; the call was not sent to it. A call whose ID is
        that of a call_tool a worker's tool is making now is nested (see _Worker.submit).
        """
        with self.lock:
            nested = tool_call_id in self.nested
        worker = self._worker()
        try:
            request, answer = worker.submit(tool_name, args, nested=nested)
        except _WorkerGone:
            worker = self._worker()
            request, answer = worker.submit(tool_name, args, nested=nested)
        return worker, request, answer

    def _finish(self, worker, request, answer, outcome, tool_name, tool_call_id):
        """Turn the worker's answer into the router envelope; outcome None means the time limit passed."""
        if outcome is None:
            if worker.abandon(request):
                self._retire(worker)
                return _failure(tool_name, tool_call_id, f"{tool_name} did not finish within {self.timeout:g} seconds",
                                "TimeoutError")
            outcome = answer.result()
        if "failure" in outcome:
            return _failure(tool_name, tool_call_id, "the generated-tool worker stopped while "
                            f"{tool_name} was running: {outcome['failure']}", outcome.get("kind", "WorkerError"))
        if outcome["ok"]:
            return _build_success_payload(tool_name, tool_call_id, outcome["result"], outcome["meta"]), 200
        return (_build_error_payload(tool_name, tool_call_id, "tool_execution_error", outcome["message"],
                                     {"exception_type": outcome["exception_type"]}),
                400 if outcome["argument_error"] else 500)

    def _start(self):
        """Start a worker and wait until it has loaded the catalog.

        Raises:
            CatalogLoadError: The catalog does not load.
            WorkerStartError: The worker does not start, load or report its tools.
        """
        try:
            worker = _Worker(self, self.files, self.environment, self.limits)
        except OSError as exc:
            raise WorkerStartError(f"the ToolSmith worker cannot start: {exc}") from exc
        with self.lock:
            self.workers.add(worker)
        try:
            ready = worker.ready.result(timeout=LOAD_SECONDS)
        except concurrent.futures.TimeoutError:
            self._discard(worker)
            raise WorkerStartError(f"the ToolSmith worker did not load the catalog within {LOAD_SECONDS} seconds")
        try:
            error = ready.get("error")
            tools = ready.get("tools")
            if error is None and not (isinstance(tools, list) and all(
                    isinstance(tool, dict) and tool_problem(tool) is None for tool in tools)):
                error = {"type": "RuntimeError", "message": "the worker reported no valid tool list"}
            if error is None and self.tools is not None and [tool["tool_name"] for tool in tools] != list(self.tools):
                error = {"type": "RuntimeError", "message": "a restarted worker loaded different tools"}
        except BaseException:  # whatever the check meets, the worker it started does not outlive it
            self._discard(worker)
            raise
        if error is not None:
            self._discard(worker)
            message = brief(str(error.get("message")).replace(str(worker.catalog), str(self.directory)), 2000)
            raise (CatalogLoadError if error.get("type") in ("CatalogLoadError", "ValueError") else
                   WorkerStartError)(message)
        worker.tools = tools
        return worker

    @staticmethod
    def _discard(worker):
        """Stop a worker that did not start and remove its private directory before the caller gives up."""
        worker.kill()
        worker.reader.join(timeout=EXIT_SECONDS + 5)
        shutil.rmtree(worker.home, ignore_errors=True)

    def _worker(self):
        """Return the worker that takes new calls, starting one if the last has stopped or been retired."""
        with self.start_lock:
            with self.lock:
                if self.closed:
                    raise RuntimeError("the tool router is shutting down")
                worker = self.current
            if worker is not None and not (worker.stopped or worker.retiring):
                return worker
            worker = self._start()
            with self.lock:
                self.current = worker
            return worker

    def _retire(self, worker):
        """Send no new calls to a worker whose call ran past the limit; it stops once its other calls finish."""
        with self.lock:
            if self.current is worker:
                self.current = None
        worker.retire()

    def _stopped(self, worker):
        """Forget a worker whose process has ended."""
        with self.lock:
            self.workers.discard(worker)
            if self.current is worker:
                self.current = None

    def _replace(self, worker):
        """Send no new calls to a worker the watchdog is stopping; the next call starts another."""
        with self.lock:
            if self.current is worker:
                self.current = None

    def _forward(self, worker, message):
        """Answer one call_tool from the worker: run it through the router, unless the tool is refused."""
        name, call_id = message["tool"], message["tool_call_id"]
        if name in self.refused:
            reply = {"payload": _build_error_payload(name, call_id, "tool_not_allowed",
                                                     f"Generated tools cannot call {name}")}
        else:
            # A call to a generated tool comes back to this object as a nested call (see _submit).
            generated = self.tools is not None and name in self.tools
            if generated:
                with self.lock:
                    self.nested[call_id] = self.nested.get(call_id, 0) + 1
            try:
                reply = {"payload": self.client.execute(name, message["args"], tool_call_id=call_id)}
            except Exception as exc:
                reply = {"error": {"module": type(exc).__module__, "type": type(exc).__name__, "message": _text(exc),
                                   "status_code": getattr(exc, "status_code", None)}}
            finally:
                if generated:
                    with self.lock:
                        self.nested[call_id] -= 1
                        if not self.nested[call_id]:
                            del self.nested[call_id]
        try:
            worker.send({"type": "reply", "id": message["id"], **reply})
        except (TypeError, ValueError) as exc:
            worker.send({"type": "reply", "id": message["id"], "error": {
                "module": "builtins", "type": "RuntimeError", "message": f"{name}: the reply is not JSON: {exc}"}})


def catalog_tools(directory=None, *, settings=None):
    """Return each catalog tool's name, description and input schema (see catalog_snapshot)."""
    return catalog_snapshot(directory, settings=settings)[0]


def catalog_snapshot(directory=None, *, settings=None):
    """Return each catalog tool's name, description and input schema without running catalog code here.

    A worker (confined as settings, a toolsmith.worker mapping, say) loads the catalog,
    reports its tools and stops. Reading metadata needs no network, so the worker gets no
    outbound proxy, as Gate 1 runs do: confined, it has no network and reaches no network
    service while catalog code runs at import. Without isolation it shares the host's
    network.

    Returns:
        (tools, fingerprint): a mapping of tool names to objects with tool_name,
        tool_description and input_schema, and the digest of the files they came from.

    Raises:
        CatalogLoadError, WorkerStartError: As CatalogWorker.
    """
    settings = {**(settings or {}), "network": False}
    worker = CatalogWorker(directory, client=_NoCalls(), environment=settings.get("environment", ()),
                           memory_gib=settings.get("memory_gib", MEMORY_GIB),
                           write_files=settings.get("write_files", False), isolation=settings.get("isolation", True),
                           network=settings.get("network", True), report_environment=False,
                           cache_layer_gib=settings.get("cache_layer_gib", CACHE_LAYER_GIB))
    try:
        return {name: SimpleNamespace(**metadata) for name, metadata in worker.tools.items()}, worker.fingerprint
    finally:
        worker.close()


class _NoCalls:
    """Client for a worker that only reports its tools: a call made while the catalog loads fails."""

    def execute(self, tool_name, args, *, tool_call_id):
        return _build_error_payload(tool_name, tool_call_id, "unavailable",
                                    "tools cannot be called while the catalog loads")


def _rebuild(error):
    """Recreate an exception the router's client raised, so call_tool fails as it does in the router."""
    message = str(error.get("message", ""))
    module, name = error.get("module"), error.get("type")
    if module == "openvlharness.services.http" and name == "HTTPStatusError":
        from ..services.http import HTTPStatusError

        return HTTPStatusError(message, error.get("status_code"))
    if module == "requests.exceptions":
        import requests

        kind = getattr(requests.exceptions, str(name), None)
        if isinstance(kind, type) and issubclass(kind, requests.exceptions.RequestException):
            return kind(message)
    return RuntimeError(message)


class _Router:
    """The client generated tools get in the worker: each call goes to the router process."""

    def __init__(self, channel):
        self.channel = channel
        self.ids = itertools.count(1)
        self.waiting = {}
        self.lock = threading.Lock()

    def execute(self, tool_name, args, *, tool_call_id):
        """Return the router's execute envelope for one call, as ToolClient.execute does.

        Raises:
            TypeError, ValueError: The call cannot be sent as JSON.
            Exception: The router's client failed; the same exception class for
                requests and HTTP status errors, otherwise RuntimeError.
        """
        if not (isinstance(tool_name, str) and isinstance(args, dict) and isinstance(tool_call_id, str)):
            raise TypeError("call_tool needs a tool name and keyword arguments")
        with self.lock:
            call = next(self.ids)
            answer = self.waiting[call] = concurrent.futures.Future()
        try:
            self.channel.send({"type": "call", "id": call, "tool": tool_name, "args": args,
                               "tool_call_id": tool_call_id}, allow_nan=False)
        except BaseException:
            with self.lock:
                self.waiting.pop(call, None)
            raise
        reply = answer.result()
        if "error" in reply:
            raise _rebuild(reply["error"])
        return reply["payload"]

    def answer(self, message):
        """Deliver the router's reply to the waiting call."""
        with self.lock:
            answer = self.waiting.pop(message.get("id"), None)
        if answer is not None:
            answer.set_result(message)


def _run(tools, message, channel):
    """Run one call and send its result; the fields mirror the router's envelope for an in-process tool."""
    reply = {"type": "result", "id": message["id"]}
    try:
        result = tools[message["tool"]].execute(**message["args"])
        extra = {}
        if isinstance(result, ServiceResult):
            extra, result = result.metadata, result.value
        result, meta = _normalize_result_payload(result)
        meta.update(extra)
        reply.update(ok=True, result=result, meta=meta)
    except BaseException as exc:
        reply.update(ok=False, message=_text(exc), exception_type=type(exc).__name__,
                     argument_error=isinstance(exc, ToolArgumentError))
    try:
        channel.send(reply)
    except MessageTooLarge:  # this call fails; the worker and its other calls go on
        channel.send({"type": "result", "id": message["id"], "ok": False, "exception_type": "MessageTooLarge",
                      "message": f"the result of {message['tool']} is larger than the {_size(channel.send_limit)} "
                                 "limit", "argument_error": False})
    except (TypeError, ValueError) as exc:
        channel.send({"type": "result", "id": message["id"], "ok": False, "message": f"the result is not JSON: {exc}",
                      "exception_type": type(exc).__name__, "argument_error": False})


def _use_outbound_proxy(path):
    """Send the tools' HTTP and HTTPS traffic to the router's outbound proxy at path, before the catalog loads."""
    from .proxy import forward

    url = f"http://127.0.0.1:{forward(path)}"
    for name in [name for name in os.environ if name.lower().endswith("_proxy")]:  # no other proxy is reachable
        del os.environ[name]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[name] = url
    for name in ("NO_PROXY", "no_proxy"):  # the sandbox's own loopback, where a tool may start a server
        os.environ[name] = "localhost,127.0.0.1,::1"


def _prefer_oom_kill():
    """Ask Linux to stop the worker before other processes when memory runs out; elsewhere do nothing."""
    try:
        with open("/proc/self/oom_score_adj", "w") as stream:
            stream.write("1000")
    except OSError:
        pass


def main(directory, limits):
    """Load the catalog, report its tools, then run calls until the router closes the pipe."""
    channel = _Channel(os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb"))
    # Messages use private copies of the pipes; prints from tools go to stderr, which the router copies to its own.
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    limits = json.loads(limits)
    _die_with(limits.pop("parent", None))
    channel.send_limit = limits.pop("message_bytes", MAX_MESSAGE_BYTES)
    proxy = limits.pop("proxy", None)
    if proxy:
        _use_outbound_proxy(proxy)
    _prefer_oom_kill()
    limit_resources(**limits)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    files = OPEN_FILES if hard == resource.RLIM_INFINITY else min(OPEN_FILES, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (files, files))
    router = _Router(channel)
    try:
        tools = load_catalog(directory, client=router)
    except BaseException as exc:  # a CatalogLoadError names the module; anything else is reported as it is
        if isinstance(exc, CatalogLoadError):
            kind, message = "CatalogLoadError", _text(exc)
        else:
            kind, message = "RuntimeError", f"the ToolSmith catalog cannot load: {describe(exc)}"
        channel.send({"type": "ready", "error": {"type": kind, "message": message}})
        os._exit(1)
    try:
        channel.send({"type": "ready", "tools": [{
            "tool_name": tool.tool_name, "tool_description": tool.tool_description, "input_schema": tool.input_schema}
            for tool in tools.values()]})
    except MessageTooLarge as exc:
        channel.send({"type": "ready", "error": {"type": "ValueError", "message": f"the tool list: {exc}"}})
        os._exit(1)
    while (message := channel.receive()) is not None:
        if message["type"] == "execute":
            threading.Thread(target=_run, args=(tools, message, channel), daemon=True).start()
        elif message["type"] == "reply":
            router.answer(message)
    # The router has gone without stopping the worker (it was killed): remove the private copy it made.
    home = Path(directory).parent
    if home.name.startswith(HOME_PREFIX):
        shutil.rmtree(home, ignore_errors=True)
    os._exit(0)


if __name__ == "__main__":
    main(*sys.argv[1:])
