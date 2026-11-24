"""The evaluator reports a ToolSmith start-up failure in one line; any other error keeps its traceback."""
import pytest

from evaluation_kit import cli
from openvlharness.toolsmith.errors import CatalogLoadError, EncoderError, WorkerStartError

ARGS = ["--data", "FSC147_val", "--config", "configs/agent/qwen3_8b_paper.yaml",
        "--judge-config", "configs/judge_no_judge.yaml"]


@pytest.fixture
def evaluator(monkeypatch, tmp_path):
    for name in ("TOOL_SERVER_URL", "SAM3_SERVER_URL", "OCR_SERVER_URL", "DEPTH_ESTIMATION_SERVER_URL",
                 "MODEL_BASE_URL", "AUXILIARY_MODEL_BASE_URL"):
        monkeypatch.setenv(name, "http://fixture.invalid:8080")
    monkeypatch.setenv("MODEL_API_KEY", "fixture-key")
    monkeypatch.setenv("AUXILIARY_MODEL_API_KEY", "fixture-key")
    monkeypatch.setenv("LMUData", str(tmp_path))

    def run(failure):
        def build_agent(config):
            raise failure
        monkeypatch.setattr(cli.bootstrap, "build_agent", build_agent)
        cli.main([*ARGS, "--work-dir", str(tmp_path / "runs")])
    return run


@pytest.mark.parametrize("failure", [
    EncoderError("The sentence encoder 'BAAI/bge-m3' needs sentence-transformers, which the toolsmith extra "
                 "installs: pip install -e '.[toolsmith]' (ModuleNotFoundError: No module named 'sentence_transformers')"),
    CatalogLoadError("the ToolSmith catalog my_catalog cannot load broken.py: SyntaxError: expected ':'"),
    CatalogLoadError("the ToolSmith catalog my_catalog has tools that shadow base tools: OCR_Tool; rename them"),
    WorkerStartError("Generated tools run confined by Linux user namespaces, which this host does not allow (...)")])
def test_a_toolsmith_start_up_failure_is_one_line(evaluator, failure):
    with pytest.raises(SystemExit) as caught:
        evaluator(failure)
    assert caught.value.code == f"error: {failure}"


@pytest.mark.parametrize("failure", [ValueError("a configuration the agent cannot use"), RuntimeError("a bug")])
def test_any_other_error_keeps_its_traceback(evaluator, failure):
    with pytest.raises(type(failure)) as caught:
        evaluator(failure)
    assert caught.value is failure


def test_the_run_records_the_catalog_snapshot_its_agent_read_and_the_router_serves(evaluator, monkeypatch, tmp_path):
    # A tool admitted between the agent's metadata read and the manifest must not change what the run records.
    import requests
    import yaml

    from openvlharness.toolsmith.catalog import catalog_fingerprint

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    module = ("from openvlharness.toolsmith import GeneratedTool\n\n\nclass Tool(GeneratedTool):\n"
              "    def __init__(self):\n        super().__init__(tool_name={name!r}, tool_description='d', "
              "input_schema={{'type': 'object'}})\n")
    (catalog / "lookup.py").write_text(module.format(name="lookup"))
    snapshot = catalog_fingerprint(catalog)
    config = yaml.safe_load(open("configs/agent/qwen3_8b_toolsmith.yaml"))
    config["toolsmith"]["catalog"] = str(catalog)
    path = tmp_path / "toolsmith.yaml"
    path.write_text(yaml.safe_dump(config))

    def build_agent(config):
        agent = type("Agent", (), {"model": "Qwen/Qwen3-VL-8B-Instruct", "catalog_sha256": snapshot})()
        (catalog / "added.py").write_text(module.format(name="added"))  # the pipeline admits a tool meanwhile
        return agent
    served = type("Reply", (), {"raise_for_status": lambda self: None,
                                "json": lambda self: {"toolsmith": {"catalog_sha256": snapshot}}})()
    recorded = []

    def prepare(pred_root, manifest, logger):
        recorded.append(manifest)
        raise SystemExit("stop before inference")
    monkeypatch.setattr(cli.bootstrap, "build_agent", build_agent)
    monkeypatch.setattr(requests, "get", lambda url, timeout: served)
    monkeypatch.setattr(cli, "_prepare_run_root", prepare)
    with pytest.raises(SystemExit, match="stop before inference"):
        cli.main(["--data", "FSC147_val", "--config", str(path), "--judge-config", "configs/judge_no_judge.yaml",
                  "--work-dir", str(tmp_path / "runs")])
    assert catalog_fingerprint(catalog) != snapshot
    assert recorded[0]["toolsmith_catalog_sha256"] == snapshot


def _toolsmith_run(tmp_path, monkeypatch, served_digest, catalog_path, **toolsmith):
    """Run the evaluator up to its run directory against a router whose /env reports served_digest."""
    import requests
    import yaml

    config = yaml.safe_load(open("configs/agent/qwen3_8b_toolsmith.yaml"))
    config["toolsmith"].update(catalog=str(catalog_path), **toolsmith)
    path = tmp_path / "toolsmith.yaml"
    path.write_text(yaml.safe_dump(config))
    built, recorded = [], []

    def build_agent(config):
        built.append(config)
        return type("Agent", (), {"model": "Qwen/Qwen3-VL-8B-Instruct", "catalog_sha256": served_digest})()
    served = type("Reply", (), {"raise_for_status": lambda self: None,
                                "json": lambda self: {"toolsmith": {"catalog_sha256": served_digest}}})()

    def prepare(pred_root, manifest, logger):
        recorded.append(manifest)
        raise SystemExit("stop before inference")
    monkeypatch.setattr(cli.bootstrap, "build_agent", build_agent)
    monkeypatch.setattr(requests, "get", lambda url, timeout: served)
    monkeypatch.setattr(cli, "_prepare_run_root", prepare)
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--data", "FSC147_val", "--config", str(path), "--judge-config", "configs/judge_no_judge.yaml",
                  "--work-dir", str(tmp_path / "runs")])
    return str(stopped.value.code), built, recorded


def test_a_router_serving_another_catalog_stops_the_run_before_the_agent_loads(evaluator, monkeypatch, tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "lookup.py").write_text("# the configured catalog B\n")
    message, built, recorded = _toolsmith_run(tmp_path, monkeypatch, "a" * 64, catalog)
    assert "serves ToolSmith catalog aaaa" in message and "Restart the router" in message
    assert (built, recorded) == ([], [])  # stopped before the agent and its encoder load


def test_a_catalog_not_readable_here_stops_the_run_unless_the_router_is_trusted(evaluator, monkeypatch, tmp_path,
                                                                                 caplog):
    import logging
    import os

    run_logger = logging.getLogger("RUN")  # the evaluator's logger does not propagate to caplog's root handler
    run_logger.addHandler(caplog.handler)
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    (unreadable / "lookup.py").write_text("# a module\n")
    unreadable.chmod(0)
    try:
        cases = [tmp_path / "not-on-this-host", tmp_path / "catalgo"]  # missing, or a typo
        locked = tmp_path / "locked_module"
        locked.mkdir()
        (locked / "lookup.py").write_text("# a module\n")
        (locked / "lookup.py").chmod(0)
        if os.geteuid() != 0 and not os.access(unreadable, os.R_OK):
            cases += [unreadable, locked]  # listing the directory, or reading a module, raises PermissionError
        for catalog in cases:
            message, built, recorded = _toolsmith_run(tmp_path, monkeypatch, "b" * 64, catalog)
            assert message.startswith(f"error: the ToolSmith catalog {str(catalog)!r} is not readable here")
            assert "toolsmith.trust_router_catalog: true" in message and (built, recorded) == ([], [])
            caplog.clear()
            caplog.set_level(logging.WARNING)
            message, built, recorded = _toolsmith_run(tmp_path, monkeypatch, "b" * 64, catalog,
                                                      trust_router_catalog=True)
            assert message == "stop before inference" and len(built) == 1
            assert recorded[0]["toolsmith_catalog_sha256"] == "b" * 64
            assert "as toolsmith.trust_router_catalog asks, the run uses the catalog the tool router" in caplog.text
    finally:
        run_logger.removeHandler(caplog.handler)
        unreadable.chmod(0o755)
        (tmp_path / "locked_module" / "lookup.py").chmod(0o644)
