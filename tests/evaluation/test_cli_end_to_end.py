"""openvlharness-eval end to end against an in-process fake model server.

A three-row FSC147-style split is written to $LMUData; the orchestrator is a
loopback HTTP server that answers with <answer>N</answer>. No GPU, tool router,
tokenizer or network access is needed. The last tests check that bad inputs stop
the CLI before it creates a run directory.
"""
import base64
import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from evaluation_kit import cli
from evaluation_kit.dataset import hf_release

NO_JUDGE = Path(__file__).resolve().parents[2] / "configs" / "judge_no_judge.yaml"

ANSWERS = {"zero": 3, "one": 5, "two": 7}
REPLIES = {"zero": "3", "one": "6", "two": "7"}   # one miss by 1 -> MAE 0.33, RMSE 0.58, acc 66.67
METRICS = {"mae": 0.33, "rmse": 0.58, "acc": 66.67}


@pytest.fixture
def fake_model(monkeypatch):
    state = {"requests": 0, "fail": set()}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            state["requests"] += 1
            word = next(w for w in ANSWERS if f"question {w}" in body)
            if word in state["fail"]:
                self.send_response(400); self.send_header("Content-Length", "2"); self.end_headers()
                self.wfile.write(b"{}")
                return
            reply = json.dumps({"id": "r", "object": "chat.completion", "choices": [{"index": 0,
                "finish_reason": "stop", "message": {"role": "assistant",
                "content": f"<answer>{REPLIES[word]}</answer>"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply))); self.end_headers(); self.wfile.write(reply)

        def do_GET(self):  # router /env snapshot
            self.send_response(404); self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("MODEL_BASE_URL", base + "/v1")
    monkeypatch.setenv("TOOL_SERVER_URL", base)
    monkeypatch.setenv("WORLD_SIZE", "1")
    yield state
    server.shutdown()


@pytest.fixture
def tiny_split(tmp_path, monkeypatch):
    rows = []
    for i, word in enumerate(ANSWERS):
        png = io.BytesIO(); Image.new("RGB", (8, 8), "red").save(png, format="PNG")
        rows.append({"index": i, "image": base64.b64encode(png.getvalue()).decode(), "image_path": f"{i}.png",
                     "question": f"How many dots? question {word}", "answer": ANSWERS[word]})
    lmu = tmp_path / "LMUData"; lmu.mkdir()
    pd.DataFrame(rows).to_csv(lmu / "FSC147_val.tsv", sep="\t", index=False)
    monkeypatch.setenv("LMUData", str(lmu))
    monkeypatch.delitem(hf_release.HF_DATASET_MD5, "FSC147_val")   # accept the synthetic TSV
    config = {"agent_name": "MainAgent", "agent_config_version": "e2e_v1",
              "agent": {"model_service": "orchestrator", "max_iterations": 3},
              "memory": {"modules": [{"name": "image"}]},
              "capabilities": {"registries": ["paper"], "tools": {"zoom": {"backend": "zoom_in_image_by_bbox"}}},
              "services": {"client": {"base_url": "${TOOL_SERVER_URL}", "retry": 0}, "server": {},
                           "tools": {"zoom_in_image_by_bbox": {}},
                           "models": {"orchestrator": {"alias": "qwen3-vl-8b-instruct", "base": "${MODEL_BASE_URL}",
                                                       "key": "unused", "max_tokens": 64, "retry": 0}}}}
    path = tmp_path / "agent.yaml"; path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def run_log(caplog):
    """Collect the CLI's log; the RUN logger does not propagate to the root logger."""
    logger = logging.getLogger("RUN")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


def run(tmp_path, config, *extra):
    cli.main(["--data", "FSC147_val", "--config", str(config), "--judge-config", str(NO_JUDGE),
              "--work-dir", str(tmp_path / "runs"), "--num-proc", "1", *extra])
    (run_dir,) = (tmp_path / "runs" / "Qwen3-VL-8B-Instruct" / "mainagent").iterdir()
    return run_dir


def metrics(run_dir):
    return pd.read_csv(run_dir / "FSC147_val" / "Qwen3-VL-8B-Instruct_acc.csv").iloc[0].to_dict()


def test_infer_then_score_then_eval_only(tmp_path, fake_model, tiny_split, run_log):
    run_dir = run(tmp_path, tiny_split)
    assert fake_model["requests"] == 3
    assert metrics(run_dir) == METRICS
    for name in ("run_manifest.json", "agent_config.redacted.yaml", "judge_config.redacted.yaml"):
        assert (run_dir / name).is_file()
    assert sorted(p.name for p in (run_dir / "FSC147_val" / "full_trajectories").glob("*.json")) == \
        ["0.json", "1.json", "2.json"]
    for key, value in METRICS.items():
        assert f'"{key}": {value}' in run_log.text
    assert f"Results for run {run_dir}" in run_log.text
    assert "samples failed" not in run_log.text
    # Evaluation-only re-scores saved predictions without contacting the model.
    run_log.clear()
    cli.main(["--data", "FSC147_val", "--mode", "eval", "--run-dir", str(run_dir)])
    assert fake_model["requests"] == 3
    assert metrics(run_dir) == METRICS
    assert '"acc": 66.67' in run_log.text


def test_failed_rows_are_retried_alone_and_db_is_kept(tmp_path, fake_model, tiny_split, run_log):
    fake_model["fail"] = {"one"}
    run_dir = run(tmp_path, tiny_split)
    work = run_dir / "FSC147_val"
    predictions = pd.read_excel(work / "Qwen3-VL-8B-Instruct.xlsx")["prediction"].tolist()
    assert predictions[1] == "Error: Unable to get response from the server."
    # The cause of the failure is kept next to the trajectory and reported in the log.
    error = json.loads((work / "full_trajectories" / "1.json").read_text())["error"]
    assert "400" in error
    assert "error" not in json.loads((work / "full_trajectories" / "0.json").read_text())
    assert "Sample 1 failed" in run_log.text
    assert f"1/3 samples failed (first error: {error}" in run_log.text
    assert f"--run-dir {run_dir} --ignore-failed" in run_log.text
    # --keep-file-db help: "Cleanup is also skipped when failures remain".
    assert (work / "results").is_dir(), "file-db deleted although a row failed"
    fake_model["fail"].clear()
    run(tmp_path, tiny_split, "--run-dir", str(run_dir), "--ignore-failed")
    assert fake_model["requests"] == 4          # only the failed row is re-sent
    assert metrics(run_dir) == METRICS


def test_interrupted_inference_resumes_without_repeating_committed_rows(tmp_path, fake_model, tiny_split):
    run_dir = run(tmp_path, tiny_split, "--mode", "infer")
    work = run_dir / "FSC147_val"
    # Simulate a crash after rows 0 and 2 committed: no aggregate file, row 1 uncommitted.
    (work / "Qwen3-VL-8B-Instruct.xlsx").unlink()
    (work / "_done" / "1.flag").unlink()
    for record in (work / "results").glob("*/1"):
        for path in record.iterdir():
            path.unlink()
        record.rmdir()
    assert run(tmp_path, tiny_split, "--resume") == run_dir
    assert fake_model["requests"] == 4                     # only row 1 is re-sent
    assert metrics(run_dir) == METRICS


def test_world_size_above_one_exits_before_creating_a_run(tmp_path, fake_model, tiny_split, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit) as exit_info:
        run(tmp_path, tiny_split)
    assert "runs in a single process" in str(exit_info.value.code)
    assert not (tmp_path / "runs").exists()
    assert fake_model["requests"] == 0


def test_unknown_dataset_exits_before_creating_a_run(tmp_path, fake_model, tiny_split):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--data", "FSC147_vall", "--config", str(tiny_split), "--judge-config", str(NO_JUDGE),
                  "--work-dir", str(tmp_path / "runs")])
    message = str(exit_info.value.code)
    assert message.startswith("Unknown dataset FSC147_vall. Did you mean: FSC147_val")
    assert not (tmp_path / "runs").exists()


def test_dataset_check_accepts_registered_and_custom_tsv_names(tmp_path, monkeypatch):
    monkeypatch.setenv("LMUData", str(tmp_path))
    (tmp_path / "MyCounting.tsv").write_text("index\tquestion\n0\tHow many?\n")
    cli._check_datasets(["FSC147_val", "CountQA", "MyCounting"])
    monkeypatch.delenv("LMUData")
    with pytest.raises(SystemExit, match="LMUData is not set"):
        cli._check_datasets(["FSC147_val"])


def test_default_configs_missing_from_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(SystemExit, match="Run from the repository root or pass --config and --judge-config"):
        cli.main(["--data", "FSC147_val"])
    assert not (tmp_path / "outputs").exists()
