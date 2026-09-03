import json

from src.toolkit.process_results import extract_results


def test_extract_results_ignores_skill_memory_audit_json(tmp_path):
    run_dir = tmp_path / "skill_memory" / "0"
    run_dir.mkdir(parents=True)

    (run_dir / "logs.json").write_text(
        json.dumps({"Top1_Acc_Stream/eval_phase/test_stream/Task000": 0.5}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "skill_memory_audit.json").write_text(
        json.dumps({"decision": "clone", "compatibility_score": 0.5}) + "\n",
        encoding="utf-8",
    )

    results = extract_results(str(tmp_path), verbose=False)

    assert set(results) == {"training"}
    assert len(results["training"]) == 1
    assert results["training"].iloc[0]["seed"] == 0


def test_extract_results_skips_unrecognized_json_without_failing(tmp_path):
    run_dir = tmp_path / "method" / "1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"note": "auxiliary metadata"}) + "\n",
        encoding="utf-8",
    )

    results = extract_results(str(tmp_path), verbose=False)

    assert results == {}
