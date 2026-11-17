import pandas as pd

from qsi.cli import main


def test_cli_creates_reports_and_shots(tmp_path):
    main(["--conditions", "logi2_all,vqc2,vqc7", "--seeds", "0", "--steps", "3", "--out", str(tmp_path)])
    runs = pd.read_csv(tmp_path / "runs.csv")
    shots = pd.read_csv(tmp_path / "shots.csv")
    assert len(runs) == 3 and runs.miou.notna().all()
    assert len(shots) == 5 and set(shots.model) == {"vqc7"}
    for filename in ("summary.csv", "summary.md", "loss_curves.png", "miou_by_condition.png"):
        assert (tmp_path / filename).stat().st_size > 0


def test_cli_logistic_only_omits_vqc_artifacts(tmp_path):
    main(["--conditions", "logi2_all", "--out", str(tmp_path)])
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "summary.md").exists()
    assert not (tmp_path / "shots.csv").exists()
    assert not (tmp_path / "loss_curves.png").exists()
