from pathlib import Path

import pytest


# commit: system/acceptance test
def test_run_full_demo_generates_visual_artifacts(tmp_path: Path):
    # TS-10 TC-097
    pytest.importorskip("matplotlib")

    from demo_driver import run_full_demo

    outputs = run_full_demo(row_limit=16, output_dir=tmp_path)

    assert outputs
    for image_path in outputs.values():
        assert image_path.exists()
        assert image_path.stat().st_size > 0
