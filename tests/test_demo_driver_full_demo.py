from pathlib import Path

import pytest


# commit: system/acceptance test
def test_run_full_demo_generates_visual_artifacts(tmp_path: Path):
    # TS-10 TC-097
    pytest.importorskip("matplotlib")

    from demo_driver import run_full_demo

    outputs = run_full_demo(row_limit=16, output_dir=tmp_path)

    assert outputs is not None, "run_full_demo should return outputs"
    assert isinstance(outputs, dict), "outputs should be a dictionary"
    assert len(outputs) > 0, "outputs should not be empty"

    for key, image_path in outputs.items():
        assert isinstance(image_path, Path), f"output value {key} should be a Path object"
        assert image_path.exists(), f"output file {image_path} should exist"
        assert (
            image_path.suffix == ".png"
        ), f"output file {key} should be a PNG file, got {image_path.suffix}"
        assert image_path.stat().st_size > 0, f"output file {image_path} should have non-zero size"
