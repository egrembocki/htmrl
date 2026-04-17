from pathlib import Path

import pytest


# Test Type: acceptance test : TS 30 TC 262
def test_run_full_demo_generates_visual_artifacts(tmp_path: Path):
    # This test validates that the full demo runs end-to-end and produces expected visual outputs.
    pytest.importorskip("matplotlib")

    from matplotlib import image as mpimg

    from demo_driver import run_full_demo

    outputs = run_full_demo(row_limit=16, output_dir=tmp_path)

    # Contract: run_full_demo returns exactly these two named plots.
    assert set(outputs.keys()) == {"signal", "fft"}

    expected_paths = {
        "signal": tmp_path / "demo_signal.png",
        "fft": tmp_path / "demo_signal_fft.png",
    }
    assert outputs == expected_paths

    for name, image_path in outputs.items():
        assert image_path.exists()
        assert image_path.parent == tmp_path
        assert image_path.suffix == ".png"
        # Ensure files are non-trivial images, not empty placeholders.
        assert image_path.stat().st_size > 1024

        image_data = mpimg.imread(image_path)
        assert image_data.size > 0
        assert image_data.ndim in (2, 3), f"Unexpected image rank for {name}: {image_data.ndim}"
