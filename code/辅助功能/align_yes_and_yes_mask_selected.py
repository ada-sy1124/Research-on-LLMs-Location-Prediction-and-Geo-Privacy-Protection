from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geoai_pipeline.pipelines.helpers.align_yes_and_yes_mask_selected import run


if __name__ == "__main__":
    run()


# python .\code\辅助功能\align_yes_and_yes_mask_selected.py
