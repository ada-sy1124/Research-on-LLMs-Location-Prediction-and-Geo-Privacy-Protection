from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geoai_pipeline.pipelines.rerun_yes_id_new_prompt2 import run


if __name__ == "__main__":
    run()


# python ./code/rerun_yes_id_new_prompt_2.py