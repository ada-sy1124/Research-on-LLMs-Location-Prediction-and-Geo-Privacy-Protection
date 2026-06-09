import os
import subprocess
import sys
from pathlib import Path


# ================= 0. 一键运行顺序 =================
RUN_STEPS = [
    "0.py",
    "SAM3使用.py",
    "筛选候选.py",
    "1_prepare_coordinate_tokens.py",
    "2_build_valid_token_table.py",
    "3_build_distance_table.py",
    "4_min_convergence.py",
]


def main():
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")

    for script in RUN_STEPS:
        path = root / script
        print("=" * 80)
        print(f"开始运行: {path}")
        print("=" * 80)
        subprocess.run([sys.executable, str(path)], check=True, env=env)

    print("=" * 80)
    print("DCSD 分层提取流程全部完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
