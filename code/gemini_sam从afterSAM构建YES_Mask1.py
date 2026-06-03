from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geoai_pipeline.pipelines.from_after_sam_build_yes_mask1 import run


if __name__ == "__main__":
    run()



# image: 原始图像
# latitude_true: 真值纬度
# longitude_true: 真值经度
# d_original: 原图误差（km）
# ablated_class: 遮挡类别列表
# q_ratio: 每个类别对应的遮挡比例
# d_prime: 每个遮挡版本下的新误差（km）
# d_diff: 每个类别的误差变化，d_diff[i] = d_prime[i] - d_original



# {
#   "image": "<PIL.Image RGB 1024x1024>",
#   "latitude_true": 51.5074,
#   "longitude_true": -0.1278,
#   "d_original": 2.31,
#   "ablated_class": ["Vehicles", "Signage & Text", "Road Markings"],
#   "q_ratio": [0.082, 0.019, 0.034],
#   "d_prime": [14.72, 6.05, 3.12],
#   "d_diff": [12.41, 3.74, 0.81]
# }
