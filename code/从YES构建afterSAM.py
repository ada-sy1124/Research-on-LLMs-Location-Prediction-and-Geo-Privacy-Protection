from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geoai_pipeline.pipelines.from_yes_build_after_sam import main


if __name__ == "__main__":
    main()




# image_original: 原始图像
# latitude: 真值纬度
# longitude: 真值经度
# d_original: 原图下模型预测误差（km）
# ablated_class: 要做遮挡的类别列表（字符串数组）
# masked_image: 与 ablated_class 等长的遮挡图列表
# q_ratio: 与 ablated_class 等长的遮挡面积比例列表（0~1）


# {
#   "image_original": "<PIL.Image RGB 1024x1024>",
#   "latitude": 51.5074,
#   "longitude": -0.1278,
#   "d_original": 2.31,
#   "ablated_class": ["Vehicles", "Signage & Text", "Road Markings"],
#   "masked_image": [
#     "<PIL.Image: masked Vehicles>",
#     "<PIL.Image: masked Signage & Text>",
#     "<PIL.Image: masked Road Markings>"
#   ],
#   "q_ratio": [0.082, 0.019, 0.034]
# }

