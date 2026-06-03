
from transformers import Sam3Processor, Sam3Model
import torch
import os
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks # 导入画掩码工具

# ================= 0. 定义要遮挡的多个目标 =================
targets = [
    "signboards", "trash cans", "traffic signs", "hedges"
]
# signboards, trash cans, traffic signs, roads, trees, hedges

# ================= 1. 设备与环境检查 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 正在使用的计算设备: {device}")

# ================= 2. 加载 Transformers 格式的模型 (SAM3) =================
print("⏳ 正在通过 Transformers 库加载 SAM3 模型到显存...")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# ================= 3. 读取本地图像 =================
image_path = r"D:\GeoAI\Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection-\切分后images\4751579_8241809.jpeg"
print(f"📂 正在读取图像: {image_path}")
image = Image.open(image_path).convert("RGB")
img_tensor = F.pil_to_tensor(image) # 保持 uint8 格式 [3, H, W]

# ================= 4. 初始化全局掩码 =================
_, H, W = img_tensor.shape
combined_mask = torch.zeros((H, W), dtype=torch.bool).to(device) # 直接放在GPU上加速
total_found_objects = 0
original_size = [image.size[::-1]] 

# ================= 5. 循环推理并累加掩码 =================
for text_prompt in targets:
    print(f"\n🎯 正在搜索目标: '{text_prompt}' ...")
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=original_size
    )[0]

    found_count = len(results['masks'])
    
    if found_count > 0:
        print(f"  ✅ 找到 {found_count} 个 '{text_prompt}'")
        total_found_objects += found_count
        
        # 提取并在 GPU 上合并当前 Mask
        current_masks = results['masks'] > 0.5 
        current_merged = current_masks.any(dim=0)
        
        # 累加到全局 Mask
        combined_mask = combined_mask | current_merged
    else:
        print(f"  ⚠️ 未找到 '{text_prompt}'")

# ================= 6. 一次性绘制纯黑块并保存 (已恢复物理遮挡) =================
print("\n=================================================")
if total_found_objects > 0:
    SAVE_DIR = r"D:\GeoAI\Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection-\切分后images"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 提取原图名
    original_basename = os.path.splitext(os.path.basename(image_path))[0]
    save_filename = f"{original_basename}_multi_masked.jpg"
    save_path = os.path.join(SAVE_DIR, save_filename)
    
    # 把 GPU 上的掩码转回 CPU 并扩展维度，以符合 draw_segmentation_masks 的输入规范
    cpu_mask = combined_mask.unsqueeze(0).cpu()
    
    # alpha=1.0 表示完全不透明（纯色覆盖），colors="black" 表示使用黑色物理遮挡
    result_tensor = draw_segmentation_masks(
        img_tensor, 
        masks=cpu_mask, 
        alpha=1.0, 
        colors="black"
    )
    
    # 转回 PIL 图像并保存
    result_img = F.to_pil_image(result_tensor)
    result_img.save(save_path)
    
    print(f"🎉 物理毁灭完成！共对 {total_found_objects} 个目标的区域进行了纯黑覆盖。")
    print(f"📁 图像已成功保存至: {save_path}")
else:
    print("⚠️ 所有给定的提示词都未匹配到物体，未生成新图像。")







# from transformers import Sam3Processor, Sam3Model
# import torch
# import os
# from PIL import Image
# import torchvision.transforms.functional as F
# from torchvision.utils import draw_segmentation_masks 

# # ================= 0. 定义要遮挡的多个目标 =================
# targets = [
#     "signboards", "trash cans", "traffic signs", "hedges"
# ]
# # trash bags, road markings, buildings, signboards, cars, manhole covers, trees, hedges, fences


# # ================= 1. 设备与环境检查 =================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🚀 正在使用的计算设备: {device}")

# # ================= 2. 加载 模型 =================
# print("⏳ 正在通过 Transformers 库加载 SAM3 模型到显存...")
# model = Sam3Model.from_pretrained("facebook/sam3").to(device)
# processor = Sam3Processor.from_pretrained("facebook/sam3")

# # ================= 3. 读取本地图像 =================
# image_path = r"D:\GeoAI\Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection-\切分后images\4751579_8241809.jpeg"
# print(f"📂 正在读取图像: {image_path}")
# image = Image.open(image_path).convert("RGB")
# img_tensor = F.pil_to_tensor(image) 

# # ================= 4. 初始化全局掩码 =================
# _, H, W = img_tensor.shape
# combined_mask = torch.zeros((H, W), dtype=torch.bool, device=device) 
# total_found_objects = 0
# original_size = [image.size[::-1]] 

# # ================= 5. 循环推理并累加矩形掩码 =================
# for text_prompt in targets:
#     print(f"\n🎯 正在搜索目标: '{text_prompt}' ...")
#     inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     results = processor.post_process_instance_segmentation(
#         outputs,
#         threshold=0.4,       # 置信度阈值
#         mask_threshold=0.5,
#         target_sizes=original_size
#     )[0]

#     # 【关键修改】：不再看 masks，而是看 boxes
#     if 'boxes' in results and len(results['boxes']) > 0:
#         found_count = len(results['boxes'])
#         print(f"  ✅ 找到 {found_count} 个 '{text_prompt}'")
#         total_found_objects += found_count
        
#         # 初始化一个当前目标的空白矩形掩码
#         current_rect_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
#         boxes = results['boxes']
        
#         # 遍历所有找到的边界框
#         for box in boxes:
#             # 提取 [xmin, ymin, xmax, ymax] 并转为整数
#             xmin, ymin, xmax, ymax = box.int().tolist()
            
#             # 安全防护：防止框的坐标越出图像边界
#             xmin, ymin = max(0, xmin), max(0, ymin)
#             xmax, ymax = min(W, xmax), min(H, ymax)
            
#             # 将这个矩形区域内所有的像素强行设为 True (变成一个实心方块)
#             current_rect_mask[ymin:ymax, xmin:xmax] = True
        
#         # 累加到全局 Mask
#         combined_mask = combined_mask | current_rect_mask
#     else:
#         print(f"  ⚠️ 未找到 '{text_prompt}'")

# # ================= 6. 一次性绘制纯黑块并保存 =================
# print("\n=================================================")
# if total_found_objects > 0:
#     SAVE_DIR = r"D:\GeoAI\Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection-\切分后images"
#     os.makedirs(SAVE_DIR, exist_ok=True)
    
#     original_basename = os.path.splitext(os.path.basename(image_path))[0]
#     # 改个名字，标注这是矩形打码
#     save_filename = f"{original_basename}_bbox_masked.jpg" 
#     save_path = os.path.join(SAVE_DIR, save_filename)
    
#     cpu_mask = combined_mask.unsqueeze(0).cpu()
    
#     result_tensor = draw_segmentation_masks(
#         img_tensor, 
#         masks=cpu_mask, 
#         alpha=1.0, 
#         colors="black"
#     )
    
#     result_img = F.to_pil_image(result_tensor)
#     result_img.save(save_path)
    
#     print(f"🎉 绝对物理毁灭完成！共对 {total_found_objects} 个目标施加了防轮廓泄露的【矩形黑块】。")
#     print(f"📁 图像已成功保存至: {save_path}")
# else:
#     print("⚠️ 所有给定的提示词都未匹配到物体，未生成新图像。")

    
# python ./SAM3使用.py
