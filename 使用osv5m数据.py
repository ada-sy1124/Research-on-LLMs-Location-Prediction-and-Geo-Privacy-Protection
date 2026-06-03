import os
from datasets import load_dataset

# 1. 路径配置 (绝不碰 C 盘)
SAVE_DIR = r"D:\GeoAI\Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection-\data\osv5m_20_samples"
CACHE_DIR = r"D:\GeoAI\Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection-\model_cache\huggingface"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print("🚀 正常下载模式启动，已将缓存区绑定至 D 盘...")

# 2. 完全按照官方指令 + 强制 D 盘缓存
dataset = load_dataset(
    'osv5m/osv5m', 
    split='train', 
    full=False, 
    trust_remote_code=True,
    cache_dir=CACHE_DIR  # <--- 就是这句，彻底守住 C 盘！
)

print("✅ 下载/加载完成！正在保存前 20 张图像...")

# 3. 老老实实顺次取前 20 张，保存走人
for i in range(20):
    try:
        sample = dataset[i] 
        img = sample["image"]
        
        # 提取经纬度
        lat = round(sample.get("latitude", 0.0), 4)
        lon = round(sample.get("longitude", 0.0), 4)
        
        filename = f"sample_{i:02d}_lat{lat}_lon{lon}.jpg"
        img.save(os.path.join(SAVE_DIR, filename))
        print(f"  ✅ 第 {i+1} 张保存成功: {filename}")
        
    except Exception as e:
        print(f"  ❌ 第 {i+1} 个样本保存失败: {e}")

print(f"\n🎉 搞定！20 张图和所有缓存都老老实实呆在 D 盘了！")


# python ./使用osv5m数据.py