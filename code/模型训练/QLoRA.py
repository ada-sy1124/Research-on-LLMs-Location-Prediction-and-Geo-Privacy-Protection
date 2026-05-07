import os
from swift.llm import sft_main, SftArguments

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = ""

print("🚀 准备启动 Qwen3-VL 的 QLoRA 微调...")

# 把刚才命令行的所有参数，优雅地写进 Python 的配置类里
args = SftArguments(
    # 1. 模型与数据路径
    model_id_or_path="/root/autodl-tmp/Qwen3-VL-8B-Instruct",
    dataset=["/root/autodl-tmp/my_train_data.jsonl"],
    output_dir="/root/autodl-tmp/qwen3_vl_qlora_output",
    
    # 2. 核心量化与微调策略 (QLoRA)
    sft_type="lora",
    quantization_bit=4,              # 开启 4-bit 量化，保住你的 4090 显存
    
    # 3. LoRA 矩阵的高阶参数（决定外挂的聪明程度）
    lora_rank=16,                    # 矩阵大小
    lora_alpha=32,                   # 缩放系数
    lora_dropout_p=0.05,
    lora_target_modules=["ALL"],     # 极其关键：同时微调视觉模块和语言模块
    
    # 4. 训练步数与硬件控制
    learning_rate=2e-4,              # 学习率
    batch_size=1,                    # 每次喂 1 张图
    gradient_accumulation_steps=4,   # 攒够 4 次再更新一次参数
    max_length=2048,                 # 文本最大长度
    logging_steps=5,                 # 每 5 步在控制台打印一次 Loss
)

# 直接调用 main 函数，拉起训练！
sft_main(args)