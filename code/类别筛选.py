import json
import os
from pathlib import Path
import numpy as np

# =====================================================================
# 🛠️ 全局配置区 (直接在此处修改参数，点击运行即可)
# =====================================================================
INPUT_PATH = "./data/YES_Mask"                       # 原始 HuggingFace 数据集路径
OUTPUT_DIR = "./data/YES_Mask_selected"              # 打上最优标签后的数据集保存路径
JSON_PATH = "./data/qwen_sft_alignment_dataset.json" # 导出的轻量级 SFT JSON 路径

TAU = 0.6              # 业务合规底线：最低信息留存率要求 (默认 70%)
PRIVACY_CAP_KM = 100.0 # 隐私误差封顶上限：定位误差增加 100km 映射为保护率 1.0
# =====================================================================


def select_optimal_category(utility_scores, privacy_scores, tau=0.7):
    """
    核心决策逻辑：优先满足留存率 >= tau (工况 A)，在此前提下最大化隐私保护。
    若全员未达标 (工况 B)，启动安全兜底，强制选择留存率最大的类别。
    """
    u_arr = np.array(utility_scores, dtype=float)
    p_arr = np.array(privacy_scores, dtype=float)

    if len(u_arr) == 0 or len(u_arr) != len(p_arr):
        return -1, "Invalid: empty or mismatched score list"

    valid_mask = u_arr >= tau

    if np.any(valid_mask):
        # 工况 A：存在达标解，非达标解的隐私分设为负无穷，提取最优解
        masked_privacy = np.where(valid_mask, p_arr, -np.inf)
        best_c = int(np.argmax(masked_privacy))
        rule_type = "Condition A: utility qualified, maximize privacy"
    else:
        # 工况 B：无解兜底，强制提取留存率最大的类别
        best_c = int(np.argmax(u_arr))
        rule_type = "Condition B: no qualified utility, maximize utility"

    return best_c, rule_type


def build_scores(q_ratio, d_diff, privacy_cap_km):
    """
    指标转换公式：
      - 信息留存率 (utility) = 1 - 掩码面积比例 (q_ratio)
      - 隐私保护率 (privacy) = 截断后的地理定位误差增量 / 封顶上限
    """
    q_arr = np.array(q_ratio, dtype=float)
    d_arr = np.array(d_diff, dtype=float)

    utility_scores = np.clip(1.0 - q_arr, 0.0, 1.0)
    
    # 防御 NaN/Inf 等长尾恶劣数值
    privacy_gain = np.nan_to_num(d_arr, nan=0.0, posinf=privacy_cap_km, neginf=0.0)
    privacy_gain = np.clip(privacy_gain, 0.0, privacy_cap_km)
    privacy_scores = privacy_gain / privacy_cap_km

    return utility_scores.tolist(), privacy_scores.tolist()


def select_for_sample(sample, sample_index, tau, privacy_cap_km):
    """为单个全景样本提取最优伪标签"""
    ablated_class = sample.get("ablated_class") or []
    q_ratio = sample.get("q_ratio") or []
    d_diff = sample.get("d_diff") or []

    # 校验字段完整性
    if not ablated_class or len(ablated_class) != len(q_ratio) or len(ablated_class) != len(d_diff):
        return {
            "sample_id": sample.get("sample_id", f"yes_mask_{sample_index:04d}"),
            "target_mask_category": "Nothing",
            "label": "Nothing",
            "target_mask_category_index": -1,
            "final_utility": 1.0,
            "final_privacy": 0.0,
            "selection_rule": "Invalid: missing or mismatched ablated_class/q_ratio/d_diff",
        }

    utility_scores, privacy_scores = build_scores(q_ratio, d_diff, privacy_cap_km)
    best_idx, rule = select_optimal_category(utility_scores, privacy_scores, tau=tau)

    if best_idx < 0:
        target_category = "Nothing"
        final_utility = 1.0
        final_privacy = 0.0
    else:
        target_category = ablated_class[best_idx]
        final_utility = utility_scores[best_idx]
        final_privacy = privacy_scores[best_idx]

    return {
        "sample_id": sample.get("sample_id", f"yes_mask_{sample_index:04d}"),
        "target_mask_category": target_category,
        "label": target_category,  # 冗余一份作为常规 SFT 的 target 标签名
        "target_mask_category_index": best_idx,
        "final_utility": float(final_utility),
        "final_privacy": float(final_privacy),
        "selection_rule": rule,
    }


def main():
    print("🚀 启动 YES_Mask 离线多目标标签筛选与清洗流水线...")
    
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("❌ 缺少依赖: 请在运行脚本前安装 datasets 库 (pip install datasets)。") from exc
    
    input_path = Path(INPUT_PATH).expanduser()
    output_path = Path(OUTPUT_DIR).expanduser()
    json_path = Path(JSON_PATH).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"❌ 找不到输入数据集，请检查路径: {input_path}")

    print(f"📦 正在加载 HuggingFace 数据集: {input_path}")
    dataset = load_from_disk(str(input_path))
    print(f"✅ 成功加载 {len(dataset)} 个样本。正在执行逐样本对偶解析解寻优...")

    # 批量计算与提取
    selections = [
        select_for_sample(sample, idx, tau=TAU, privacy_cap_km=PRIVACY_CAP_KM)
        for idx, sample in enumerate(dataset)
    ]

    # 回填结果到数据集中
    annotated = dataset
    columns_to_update = [
        "sample_id", "target_mask_category", "label", 
        "target_mask_category_index", "final_utility", 
        "final_privacy", "selection_rule"
    ]
    
    for column in columns_to_update:
        if column in annotated.column_names:
            annotated = annotated.remove_columns(column)
        annotated = annotated.add_column(column, [row[column] for row in selections])

    # 安全保存检测 (防止意外覆盖旧数据)
    if output_path.exists():
        raise FileExistsError(f"⚠️ 输出目录已存在，为了防止覆盖珍贵数据，请手动删除旧目录或更换路径: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save_to_disk(str(output_path))

    # 导出轻量级微调 JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "sample_id": row["sample_id"],
                    "target_mask_category": row["target_mask_category"],
                    "target_mask_category_index": row["target_mask_category_index"],
                }
                for row in selections
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 统计审查报告
    rules = {}
    for row in selections:
        rules[row["selection_rule"]] = rules.get(row["selection_rule"], 0) + 1

    mean_utility = float(np.mean([row["final_utility"] for row in selections])) if selections else 0.0
    mean_privacy = float(np.mean([row["final_privacy"] for row in selections])) if selections else 0.0

    print("\n📊 标签对齐审计报告 (Selection Audit):")
    for rule, count in sorted(rules.items()):
        print(f"  - {rule}: {count} 个样本")
        
    print("\n🎯 最终大盘全局指标预期：")
    print(f"   - 平均信息留存率 (Average utility retention): {mean_utility * 100:.2f}%")
    print(f"   - 平均隐私保护率 (Average privacy score): {mean_privacy * 100:.2f}%")
    
    print(f"\n💾 完整 HF 数据集已保存至: {output_path}")
    print(f"💾 Qwen SFT 专用 JSON 已保存至: {json_path}")
    
    # =====================================================================
    # 📊 打印 6 个类别各自命中的样本数量和百分比 (防范模式崩溃审查)
    # =====================================================================
    print("\n🏷️ 类别命中大盘分布 (Category Distribution Audit):")
    cat_counts = {}
    total_samples = len(selections)
    
    # 逐一统计每个类别的频次
    for row in selections:
        cat = row["target_mask_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
    # 按命中数量从高到低排序打印
    for cat, count in sorted(cat_counts.items(), key=lambda item: item[1], reverse=True):
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"   - {cat:<20}: {count:>5} 张图 ({percentage:.2f}%)")
    # =====================================================================


if __name__ == "__main__":
    main()

# python ./code/类别筛选.py