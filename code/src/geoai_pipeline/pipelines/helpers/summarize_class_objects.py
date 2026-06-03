from collections import defaultdict

from datasets import load_from_disk

from geoai_pipeline.config import get_path
from geoai_pipeline.tools.reasoning import extract_class_objects_from_reason


def run():
    dataset_path = get_path(
        "YES_DATASET_PATH",
        "./data/YES",
    )
    output_txt_path = get_path(
        "SAM_PROMPTS_OUTPUT_PATH",
        "./data/sam_prompts.txt",
    )

    dataset = load_from_disk(dataset_path)
    class_to_objects = defaultdict(set)

    print(f"开始解析 {len(dataset)} 条数据的 reason 字段...")

    for item in dataset:
        reason_text = item.get("reason", "")
        extracted = extract_class_objects_from_reason(reason_text)
        for cls_name, obj_set in extracted.items():
            class_to_objects[cls_name].update(obj_set)

    print(f"解析完成，正在将结果写入: {output_txt_path}")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("🎯 各大类提取出的具体目标汇总:\n" + "=" * 50 + "\n\n")

        for cls_name, obj_set in sorted(class_to_objects.items()):
            if not cls_name:
                continue
            sam_prompt_string = ". ".join(sorted(list(obj_set))) + "."
            f.write(f"【{cls_name}】 (共 {len(obj_set)} 种):\n")
            f.write(f"{sam_prompt_string}\n\n")

        f.write("=" * 50 + "\n")

    print("✅ txt 文件保存成功！可以直接打开查看了。")


if __name__ == "__main__":
    run()
