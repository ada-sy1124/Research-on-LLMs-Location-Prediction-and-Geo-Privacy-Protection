import re
from collections import defaultdict


def extract_class_objects_from_reason(reason_text: str):
    class_to_objects = defaultdict(set)
    if not reason_text:
        return class_to_objects

    for seg in reason_text.split(";"):
        seg = seg.strip()
        if ":" not in seg:
            continue
        cls_name, objs_text = seg.split(":", 1)
        cls_name = cls_name.strip()

        for obj in objs_text.split(","):
            cleaned_obj = re.sub(r"\s*#\d+", "", obj).strip().lower()
            if cleaned_obj:
                class_to_objects[cls_name].add(cleaned_obj)

    return class_to_objects
