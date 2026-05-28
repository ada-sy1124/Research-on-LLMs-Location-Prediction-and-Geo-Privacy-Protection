import html
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from PIL import Image


# ===== Edit these parameters before running =====
DATASET_PATH = "./data/YES_Mask_selected_id_flat"

# Choose one or both:
# - ROW_RANGE examples: "0:10", "5:20", "7"; set to None to use the first 5 rows.
# - IDS examples: ["yes_000000", "yes_000123"]; set to [] to ignore IDs.
ROW_RANGE = "0:10"
IDS: list[str] = []

# Set to None to auto-detect from alignment_id/sample_id/id.
ID_COLUMN = None

# Set to None to display all columns.
COLUMNS = ["alignment_id", "target_mask_category", "q_ratio", "d_diff", "image"]

OUTPUT_DIR = "./dataset_preview/yes_mask_selected_flat"
MAX_TEXT_LENGTH = 1000

# Optional before/after comparison.
# For YES_Mask_selected_id_flat, image is the masked image, and YES_id provides the original image.
BEFORE_DATASET = "./data/YES_id"
BEFORE_ID_COLUMN = None
BEFORE_IMAGE_COLUMN = "image"
AFTER_IMAGE_COLUMN = "image"


IMAGE_EXT = ".jpg"
DEFAULT_ID_COLUMNS = ("alignment_id", "sample_id", "id")


def safe_name(value: Any) -> str:
    text = str(value)
    for old, new in (("/", "_"), ("\\", "_"), (" ", "_"), (":", "_"), ("|", "_")):
        text = text.replace(old, new)
    return text[:120]


def get_id_column(dataset, explicit: str | None) -> str:
    if explicit:
        if explicit not in dataset.column_names:
            raise KeyError(f"ID column not found: {explicit}. Columns: {dataset.column_names}")
        return explicit
    for column in DEFAULT_ID_COLUMNS:
        if column in dataset.column_names:
            return column
    raise KeyError(f"No ID column found. Please pass --id-column. Columns: {dataset.column_names}")


def parse_range(row_range: str | None, dataset_len: int) -> list[int]:
    if not row_range:
        return list(range(min(5, dataset_len)))
    if ":" not in row_range:
        index = int(row_range)
        return [index]
    start_text, end_text = row_range.split(":", 1)
    start = int(start_text) if start_text else 0
    end = int(end_text) if end_text else dataset_len
    start = max(start, 0)
    end = min(end, dataset_len)
    if start >= end:
        return []
    return list(range(start, end))


def build_id_index(dataset, id_column: str) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, value in enumerate(dataset[id_column]):
        value_text = str(value)
        if value_text not in index:
            index[value_text] = i
    return index


def resolve_indices(dataset, row_range: str | None, ids: str | None, id_column: str | None) -> list[int]:
    indices = parse_range(row_range, len(dataset))
    if not ids:
        return indices

    resolved_id_column = get_id_column(dataset, id_column)
    id_index = build_id_index(dataset, resolved_id_column)
    for sample_id in [part.strip() for part in ids.split(",") if part.strip()]:
        if sample_id not in id_index:
            print(f"[WARN] ID not found in {resolved_id_column}: {sample_id}")
            continue
        indices.append(id_index[sample_id])

    seen = set()
    unique_indices = []
    for index in indices:
        if index not in seen:
            seen.add(index)
            unique_indices.append(index)
    return unique_indices


def value_to_text(value: Any, max_length: int) -> str:
    if isinstance(value, Image.Image):
        return "<image>"
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    if len(text) > max_length:
        return text[:max_length] + "...[truncated]"
    return text


def save_image(value: Any, path: Path) -> bool:
    if isinstance(value, Image.Image):
        value.convert("RGB").save(path)
        return True
    if isinstance(value, dict) and "bytes" in value:
        path.write_bytes(value["bytes"])
        return True
    return False


def render_image_cell(image_paths: list[Path], output_dir: Path) -> str:
    if not image_paths:
        return ""
    tags = []
    for image_path in image_paths:
        rel_path = image_path.relative_to(output_dir).as_posix()
        tags.append(f'<a href="{html.escape(rel_path)}"><img src="{html.escape(rel_path)}"></a>')
    return "".join(tags)


def export_column_images(sample: dict, row_index: int, columns: list[str], output_dir: Path) -> dict[str, list[Path]]:
    exported: dict[str, list[Path]] = {}
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for column in columns:
        if column not in sample:
            continue
        value = sample[column]
        paths: list[Path] = []
        if isinstance(value, Image.Image) or (isinstance(value, dict) and "bytes" in value):
            path = image_dir / f"row_{row_index:06d}_{safe_name(column)}{IMAGE_EXT}"
            if save_image(value, path):
                paths.append(path)
        elif isinstance(value, list):
            for item_index, item in enumerate(value):
                path = image_dir / f"row_{row_index:06d}_{safe_name(column)}_{item_index:02d}{IMAGE_EXT}"
                if save_image(item, path):
                    paths.append(path)
        exported[column] = paths
    return exported


def normalize_ids(ids: list[str] | str | None) -> str | None:
    if ids is None:
        return None
    if isinstance(ids, str):
        return ids
    return ",".join(ids)


def build_before_lookup():
    if not BEFORE_DATASET:
        return None, None
    before_dataset = load_from_disk(BEFORE_DATASET)
    before_id_column = get_id_column(before_dataset, BEFORE_ID_COLUMN)
    return before_dataset, build_id_index(before_dataset, before_id_column)


def main() -> None:
    dataset = load_from_disk(DATASET_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = dataset.column_names if COLUMNS is None else COLUMNS
    ids_text = normalize_ids(IDS)
    indices = resolve_indices(dataset, ROW_RANGE, ids_text, ID_COLUMN)
    before_dataset, before_index = build_before_lookup()
    id_column = get_id_column(dataset, ID_COLUMN) if (ids_text or before_dataset is not None) else None

    rows_html = []
    text_lines = [
        f"DATASET_PATH = {DATASET_PATH}",
        f"ROWS = {len(dataset)}",
        f"COLUMNS = {dataset.column_names}",
        f"SELECTED_INDICES = {indices}",
        "",
    ]

    for row_index in indices:
        sample = dataset[row_index]
        sample_id = str(sample.get(id_column, row_index)) if id_column else str(row_index)
        image_paths_by_column = export_column_images(sample, row_index, columns, output_dir)

        before_paths: list[Path] = []
        if before_dataset is not None and before_index is not None:
            before_row_index = before_index.get(sample_id)
            if before_row_index is not None:
                before_sample = before_dataset[before_row_index]
                before_image = before_sample.get(BEFORE_IMAGE_COLUMN)
                before_path = output_dir / "images" / f"row_{row_index:06d}_before_{safe_name(sample_id)}{IMAGE_EXT}"
                if save_image(before_image, before_path):
                    before_paths.append(before_path)

        after_paths = image_paths_by_column.get(AFTER_IMAGE_COLUMN, [])
        text_lines.append(f"===== row {row_index} | id {sample_id} =====")
        for column in columns:
            text_lines.append(f"{column}: {value_to_text(sample.get(column, 'COLUMN NOT FOUND'), MAX_TEXT_LENGTH)}")
        text_lines.append("")

        before_after_html = ""
        if before_paths or after_paths:
            before_after_html = (
                "<div class='compare'>"
                f"<div><h4>before</h4>{render_image_cell(before_paths, output_dir)}</div>"
                f"<div><h4>after</h4>{render_image_cell(after_paths, output_dir)}</div>"
                "</div>"
            )

        column_items = []
        for column in columns:
            value = sample.get(column, "COLUMN NOT FOUND")
            image_html = render_image_cell(image_paths_by_column.get(column, []), output_dir)
            text = html.escape(value_to_text(value, MAX_TEXT_LENGTH))
            column_items.append(f"<tr><th>{html.escape(column)}</th><td>{image_html}<pre>{text}</pre></td></tr>")

        rows_html.append(
            "<section>"
            f"<h2>row {row_index} | {html.escape(sample_id)}</h2>"
            f"{before_after_html}"
            f"<table>{''.join(column_items)}</table>"
            "</section>"
        )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset Preview</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
section {{ border-top: 1px solid #d1d5db; padding: 18px 0; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
th {{ width: 220px; background: #f3f4f6; text-align: left; }}
pre {{ white-space: pre-wrap; word-break: break-word; margin: 6px 0 0; }}
img {{ max-width: 520px; max-height: 360px; margin: 4px 8px 4px 0; border: 1px solid #d1d5db; }}
.compare {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin: 12px 0; }}
.compare h4 {{ margin: 0 0 6px; }}
</style>
</head>
<body>
<h1>Dataset Preview</h1>
<p><b>Dataset:</b> {html.escape(DATASET_PATH)}</p>
<p><b>Rows:</b> {len(dataset)} | <b>Columns:</b> {html.escape(', '.join(dataset.column_names))}</p>
{''.join(rows_html)}
</body>
</html>
"""

    text_path = output_dir / "preview_text.txt"
    html_path = output_dir / "report.html"
    text_path.write_text("\n".join(text_lines), encoding="utf-8")
    html_path.write_text(report_html, encoding="utf-8")

    print(f"Saved text preview: {text_path.resolve()}")
    print(f"Saved HTML report: {html_path.resolve()}")
    print(f"Saved images under: {(output_dir / 'images').resolve()}")


if __name__ == "__main__":
    main()
