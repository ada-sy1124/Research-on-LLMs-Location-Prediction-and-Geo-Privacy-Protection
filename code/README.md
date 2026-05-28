# GeoAI Pipeline

`code/` 目录现在只保留主流程入口；具体逻辑集中在 `src/geoai_pipeline/`。

## Structure

- `src/geoai_pipeline/config.py`: `.env` configuration helpers
- `src/geoai_pipeline/constants.py`: prompts and class mappings
- `src/geoai_pipeline/tools/`: shared dataset, local-model, and geo utilities
- `src/geoai_pipeline/pipelines/`: main data pipeline implementations
- `模型训练/QLoRA.py`: optional QLoRA training example

## Run

Run from this directory:

```bash
python 本地模型筛数据集构建YESandNO.py
python 从YES构建afterSAM.py
python 本地模型从afterSAM构建YES_Mask.py
python 类别筛选.py
python 训练集文本构建.py
```

The pipeline reads `code/.env` when present. Copy `.env.example` to `.env`, set `LOCAL_MODEL_PATH` to your local multimodal model directory, and adjust paths plus index ranges for your machine. The default local backend is `MODEL_BACKEND=swift`.
