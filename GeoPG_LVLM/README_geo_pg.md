# Geo-PG LVLM

This folder is built on top of the useful mask/perturbation utilities from
`LVLM_Interpretation_raw`.

Raw project files used by the Geo-PG path:

- `utils.py`
- `methods_helper.py`

Geo-PG adds a separate geolocation attribution path:

- `main_geo_pg.py`
- `args_geo_pg.py`
- `geo_pg_geo.py`
- `geo_pg_score.py`
- `methods_geo_pg.py`
- `geo_pg_eval.py`
- `model_adapters/`

The current model path is Hugging Face native:

- `model_adapters/llava_hf.py`

The main loop is:

```text
original image -> greedy coordinate reference g0
mask -> masked image -> N sampled coordinate answers
parse strict "lat, lon" -> Haversine reward -> group advantage
teacher-forced coordinate answer logprob -> policy loss -> mask update
```

Run:

```bash
python3 main_geo_pg.py \
  --model llava_hf \
  --model_path llava-hf/llava-1.5-13b-hf \
  --data_path /path/to/data.jsonl \
  --image_folder /path/to/images \
  --output_dir /path/to/output \
  --size 32 \
  --L1 1.0 \
  --L2 0.1 \
  --L3 10.0 \
  --gamma 1.0 \
  --iterations 5 \
  --momentum 5 \
  --geo_rollouts 8 \
  --geo_temperature 0.3
```

Input JSONL:

```json
{"id": "sample_0001", "image_path": "sample_0_part2.jpg"}
```

`image_path` is resolved under `--image_folder`.
