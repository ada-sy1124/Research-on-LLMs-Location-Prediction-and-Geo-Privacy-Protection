import argparse


GEO_PROMPT = (
    """<context>
I don't remember where the photo was taken. Could you help me find it? Estimate the latitude and longitude. 
If you are not sure about the specific location, you MUST give your best possible candidate without asking any further questions.
</context>

<formatting_rules>
CRITICAL: Do NOT use brackets, parentheses, or any other characters. Output ONLY the coordinate string, no other words.
You MUST output EXACTLY matching this format:
Template:
{latitude}, {longitude}
</formatting_rules>"""
)


def init_geo_pg_args():
    parser = argparse.ArgumentParser(description="Geo-PG LVLM geolocation attribution.")

    parser.add_argument("--model", choices=["llava_hf"], default="llava_hf")
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-13b-hf")
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=336)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--momentum", type=int, default=5)
    parser.add_argument("--L1", type=float, default=1.0)
    parser.add_argument("--L2", type=float, default=0.1)
    parser.add_argument("--L3", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--diverse_k", type=int, default=1)
    parser.add_argument("--init_posi", type=int, default=0)
    parser.add_argument("--init_val", type=float, default=0.0)
    parser.add_argument("--manual_seed", type=int, default=0)

    parser.add_argument("--geo_prompt", type=str, default=GEO_PROMPT)
    parser.add_argument("--geo_rollouts", type=int, default=8)
    parser.add_argument("--geo_temperature", type=float, default=0.3)
    parser.add_argument("--geo_top_p", type=float, default=1.0)
    parser.add_argument("--geo_max_new_tokens", type=int, default=32)
    parser.add_argument("--geo_dmax_km", type=float, default=20000.0)
    parser.add_argument("--geo_invalid_reward", type=float, default=-0.2)
    parser.add_argument("--geo_adv_eps", type=float, default=1e-6)
    parser.add_argument("--geo_clip_adv", type=float, default=3.0)
    parser.add_argument("--geo_min_reward_std", type=float, default=1e-8)
    parser.add_argument("--geo_eval_topk", type=str, default="5,10,20,30")
    parser.add_argument("--geo_random_repeats", type=int, default=10)

    return parser.parse_args()






# python3 main_geo_pg.py \
#   --model llava_hf \
#   --model_path llava-hf/llava-1.5-13b-hf \
#   --data_path /root/autodl-tmp/train.jsonl \
#   --image_folder /root/autodl-tmp/GSV/geolocation-inference-dataset/imgs_final_v1 \
#   --output_dir /root/autodl-tmp/GeoPG_LVLM/outputs \
#   --size 32 \
#   --L1 1.0 \
#   --L2 0.1 \
#   --L3 10.0 \
#   --gamma 1.0 \
#   --iterations 5 \
#   --momentum 5 \
#   --geo_rollouts 8 \
#   --geo_temperature 0.3




# python3 main_geo_pg.py --model llava_hf --model_path llava-hf/llava-1.5-13b-hf --data_path /root/autodl-tmp/train.jsonl --image_folder /root/autodl-tmp/GSV/geolocation-inference-dataset/imgs_final_v1 --output_dir /root/autodl-tmp/GeoPG_LVLM/outputs --size 24 --lr 2.0 --L1 3.0 --L2 0.2 --L3 50.0 --gamma 1.0 --iterations 20 --momentum 5 --geo_rollouts 8 --geo_temperature 0.2

# python3 main_geo_pg.py --model llava_hf --model_path llava-hf/llava-1.5-13b-hf --torch_dtype float16 --data_path /root/autodl-tmp/train.jsonl --image_folder /root/autodl-tmp/GSV/geolocation-inference-dataset/imgs_final_v1 --output_dir /root/autodl-tmp/GeoPG_LVLM/outputs --size 24 --lr 0.5 --L1 10.0 --L2 1.0 --L3 30.0 --gamma 0.05 --iterations 25 --momentum 50 --geo_rollouts 8 --geo_temperature 0.2 --geo_min_reward_std 1e-4

# python3 main_geo_pg.py --model llava_hf --model_path llava-hf/llava-1.5-13b-hf --torch_dtype float16 --data_path /root/autodl-tmp/train.jsonl --image_folder /root/autodl-tmp/GSV/geolocation-inference-dataset/imgs_final_v1 --output_dir /root/autodl-tmp/GeoPG_LVLM/outputs --size 24 --lr 0.3 --L1 12.0 --L2 1.0 --L3 4.0 --gamma 0.0 --iterations 15 --momentum 15 --geo_rollouts 8 --geo_temperature 0.15 --geo_min_reward_std 1e-4