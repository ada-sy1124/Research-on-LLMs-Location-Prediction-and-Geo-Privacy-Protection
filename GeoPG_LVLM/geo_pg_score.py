import torch

from geo_pg_geo import clipped_haversine_reward, parse_coordinate, reward_to_advantage
from methods_helper import phi


def strip_edge_special_tokens(token_ids, tokenizer):
    ids = token_ids
    if ids.dim() == 2:
        ids = ids[0]
    ids = ids.clone()
    if ids.numel() > 0 and tokenizer.bos_token_id is not None and int(ids[0]) == int(tokenizer.bos_token_id):
        ids = ids[1:]
    if ids.numel() > 0 and tokenizer.eos_token_id is not None and int(ids[-1]) == int(tokenizer.eos_token_id):
        ids = ids[:-1]
    return ids.unsqueeze(0)


def compose_masked_image(image, baseline, up_mask):
    return phi(image, baseline, up_mask).to(dtype=image.dtype)


def generate_coordinate_text(
    adapter,
    model,
    tokenizer,
    prompt_inputs,
    image,
    temperature,
    top_p,
    max_new_tokens,
):
    output_ids = adapter.generate_coordinate_ids(
        model=model,
        prompt_inputs=prompt_inputs,
        image=image,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    answer_ids = strip_edge_special_tokens(output_ids, tokenizer).to(
        prompt_inputs["input_ids"].device
    )
    raw_text = tokenizer.decode(answer_ids[0], skip_special_tokens=True).strip()
    if hasattr(adapter, "postprocess_coordinate_answer"):
        text, answer_ids = adapter.postprocess_coordinate_answer(
            tokenizer,
            answer_ids,
            raw_text,
        )
    else:
        text = raw_text
    return text, answer_ids, output_ids, raw_text


def generate_reference_coordinate(args, adapter, model, tokenizer, prompt_inputs, image):
    text, answer_ids, raw_ids, raw_text = generate_coordinate_text(
        adapter=adapter,
        model=model,
        tokenizer=tokenizer,
        prompt_inputs=prompt_inputs,
        image=image,
        temperature=0.0,
        top_p=args.geo_top_p,
        max_new_tokens=args.geo_max_new_tokens,
    )
    return parse_coordinate(text), raw_text, answer_ids, raw_ids


def geo_pg_policy_loss(
    args,
    adapter,
    model,
    tokenizer,
    prompt_inputs,
    image_masked,
    reference_coord,
):
    texts = []
    answer_ids_list = []
    coords = []
    rewards = []
    distances = []
    valid = []
    image_for_rollout = image_masked.detach()

    with torch.no_grad():
        for _ in range(args.geo_rollouts):
            text, answer_ids, _, raw_text = generate_coordinate_text(
                adapter=adapter,
                model=model,
                tokenizer=tokenizer,
                prompt_inputs=prompt_inputs,
                image=image_for_rollout,
                temperature=args.geo_temperature,
                top_p=args.geo_top_p,
                max_new_tokens=args.geo_max_new_tokens,
            )
            try:
                coord = parse_coordinate(text)
                reward, distance = clipped_haversine_reward(
                    coord, reference_coord, args.geo_dmax_km
                )
                is_valid = True
            except ValueError:
                coord = None
                reward = float(args.geo_invalid_reward)
                distance = None
                is_valid = False

            texts.append(raw_text)
            answer_ids_list.append(answer_ids)
            coords.append(coord)
            rewards.append(float(reward))
            distances.append(distance)
            valid.append(is_valid)

    rewards_t, adv, reward_std = reward_to_advantage(
        rewards,
        prompt_inputs["input_ids"].device,
        args.geo_adv_eps,
        args.geo_clip_adv,
        args.geo_min_reward_std,
    )

    logps = []
    zero = image_masked.sum() * 0.0
    for answer_ids, is_valid in zip(answer_ids_list, valid):
        if is_valid:
            logps.append(
                adapter.teacher_forced_logprob(
                    model=model,
                    prompt_inputs=prompt_inputs,
                    answer_ids=answer_ids,
                    image=image_masked,
                )
            )
        else:
            logps.append(zero)

    logps = torch.stack(logps)
    loss = -(adv * logps).mean()
    stats = {
        "texts": texts,
        "coords": [None if coord is None else [coord.lat, coord.lon] for coord in coords],
        "valid": valid,
        "rewards": rewards,
        "distances_km": [None if item is None else float(item) for item in distances],
        "reward_mean": float(rewards_t.mean().detach().cpu()),
        "reward_std": float(reward_std.detach().cpu()),
        "advantages": [float(item) for item in adv.detach().cpu().tolist()],
        "logps": [float(item) for item in logps.detach().cpu().tolist()],
        "policy_loss": float(loss.detach().cpu()),
    }
    return loss, stats
