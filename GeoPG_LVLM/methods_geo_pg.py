import torch
from torch.autograd import Variable

from geo_pg_score import compose_masked_image, geo_pg_policy_loss
from methods_helper import bilateral_tv_norm, exp_decay, upscale


def geo_pg_igos_mask_optimization(
    args,
    adapter,
    model,
    tokenizer,
    init_mask,
    image,
    baseline,
    prompt_inputs,
    reference_coord,
):
    device = image.device
    masks = torch.ones((1, 1, args.size, args.size), dtype=torch.float32, device=device)
    masks = Variable(masks * init_mask.to(device), requires_grad=True)
    cita = torch.zeros_like(masks.data)
    history = []

    for step in range(args.iterations):
        up_mask = upscale(masks, image, None)
        image_masked = compose_masked_image(image, baseline, up_mask)
        loss_policy, policy_stats = geo_pg_policy_loss(
            args=args,
            adapter=adapter,
            model=model,
            tokenizer=tokenizer,
            prompt_inputs=prompt_inputs,
            image_masked=image_masked,
            reference_coord=reference_coord,
        )

        loss_area = args.L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1).sum()
        loss_tv = args.L3 * bilateral_tv_norm(image, masks, None, tv_beta=2, sigma=0.01).sum()
        l2_weight = exp_decay(args.L2, step, args.gamma)
        loss_l2 = l2_weight * torch.sum((1 - masks) ** 2, dim=[1, 2, 3]).sum()
        loss = loss_policy + loss_area + loss_tv + loss_l2

        loss.backward()
        grad = masks.grad.clone()
        e = step / (step + args.momentum)
        cita_prev = cita
        cita = masks.data - args.lr * grad
        masks.data = cita + e * (cita - cita_prev)
        masks.grad.zero_()
        masks.data.clamp_(0, 1)

        valid_distances = [
            item for item in policy_stats["distances_km"] if item is not None
        ]
        distance_mean_km = (
            sum(valid_distances) / len(valid_distances)
            if len(valid_distances) > 0
            else None
        )
        distance_max_km = max(valid_distances) if len(valid_distances) > 0 else None
        deletion_area = float(torch.mean(1 - masks).detach().cpu())

        record = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(loss_policy.detach().cpu()),
            "area": float(loss_area.detach().cpu()),
            "tv": float(loss_tv.detach().cpu()),
            "l2": float(loss_l2.detach().cpu()),
            "reward_mean": policy_stats["reward_mean"],
            "reward_std": policy_stats["reward_std"],
            "distance_mean_km": distance_mean_km,
            "distance_max_km": distance_max_km,
            "deletion_area": deletion_area,
            "valid_count": int(sum(policy_stats["valid"])),
            "rollouts": policy_stats,
        }
        history.append(record)
        d_mean = "nan" if distance_mean_km is None else f"{distance_mean_km:.2f}"
        d_max = "nan" if distance_max_km is None else f"{distance_max_km:.2f}"
        print(
            "step={step} loss={loss:.6f} policy={policy_loss:.6f} "
            "reward_mean={reward_mean:.6f} reward_std={reward_std:.6f} "
            "hav_mean_km={d_mean} hav_max_km={d_max} "
            "del_area={deletion_area:.4f} valid={valid_count}".format(
                d_mean=d_mean,
                d_max=d_max,
                **record,
            )
        )

    return masks.detach(), history
