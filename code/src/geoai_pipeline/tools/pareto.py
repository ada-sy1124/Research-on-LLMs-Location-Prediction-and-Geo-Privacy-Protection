def get_best_mask_category(category_scores, privacy_threshold=0.20, min_area_threshold=0.01):
    valid_candidates = {}

    for cat, scores in category_scores.items():
        if cat == "Nothing":
            continue
        if scores["area_loss"] >= min_area_threshold:
            valid_candidates[cat] = scores

    pareto_front = {}
    for cat1, scores1 in valid_candidates.items():
        is_dominated = False
        for cat2, scores2 in valid_candidates.items():
            if cat1 == cat2:
                continue
            if (
                (scores2["privacy_gain"] >= scores1["privacy_gain"])
                and (scores2["area_loss"] <= scores1["area_loss"])
                and (
                    scores2["privacy_gain"] > scores1["privacy_gain"]
                    or scores2["area_loss"] < scores1["area_loss"]
                )
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_front[cat1] = scores1

    best_category = "Nothing"
    min_area_for_best = float("inf")

    for cat, scores in pareto_front.items():
        if scores["privacy_gain"] >= privacy_threshold and scores["area_loss"] < min_area_for_best:
            best_category = cat
            min_area_for_best = scores["area_loss"]

    return best_category
