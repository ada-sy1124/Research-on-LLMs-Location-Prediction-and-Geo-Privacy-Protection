import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration


def load_model(model_path, torch_dtype_name):
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[torch_dtype_name]
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    return processor.tokenizer, model, processor, None


def build_prompt(processor, prompt):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


def prepare_inputs(model, image_processor, image, baseline_image, prompt):
    prompt_text = build_prompt(image_processor, prompt)
    inputs = image_processor(text=prompt_text, images=image, return_tensors="pt")
    baseline_inputs = image_processor(
        text=prompt_text,
        images=baseline_image,
        return_tensors="pt",
    )

    device = model.device
    dtype = model.dtype
    image_tensor = inputs["pixel_values"].to(device=device, dtype=dtype).detach()
    baseline_tensor = baseline_inputs["pixel_values"].to(device=device, dtype=dtype).detach()
    prompt_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
    }
    return image_tensor, baseline_tensor, prompt_inputs


def generate_coordinate_ids(model, prompt_inputs, image, temperature, top_p, max_new_tokens):
    generation_kwargs = {
        "input_ids": prompt_inputs["input_ids"],
        "attention_mask": prompt_inputs["attention_mask"],
        "pixel_values": image,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    output_ids = model.generate(**generation_kwargs)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    return output_ids[:, prompt_len:]


def postprocess_coordinate_answer(tokenizer, answer_ids, raw_text):
    coord_text = raw_text.strip()
    if coord_text.startswith("[") and coord_text.endswith("]"):
        coord_text = coord_text[1:-1].strip()
    coord_ids = tokenizer(
        coord_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(answer_ids.device)
    return coord_text, coord_ids


def teacher_forced_logprob(model, prompt_inputs, answer_ids, image):
    input_ids = torch.cat((prompt_inputs["input_ids"], answer_ids), dim=1)
    answer_attention = torch.ones_like(answer_ids)
    attention_mask = torch.cat((prompt_inputs["attention_mask"], answer_attention), dim=1)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=image,
        use_cache=False,
        return_dict=True,
    )
    if not torch.isfinite(outputs.logits).all():
        raise FloatingPointError("non-finite logits in teacher-forced coordinate logprob")
    prompt_len = prompt_inputs["input_ids"].shape[1]
    logits = outputs.logits[:, prompt_len - 1 : -1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logps = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(token_logps).all():
        raise FloatingPointError("non-finite token logprob in teacher-forced coordinate logprob")
    return token_logps.sum()
