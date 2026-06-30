def get_model_adapter(model_name):
    if model_name == "llava_hf":
        from model_adapters import llava_hf

        return llava_hf
    raise ValueError(f"unsupported model: {model_name}")
