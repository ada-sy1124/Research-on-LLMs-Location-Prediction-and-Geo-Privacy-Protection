# Install the PyTorch/CUDA build that matches the machine first.
pip3 install -U "transformers>=4.49.0" accelerate sentencepiece protobuf safetensors huggingface_hub
pip3 install -U pillow opencv-python matplotlib scipy yake requests
pip3 install -U --force-reinstall "numpy==1.26.4" "scikit-learn==1.5.2"
pip3 install -U pandas tqdm
