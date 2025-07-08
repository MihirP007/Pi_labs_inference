from huggingface_hub import hf_hub_download,login
login(token = "HF_TOKEN_REMOVED")
hf_hub_download(
    repo_id="google/gemma-3-1b-it-qat-q4_0-gguf",
    filename="google/gemma-3-1b-it-qat-q4_0-gguf"
)
