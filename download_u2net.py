from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="akhaliq/U-2-Net",
    filename="u2net.pth",
    local_dir="./backend/ai_models/checkpoints/u2net",
    local_dir_use_symlinks=False,
)