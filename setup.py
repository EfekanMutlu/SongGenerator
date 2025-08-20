import os
from huggingface_hub import snapshot_download

current_dir = os.path.dirname(os.path.abspath(__file__))

ckpt_dir = os.path.join(current_dir, "ckpt")

os.makedirs(ckpt_dir, exist_ok=True)

snapshot_download(
    repo_id="mtlefess/SongGenerator",
    local_dir=ckpt_dir,
    local_dir_use_symlinks=False,
)