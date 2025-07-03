import os

from huggingface_hub import snapshot_download

current_dir = os.path.dirname(os.path.abspath(__file__))

snapshot_download(
    repo_id="tencent/SongGeneration",
    local_dir=current_dir,
    local_dir_use_symlinks=False,
)
