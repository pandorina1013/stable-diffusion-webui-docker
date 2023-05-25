import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "lllyasviel/ControlNet-v1-1"

FILENAMES = [
    "control_v11p_sd15_softedge",
    "control_v11f1p_sd15_depth",
    "control_v11p_sd15_canny",
    "control_v11p_sd15_lineart",
    "control_v11p_sd15_scribble",
]



for f_name in FILENAMES:
    for f_ext in [".pth", ".yaml"]:
        download_dir = hf_hub_download(
            repo_id=REPO_ID, 
            filename=f"{f_name}{f_ext}"
        )

        shutil.copyfile(download_dir, f"./data/config/auto/extensions/sd-webui-controlnet/models/{f_name}{f_ext}")
        print(f"download {f_name}{f_ext}")