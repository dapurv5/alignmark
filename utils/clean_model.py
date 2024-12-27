import glob
import os
import shutil
import time

import fire


def clean_model(model_name):
    model_last_name = model_name.split("/")[-1]
    # Get huggingface cache directory
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    # Find any directories with the name model_last_name in its name
    model_dirs = glob.glob(f"{cache_dir}/*{model_last_name}*")
    for model_dir in model_dirs:
        # Delete all files and subdirectories in the model directory
        for file in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, file)):
                shutil.rmtree(os.path.join(model_dir, file))
            else:
                os.remove(os.path.join(model_dir, file))
    # Delete the model directory
    os.rmdir(model_dir)
    time.sleep(10)


if __name__ == "__main__":
    fire.Fire(clean_model)
