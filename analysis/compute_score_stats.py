import glob
import json
import os

import numpy as np
from fire import Fire

from plots.plot_utils import parse_filename


def main(input_dir: str, score_name: str = "reward"):
    """
    Compute statistics for the score
    """
    files = glob.glob(os.path.join(input_dir, f"*_{score_name}*.jsonl"))
    print(
        f"{'model_name':<20} {'watermark_type':<15} {'temperature':<12} {'mean_wm_score':<15} {'mean_unwm_score':<15} {'std_wm_score':<15} {'std_unwm_score':<15}"
    )
    # Collect all stats first
    stats = []
    for file in files:
        filename = os.path.basename(file)
        parse_info = parse_filename(filename)
        watermark = parse_info["watermark_type"]
        model = parse_info["model_name"]
        temperature = parse_info["temperature"]
        wm_score_list = []
        unwm_score_list = []
        with open(file, "r") as input_fp:
            for line in input_fp:
                line = line.strip()
                data = json.loads(line)
                wm_score = data[f"watermarked_text_{score_name}_score"]
                unwm_score = data[f"unwatermarked_text_{score_name}_score"]
                wm_score_list.append(np.mean(wm_score))
                unwm_score_list.append(np.mean(unwm_score))

        stats.append(
            {
                "model": model,
                "watermark": watermark,
                "temperature": float(temperature),
                "wm_mean_mean": np.mean(wm_score_list),
                "unwm_mean_mean": np.mean(unwm_score_list),
                "wm_mean_std": np.std(wm_score_list),
                "unwm_mean_std": np.std(unwm_score_list),
            }
        )

    # Sort by temperature then model name
    stats.sort(key=lambda x: (x["temperature"], x["model"]))

    # Print sorted stats
    for stat in stats:
        print("-" * 107)
        print(
            f"{stat['model']:<25} {stat['watermark']:<20} {stat['temperature']:<12} "
            f"{stat['wm_mean_mean']:15.4f} {stat['unwm_mean_mean']:15.4f} "
            f"{stat['wm_mean_std']:15.4f} {stat['unwm_mean_std']:15.4f} "
            f"{stat['wm_mean_std'] > stat['unwm_mean_std']}"
        )


if __name__ == "__main__":
    Fire(main)
