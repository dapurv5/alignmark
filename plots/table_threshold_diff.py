import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from fire import Fire

from plots.plot_utils import process_files


@dataclass
class ScoreData:
    example_idx: int
    watermarked_scores: list[float]  # its length is the number of seeds
    unwatermarked_scores: list[float]  # its length is the number of seeds


def _process_scores_for_strength(
    watermarked_scores: list[float],
    unwatermarked_scores: list[float],
    example_scores: list[ScoreData],
) -> None:
    """Process scores for a single watermark strength."""
    for example_idx in range(len(watermarked_scores)):
        if len(example_scores) < example_idx + 1:
            example_scores.append(
                ScoreData(
                    example_idx=example_idx,
                    watermarked_scores=[],
                    unwatermarked_scores=[],
                )
            )
        example_scores[example_idx].watermarked_scores.append(
            watermarked_scores[example_idx]
        )
        example_scores[example_idx].unwatermarked_scores.append(
            unwatermarked_scores[example_idx]
        )


def collect_scores_by_example_across_seeds(data: dict) -> dict[float, list[ScoreData]]:
    res = {}
    for (watermark_type, dataset_name), seeds_data in data.items():
        print(f"Reading data for {watermark_type} on {dataset_name} and tabulating")
        example_scores_across_seeds: dict[float, list[ScoreData]] = {}
        for seed, values in seeds_data.items():
            wm_strengths, watermarked_scores, unwatermarked_scores = zip(*values)
            if not example_scores_across_seeds:
                example_scores_across_seeds: dict[float, list[ScoreData]] = {
                    strength: [] for strength in wm_strengths
                }
            for idx, wm_strength in enumerate(wm_strengths):
                _process_scores_for_strength(
                    watermarked_scores[idx],
                    unwatermarked_scores[idx],
                    example_scores_across_seeds[wm_strength],
                )
        res[(watermark_type, dataset_name)] = example_scores_across_seeds
    return res


def _calculate_pair_counts(
    wm_scores: list[float],
    unwm_scores: list[float],
    wm_mean: float,
    wm_std: float,
    threshold_diff: float,
) -> tuple[int, int, int]:
    """Calculate counts of pairs meeting different criteria."""
    cnt_feasible_pairs = cnt_1std_away = cnt_2std_away = 0
    uniq_wm_examples_used = set()
    uniq_unwm_examples_used = set()
    for i, wm_score in enumerate(wm_scores):
        for j, unwm_score in enumerate(unwm_scores):
            if wm_score - unwm_score > threshold_diff:
                cnt_feasible_pairs += 1
                if wm_score - wm_mean > 1 * wm_std:
                    cnt_1std_away += 1
                    uniq_wm_examples_used.add(i)
                    uniq_unwm_examples_used.add(j)
                if wm_score - wm_mean > 2 * wm_std:
                    cnt_2std_away += 1
    return (
        cnt_feasible_pairs,
        cnt_1std_away,
        cnt_2std_away,
        len(uniq_wm_examples_used),
        len(uniq_unwm_examples_used),
    )


def _create_score_row(
    example_score: ScoreData,
    threshold_diff: float,
) -> dict:
    """Create a single row of score data."""
    wm_scores = example_score.watermarked_scores
    unwm_scores = example_score.unwatermarked_scores
    wm_mean, wm_std = np.mean(wm_scores), np.std(wm_scores)
    unwm_mean, unwm_std = np.mean(unwm_scores), np.std(unwm_scores)

    (
        cnt_feasible_pairs,
        cnt_1std_away,
        cnt_2std_away,
        uniq_wm_examples_used,
        uniq_unwm_examples_used,
    ) = _calculate_pair_counts(wm_scores, unwm_scores, wm_mean, wm_std, threshold_diff)

    return {
        "idx": example_score.example_idx,
        "cnt_feasible_pairs": cnt_feasible_pairs,
        "cnt_1std_away": cnt_1std_away,
        "cnt_2std_away": cnt_2std_away,
        "cnt_1std_away_gt1": 1 if cnt_1std_away > 1 else 0,
        "cnt_2std_away_gt1": 1 if cnt_2std_away > 1 else 0,
        "uniq_wm_examples_used": uniq_wm_examples_used,
        "uniq_unwm_examples_used": uniq_unwm_examples_used,
        "unwm_mean": unwm_mean,
        "unwm_std": unwm_std,
        "wm_mean": wm_mean,
        "wm_std": wm_std,
        "cnt_total_pairs": len(wm_scores) * len(unwm_scores),
    }


def construct_table(data: dict, threshold_diff: float, output_dir: str):
    res = collect_scores_by_example_across_seeds(data)
    for (watermark_type, dataset_name), example_scores_across_seeds in res.items():
        print(f"Watermark type: {watermark_type}, dataset: {dataset_name}")
        for wm_strength, example_scores in example_scores_across_seeds.items():
            print(f"Watermark strength: {wm_strength}")
            rows = [
                _create_score_row(score, threshold_diff) for score in example_scores
            ]
            df = pd.DataFrame(rows)
            df.to_csv(
                os.path.join(
                    output_dir,
                    f"table_{watermark_type}_{dataset_name}_{wm_strength}.csv",
                ),
                index=False,
            )
            # Compute averages and medians and print those as two separate rows
            print(df)
            print("Means:")
            avg_row = df.mean()
            print(avg_row)
            print("Medians:")
            med_row = df.median()
            print(med_row)


def main(
    input_dir: str,
    model_name_to_plot: str = "Llama-3.1-8B-Instruct",
    score_name: str = "reward",  # or "truthfulness",
    threshold_diff: float = 0.1,
):
    output_dir = os.path.join(input_dir, "tables")
    os.makedirs(output_dir, exist_ok=True)
    data = process_files(input_dir, model_name_to_plot, "temperature", score_name)
    construct_table(data, threshold_diff, output_dir)


if __name__ == "__main__":
    Fire(main)
