import json
import math
import re
from collections import Counter
from typing import Any, Dict, List

from fire import Fire
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Try to import nltk for Self-BLEU, but make it optional
try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Self-BLEU will be skipped.")


def compute_metrics(input_path: str, compute_diversity: bool = False):
    hyp = []
    ref = []
    watermarked_texts = []
    unwatermarked_texts = []

    with open(input_path, "r") as input_fp:
        for line in input_fp:
            data = json.loads(line)

            # Original watermark detection logic
            if data["watermarked_text.is_watermarked"]:
                hyp.append(1)
            else:
                hyp.append(0)
            ref.append(1)
            if data["unwatermarked_text.is_watermarked"]:
                hyp.append(1)
            else:
                hyp.append(0)
            ref.append(0)

            # Collect texts for diversity metrics if requested
            if compute_diversity:
                watermarked_texts.append(data["watermarked_text"])
                unwatermarked_texts.append(data["unwatermarked_text"])

    # Compute original metrics
    print("=== Watermark Detection Metrics ===")
    # Compute Precision, Recall, F1
    precision = precision_score(ref, hyp)
    recall = recall_score(ref, hyp)
    f1 = f1_score(ref, hyp)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

    # Computer FPR, FNR, TPR, TNR
    fpr = false_positive_rate(ref, hyp)
    fnr = false_negative_rate(ref, hyp)
    tpr = true_positive_rate(ref, hyp)
    tnr = true_negative_rate(ref, hyp)
    print(f"FPR: {fpr}, FNR: {fnr}, TPR: {tpr}, TNR: {tnr}")

    # Compute AUC
    auc = roc_auc_score(ref, hyp)
    print(f"AUC: {auc}")

    # Compute diversity metrics if requested
    if compute_diversity:
        print("\n=== Text Diversity Metrics ===")

        print("\n--- Watermarked Texts ---")
        wm_diversity = compute_text_diversity(watermarked_texts)
        print_diversity_metrics(wm_diversity)

        print("\n--- Unwatermarked Texts ---")
        unwm_diversity = compute_text_diversity(unwatermarked_texts)
        print_diversity_metrics(unwm_diversity)


def false_positive_rate(y_true, y_pred):
    """Calculate FPR: FP / (FP + TN)"""
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def false_negative_rate(y_true, y_pred):
    """Calculate FNR: FN / (FN + TP)"""
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    return fn / (fn + tp) if (fn + tp) > 0 else 0


def true_positive_rate(y_true, y_pred):
    """Calculate TPR: TP / (TP + FN)"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def true_negative_rate(y_true, y_pred):
    """Calculate TNR: TN / (TN + FP)"""
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization - split on whitespace and punctuation"""
    # Convert to lowercase and split on whitespace and punctuation
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def compute_ttr(texts: List[str]) -> float:
    """Compute Type-Token Ratio (TTR)"""
    all_tokens = []
    for text in texts:
        tokens = tokenize_text(text)
        all_tokens.extend(tokens)

    if len(all_tokens) == 0:
        return 0.0

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    return unique_tokens / total_tokens


def compute_distinct_n(texts: List[str], n: int) -> float:
    """Compute Distinct-n metric"""
    all_ngrams = []
    for text in texts:
        tokens = tokenize_text(text)
        if len(tokens) >= n:
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

    if len(all_ngrams) == 0:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    return unique_ngrams / total_ngrams


def compute_repetition_rate(texts: List[str], n: int = 2) -> float:
    """Compute repetition rate for n-grams"""
    all_ngrams = []
    for text in texts:
        tokens = tokenize_text(text)
        if len(tokens) >= n:
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

    if len(all_ngrams) == 0:
        return 0.0

    ngram_counts = Counter(all_ngrams)
    repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
    return repeated_ngrams / len(all_ngrams)


def compute_entropy(texts: List[str]) -> float:
    """Compute token entropy"""
    all_tokens = []
    for text in texts:
        tokens = tokenize_text(text)
        all_tokens.extend(tokens)

    if len(all_tokens) == 0:
        return 0.0

    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)

    entropy = 0.0
    for count in token_counts.values():
        prob = count / total_tokens
        entropy -= prob * math.log2(prob)

    return entropy


def compute_self_bleu(texts: List[str], n_samples: int = 100) -> float:
    """Compute Self-BLEU score"""
    if not NLTK_AVAILABLE:
        return -1.0  # Indicate unavailable

    if len(texts) < 2:
        return 0.0

    # Sample texts if too many (for computational efficiency)
    if len(texts) > n_samples:
        import random

        texts = random.sample(texts, n_samples)

    smoothing = SmoothingFunction().method1
    bleu_scores = []

    for i, text in enumerate(texts):
        hypothesis = tokenize_text(text)
        references = [tokenize_text(t) for j, t in enumerate(texts) if i != j]

        if len(hypothesis) > 0 and len(references) > 0:
            try:
                bleu = sentence_bleu(
                    references, hypothesis, smoothing_function=smoothing
                )
                bleu_scores.append(bleu)
            except:
                continue

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


def compute_text_diversity(texts: List[str]) -> Dict[str, Any]:
    """Compute all diversity metrics for a list of texts"""
    metrics = {}

    # TTR
    metrics["TTR"] = compute_ttr(texts)

    # Distinct-n
    metrics["Distinct-1"] = compute_distinct_n(texts, 1)
    metrics["Distinct-2"] = compute_distinct_n(texts, 2)

    # Repetition rates
    metrics["Repetition-2"] = compute_repetition_rate(texts, 2)
    metrics["Repetition-3"] = compute_repetition_rate(texts, 3)

    # Entropy
    metrics["Entropy"] = compute_entropy(texts)

    # Self-BLEU
    metrics["Self-BLEU"] = compute_self_bleu(texts)

    # Basic stats
    avg_length = (
        sum(len(tokenize_text(text)) for text in texts) / len(texts) if texts else 0
    )
    metrics["Avg_Length"] = avg_length

    return metrics


def print_diversity_metrics(metrics: Dict[str, Any]):
    """Pretty print diversity metrics"""
    print(f"TTR (Type-Token Ratio): {metrics['TTR']:.4f}")
    print(f"Distinct-1: {metrics['Distinct-1']:.4f}")
    print(f"Distinct-2: {metrics['Distinct-2']:.4f}")
    print(f"Repetition-2: {metrics['Repetition-2']:.4f}")
    print(f"Repetition-3: {metrics['Repetition-3']:.4f}")
    print(f"Entropy: {metrics['Entropy']:.4f}")
    if metrics["Self-BLEU"] >= 0:
        print(f"Self-BLEU: {metrics['Self-BLEU']:.4f}")
    else:
        print("Self-BLEU: N/A (NLTK not available)")
    print(f"Average Length: {metrics['Avg_Length']:.2f} tokens")


if __name__ == "__main__":
    Fire(compute_metrics)
