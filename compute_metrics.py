import json

from fire import Fire
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(input_path: str):
    hyp = []
    ref = []
    with open(input_path, "r") as input_fp:
        for line in input_fp:
            data = json.loads(line)
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


if __name__ == "__main__":
    Fire(compute_metrics)
