import numpy as np
from sklearn.metrics import f1_score


def optimize_threshold(labels, probs, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, 0.0
    for t in grid:
        preds = (np.array(probs) >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# Voorbeeld gebruik
best_t, best_f1 = optimize_threshold(all_labels, all_probs)
print(f"Beste drempel: {best_t:.2f} met F1={best_f1:.4f}")
