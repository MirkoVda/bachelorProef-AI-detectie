import numpy as np

def most_confident_errors(labels, probs, paths, top_k=100):
    """
    Selecteert de `top_k` voorbeelden waarvoor het model het meest zeker
    was, maar foutief voorspelde. Wordt gebruikt om systematische fouten
    of dataset issues te analyseren.
    Args:
        labels: lijst van ware labels
        probs: lijst van \(P(\text{fake})\) als float
        paths: padnamen van de voorbeelden
    Returns:
        lijst van tuples (path, true_label, pred_prob)
    """
    preds = (np.array(probs) > 0.5).astype(int)
    incorrect = preds != np.array(labels)
    incorrect_indices = np.where(incorrect)[0]
    
    # Bereken confidence van de foutieve voorbeelden
    confs = np.abs(np.array(probs)[incorrect_indices] - 0.5) + 0.5
    sorted_idx = np.argsort(-confs)  # aflopende
    chosen = incorrect_indices[sorted_idx][:top_k]
    
    return [(paths[i], labels[i], probs[i]) for i in chosen]
