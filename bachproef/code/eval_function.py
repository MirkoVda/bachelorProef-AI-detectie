import torch
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


def evaluate_model(arch: str, test_loader) -> dict:
    """
    Laadt het beste model en berekent alle kernmetrieken.
    """
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(
        OUTPUT_DIR / 'models' / f'{arch}_best.pt'))
    model.eval()
    
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            
            # Softmax converteert logits naar probabilities [0,1]
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            preds = logits.argmax(dim=1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Berekening metrieken
    report = classification_report(all_labels, all_preds,
                                    output_dict=True)
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    
    return {
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'report': report,
        'auc': auc,
        'ap': ap,
    }
