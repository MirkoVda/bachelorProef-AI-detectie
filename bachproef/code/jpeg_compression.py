import io
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score


def apply_jpeg_compression(img: Image.Image, 
                          quality: int) -> Image.Image:
    """Pas JPEG-compressie toe op een PIL-afbeelding.
    
    Args:
        quality: 100 = geen compressie, 10 = zeer lossy
    """
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def robustness_jpeg(arch: str, test_ds: DeepfakeDataset,
                    qualities: list = [90, 70, 50, 30, 10]):
    """
    Evalueert het model onder toenemende JPEG-compressie.
    
    Return: dict {quality: f1_score}
    """
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(
        OUTPUT_DIR / 'models' / f'{arch}_best.pt'))
    model.eval()
    
    results = {}
    
    for quality in qualities:
        all_preds, all_labels = [], []
        
        for path, label in tqdm(test_ds.samples,
                               desc=f"JPEG q={quality}"):
            try:
                # Laad origineel
                img = Image.open(path).convert('RGB')
                
                # Pas compressie toe
                img = apply_jpeg_compression(img, quality)
                
                # Pre-process en voorspel
                tensor = val_transform(img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logit = model(tensor)
                    pred = logit.argmax(dim=1).item()
                
                all_preds.append(pred)
                all_labels.append(label)
                
            except Exception:
                continue
        
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        results[quality] = f1
        print(f"[{arch.upper()}] JPEG q={quality}: F1={f1:.4f}")
    
    return results


# Voer analyse uit
robustness_results = {}
for arch in ['cnn', 'vit', 'hybrid']:
    robustness_results[arch] = robustness_jpeg(arch, test_ds)

# Visualisatie
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
for arch, results in robustness_results.items():
    qualities = sorted(results.keys(), reverse=True)
    f1_scores = [results[q] for q in qualities]
    plt.plot(qualities, f1_scores, marker='o', 
            label=arch.upper())

plt.xlabel('JPEG Kwaliteit (100=geen, 10=max compressie)')
plt.ylabel('F1-score')
plt.title('Model Robuustheid onder JPEG-Compressie')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / 'figures' / 'robustness_jpeg.png')
plt.show()
