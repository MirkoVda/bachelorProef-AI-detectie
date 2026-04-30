from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np


def analyse_dataset(root: Path) -> dict:
    """Verzamelt basisstatistieken over de dataset:
    - Aantal afbeeldingen per klasse (real/fake)
    - Resolutie-variatie
    - Beschadigde/onleesbare bestanden
    """
    stats = {}
    img_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    for cls in ['real', 'fake']:
        folder = root / cls
        files = [f for f in folder.iterdir() 
                if f.suffix.lower() in img_ext]
        sizes, corrupted = [], 0
        
        for f in tqdm(files, desc=f"Analyseer {cls}"):
            try:
                with Image.open(f) as img:
                    sizes.append(img.size)  # (width, height)
            except Exception:
                corrupted += 1
                
        stats[cls] = {
            'count': len(files),
            'valid': len(files) - corrupted,
            'corrupted': corrupted,
            'resolution_stats': {
                'min_width': min(s[0] for s in sizes) if sizes else 0,
                'max_width': max(s[0] for s in sizes) if sizes else 0,
                'mean_width': np.mean([s[0] for s in sizes]) if sizes else 0,
            }
        }
    return stats
