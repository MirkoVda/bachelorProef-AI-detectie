from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_target_layer(model, arch: str):
    """Selecteert de juiste convolutionele laag per architectuur."""
    if arch == 'cnn':
        # EfficientNet: laatste convolutionele blok
        return [model.blocks[-1]]
    elif arch == 'vit':
        # ViT: normalisatielaag van laatst transformer block
        return [model.blocks[-1].norm1]
    elif arch == 'hybrid':
        # ConvNeXt: laatste stage
        return [model.stages[-1]]


def reshape_transform_vit(tensor, height=14, width=14):
    """Vouwt ViT output [Batch, 197, 768] terug naar [B, 768, 14, 14]."""
    result = tensor[:, 1:, :]  # Verwijder [CLS] token
    result = result.reshape(tensor.size(0), height, width, 
                           tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_gradcam(arch: str, image_paths: list,
                     true_labels: list, pred_labels: list,
                     n_images: int = 6):
    """
    Genereert Grad-CAM++ heatmaps voor willekeurige testafbeeldingen.
    Toont: origineel | Grad-CAM overlay | heatmap-alleen
    """
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(
        OUTPUT_DIR / 'models' / f'{arch}_best.pt'))
    model.eval()
    
    target_layers = get_target_layer(model, arch)
    
    # ViT vereist een speciale reshape-functie
    if arch == 'vit':
        cam = GradCAMPlusPlus(model=model, 
                             target_layers=target_layers,
                             reshape_transform=reshape_transform_vit)
    else:
        cam = GradCAMPlusPlus(model=model, 
                             target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(1)]  # Target "fake" klasse
    
    # Selecteer willekeurig n_images
    indices = random.sample(range(len(image_paths)), 
                           min(n_images, len(image_paths)))
    
    fig, axes = plt.subplots(len(indices), 3, 
                             figsize=(12, 4*len(indices)))
    
    for row, idx in enumerate(indices):
        path = image_paths[idx]
        
        # Laad afbeelding
        orig_img = Image.open(path).convert('RGB')
        orig_img = orig_img.resize((224, 224))
        rgb_img = np.array(orig_img) / 255.0  # float [0,1]
        
        # Bereken Grad-CAM
        tensor = val_transform(orig_img).unsqueeze(0).to(DEVICE)
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
        
        # Overlay heatmap op origineel
        heatmap_overlay = show_cam_on_image(
            rgb_img.astype(np.float32), grayscale_cam, 
            use_rgb=True)
        
        # Kolom 1: Origineel
        axes[row][0].imshow(orig_img)
        axes[row][0].set_title(f'Origineel\nWaar: {CLASS_NAMES[true_labels[idx]]}')
        axes[row][0].axis('off')
        
        # Kolom 2: Grad-CAM overlay
        axes[row][1].imshow(heatmap_overlay)
        pred_correct = '✔' if pred_labels[idx]==true_labels[idx] else '✘'
        axes[row][1].set_title(
            f'Grad-CAM Overlay\nVoorspeld: {CLASS_NAMES[pred_labels[idx]]} {pred_correct}')
        axes[row][1].axis('off')
        
        # Kolom 3: Pure heatmap
        im = axes[row][2].imshow(grayscale_cam, cmap='jet')
        axes[row][2].set_title('Activatiekaart\n(rood=fake-signaal)')
        axes[row][2].axis('off')
        plt.colorbar(im, ax=axes[row][2])
    
    plt.suptitle(f'Grad-CAM++ Interpretatie – {arch.upper()}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figures' / f'gradcam_{arch}.png')
    plt.show()
