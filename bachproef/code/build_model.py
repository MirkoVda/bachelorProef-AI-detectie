import timm
import torch


def build_model(arch: str, num_classes: int = 2, pretrained: bool = True):
    """
    Laadt de gekozen architectuur met ImageNet-voortraining.
    Vervangt de classifier-head voor binaire classificatie.
    """
    if arch == 'cnn':
        # EfficientNet-B3: 12M parameters, sterk in lokale artefacten
        model = timm.create_model('efficientnet_b3', 
                                   pretrained=pretrained,
                                   num_classes=num_classes)
        print(f"CNN (EfficientNet-B3): " + 
              f"{sum(p.numel() for p in model.parameters()):,} params")
        
    elif arch == 'vit':
        # Vision Transformer: 86M parameters, sterk in globale context
        model = timm.create_model('vit_base_patch16_224',
                                   pretrained=pretrained, 
                                   num_classes=num_classes)
        print(f"ViT-B/16: " + 
              f"{sum(p.numel() for p in model.parameters()):,} params")
        
    elif arch == 'hybrid':
        # ConvNeXt-Small: hybride CNN + Transformer-achtig
        model = timm.create_model('convnext_small',
                                   pretrained=pretrained,
                                   num_classes=num_classes)
        print(f"Hybrid (ConvNeXt-Small): " + 
              f"{sum(p.numel() for p in model.parameters()):,} params")
    else:
        raise ValueError(f"Onbekende architectuur: {arch}")
        
    return model.to(DEVICE)
