import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_model(arch: str, train_loader, val_loader,
                num_epochs=30, lr=1e-4):
    """
    Traint het model en slaat het beste checkpoint op.
    Beste model = hoogste validatie F1-score.
    """
    model = build_model(arch, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                             lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6)
    
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [],
               'val_f1': []}
    best_val_f1 = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for imgs, labels, _ in tqdm(train_loader,
                    desc=f"[{arch}] Epoch {epoch}/{num_epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)
        
        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)
        val_f1 = f1_score(all_labels, all_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 
                      OUTPUT_DIR / 'models' / f'{arch}_best.pt')
    
    return model, history
