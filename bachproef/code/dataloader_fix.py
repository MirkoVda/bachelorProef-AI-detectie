from torch.utils.data import DataLoader

# FOUT (causes OOM):
# train_loader = DataLoader(train_ds, batch_size=64, 
#                           num_workers=4)  # 4 sub-processen!

# CORRECT (stabiel):
train_loader = DataLoader(train_ds, batch_size=32, 
                          sampler=sampler,
                          num_workers=2,        # Minder processen
                          pin_memory=True)      # Sneller transfer

# Verhoogt snelheid ~20-30% zonder OOM-crashes
