import numpy as np
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

# Bereken gewichten op basis van klasseverdeling
labels = [s[1] for s in train_ds.samples]
class_counts = np.bincount(labels)  # [n_real, n_fake]
class_weights = 1.0 / class_counts  # Inverse frequentie

# WeightedRandomSampler zorgt voor gelijke representatie
sample_weights = [class_weights[l] for l in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler,
                          num_workers=2, pin_memory=True)
