from torch.utils.data import DataLoader
from data import CROHMEDataset
from split_utils import split_samples
from pathlib import Path

def build_dataloaders(
    root_dir,
    image_size,
    batch_size,
    epoch,
    val_ratio=0.1,
    split_every=10,
    num_workers=4
):
    # фиксированный список всех файлов
    all_samples = sorted([
        p.stem.replace(".inkml", "")
        for p in Path(root_dir).glob("*.inkml.png")
    ])

    # новый сплит каждые N эпох
    if epoch % split_every == 0:
        seed = epoch  # controlled randomness
    else:
        seed = None

    train_samples, val_samples = split_samples(
        all_samples,
        val_ratio=val_ratio,
        seed=seed
    )

    train_ds = CROHMEDataset(
        root_dir,
        image_size=image_size,
        samples=train_samples
    )

    val_ds = CROHMEDataset(
        root_dir,
        image_size=image_size,
        samples=val_samples
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
