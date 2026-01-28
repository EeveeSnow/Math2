import random

def split_samples(
    all_samples: list[str],
    val_ratio: float = 0.01,
    seed: int | None = None
):
    if seed is not None:
        random.seed(seed)

    samples = all_samples.copy()
    random.shuffle(samples)

    val_size = int(5 * 16)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    return train_samples, val_samples
