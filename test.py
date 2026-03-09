from datasets import load_dataset
from torch.utils.data import DataLoader


from wraper import MathWritingDataset, Vocabulary, encode_batch


def start_test(model, DEVICE, vocab, ):
    ds = load_dataset("deepcopy/MathWriting-Human")
    test_dataset = MathWritingDataset(ds["test"], image_size=(384, 384))
    
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=12, pin_memory=True)

    for batch in test_loader:
        images = batch["image"].to(DEVICE)
        captions = encode_batch(batch["latex"], vocab).to(DEVICE)
        inp = captions[:, :-1]
        with autocast('cuda'): # Use FP16
            logits, _ = model(images, inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
            loss = loss / ACCUMULATION_STEPS
        val_losses.append(loss.item())
