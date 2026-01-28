**init**
```bash
    python3 -m venv venv
    source venv/bin/activate   
```

if using cuda 12.1

```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

if using cpu

```bash
    pip install torch torchvision torchaudio
```

```bash
    pip install timm pathlib split_utils tqdm
```

**Training**
```bash
    python training.py
```