**init**

```bash
    python3 -m venv venv
    source venv/bin/activate   
```

if using cuda 12.1

```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

if using cpu

```bash
    pip install torch torchvision
```

```bash
    pip install timm pathlib tqdm gradio datasets pandas nltk plotly
```

only for mambda model
requared

Linux
NVIDIA GPU Amper+
PyTorch 2.10
CUDA 13.0

```bash
    pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```


**Training**

```bash
    python training.py
```

**Interface**

```bash
    python app.py
```
