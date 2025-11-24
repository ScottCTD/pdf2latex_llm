# pdf2latex with VLMs

## Dev Environment Setup

### CUDA

Setup development environment:

```sh
conda create -n pdf2latex python=3.11 -y
conda activate pdf2latex
```

Install necessary packages:

```sh
pip install notebook tqdm wandb
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install transformers datasets accelerate
pip install peft
pip install flash-attn --no-build-isolation
```

### MPS

```sh
uv sync
```

Note that `flash-attn` is not available for MPS.

## Serving

```
ssh -L 8001:gpunode24:8000 cs.edu
```