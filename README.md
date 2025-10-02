# pdf2latex with VLMs

Setup development environment:
```sh
conda create -n pdf2latex python=3.11 -y
conda activate pdf2latex
```

Install necessary packages:
```sh
pip install notebook tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install transformers datasets
pip install peft
pip install flash-attn --no-build-isolation
```