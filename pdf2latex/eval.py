from datasets import load_dataset
from io import BytesIO
import base64
from tqdm import tqdm

# columns: image, latex_formula, category
eval_dataset = load_dataset("OleehyO/latex-formulas-80M", "benchmark_ordinary", split="train")

eval_dataset = eval_dataset.select(range(100))

def make_prompt(row):
    return {
        "role": "user",
        "content": [
            {"type": "image", "url": row["image"]},
            {"type": "text", "text": "Transcribe the given image to LaTeX."}
        ]
    }

prompts = [make_prompt(row) for row in eval_dataset]

# load model
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

for prompt in tqdm(prompts):
    inputs = processor.apply_chat_template(
        [prompt],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
