from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base_model = "Qwen/Qwen2-VL-2B-Instruct"
ckpt_path = "outputs/checkpoint-5231"

processor = AutoProcessor.from_pretrained(base_model, use_fast=True)
base_model = AutoModelForImageTextToText.from_pretrained(base_model, dtype="bfloat16")

model = PeftModel.from_pretrained(base_model, ckpt_path)
merged = model.merge_and_unload()

out_dir = "outputs/merged_checkpoint"
merged.save_pretrained(out_dir)
processor.save_pretrained(out_dir)
