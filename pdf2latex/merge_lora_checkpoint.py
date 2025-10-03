import argparse
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint with base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Base model name or path")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for merged model")

    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = f"{args.ckpt_path}/merged"

    print(f"Loading processor from {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=True)

    print(f"Loading base model from {args.base_model}...")
    base_model = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype="bfloat16")

    print(f"Loading LoRA adapter from {args.ckpt_path}...")
    model = PeftModel.from_pretrained(base_model, args.ckpt_path)

    print("Merging and unloading...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.out_dir}...")
    merged.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)

    print("Done!")

if __name__ == "__main__":
    main()
