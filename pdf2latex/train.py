import torch
from collections import defaultdict
from typing import Dict
import wandb

import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoProcessor,
    AutoModelForImageTextToText,
    EvalPrediction,
    GenerationConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

import metrics
from pdf2latex_trainer import Pdf2LatexTrainer
import utils
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# (Optional on PyTorch 2.x)
torch.set_float32_matmul_precision("high")

wandb.init(project="pdf2latex_llm")

raw_dataset_file = "datasets/latex80m_en_1m.parquet"
validation_size = 640
train_dataset = load_dataset(
    "parquet", data_files=raw_dataset_file, split=f"train[:-{validation_size}]"
)
validation_dataset = load_dataset(
    "parquet", data_files=raw_dataset_file, split=f"train[-{validation_size}:]"
)

# this processor is responsible for processing both images and texts for the model
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    use_fast=True,
    # min_pixels=64 * 28 * 28,  # about 64 visual tokens
    max_pixels=896 * 28 * 28,
)

tokenizer = processor.tokenizer


def _image_size(image):
    """Return (width, height) for PIL, numpy, or torch image-like objects."""
    if isinstance(image, Image.Image):
        return image.size
    try:
        import numpy as _np
        if isinstance(image, _np.ndarray):
            if image.ndim == 3:
                h, w = image.shape[:2]
            elif image.ndim == 2:
                h, w = image.shape
            else:
                return None
            return (w, h)
    except Exception:
        pass
    try:
        import torch as _torch
        if _torch.is_tensor(image):
            if image.ndim == 3:
                # channels-first or channels-last
                if image.shape[0] in (1, 3):
                    _, h, w = image.shape
                else:
                    h, w, _ = image.shape
            elif image.ndim == 2:
                h, w = image.shape
            else:
                return None
            return (w, h)
    except Exception:
        pass
    return None


def _is_aspect_ok(image, max_abs_ratio: float = 200.0) -> bool:
    size = _image_size(image)
    if size is None:
        return True
    w, h = size
    if w == 0 or h == 0:
        return True
    return max(w, h) / max(1, min(w, h)) <= max_abs_ratio


def build_messages(image, latex_formula):
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Transcribe the given image to LaTeX."},
        ],
    }
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": latex_formula}],
    }
    return [user], [user, assistant]


def vlm_collator(examples):
    # Skip items with extreme aspect ratio; if all are skipped, fall back to original batch
    filtered = [ex for ex in examples if _is_aspect_ok(ex["image"], 200.0)]
    effective = filtered if len(filtered) > 0 else examples

    user_only_msgs, full_msgs = [], []
    for ex in effective:
        u, f = build_messages(ex["image"], ex["latex_formula"])
        user_only_msgs.append(u)
        full_msgs.append(f)

    # this will left-pad messages
    full_inputs = processor.apply_chat_template(
        full_msgs,
        tokenize=True,
        padding=True,
        return_dict=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    prompt_only = processor.apply_chat_template(
        user_only_msgs,
        tokenize=True,
        padding=True,
        return_dict=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    input_ids = full_inputs["input_ids"]
    attn_mask = full_inputs["attention_mask"]

    labels = input_ids.clone()
    prompt_lens = prompt_only["attention_mask"].sum(dim=1).tolist()
    padding_lens = utils.index(attn_mask, 1).tolist()
    for i, (prompt_len, padding_len) in enumerate(zip(prompt_lens, padding_lens)):
        labels[i, : int(prompt_len + padding_len)] = -100

    return {
        **full_inputs,
        "labels": labels,
        "prompt_input_ids": prompt_only["input_ids"],
        "prompt_attention_mask": prompt_only["attention_mask"],
    }


model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
    device_map="auto",
)

target_modules = [
    # text attention
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # text MLP
    "gate_proj",
    "up_proj",
    "down_proj",
    # vision attention
    "qkv",
    "proj",
    # vision MLP
    "fc1",
    "fc2",
    # merger: target only the Linear leaves (avoid the container)
    "visual.merger.mlp.0",
    "visual.merger.mlp.2",
]


peft_config = LoraConfig(
    target_modules=target_modules,
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    metrics_dict = defaultdict(list)
    i = 0
    for pred, label in zip(preds, labels):
        metrics_dict["exact_match"].append(metrics.metric_exact_match(pred, label))
        metrics_dict["normalized_exact_match"].append(
            metrics.metric_normalized_exact_match(pred, label)
        )
        metrics_dict["normalized_edit_similarity"].append(
            metrics.metric_normalized_edit_similarity(label, pred)
        )

        if i % 64 == 0:
            print("=" * 100)
            print(f"Pred: {pred}")
            print(f"Label: {label}")
            print(
                f"EM={metrics_dict['exact_match'][-1]}  "
                f"NEM={metrics_dict['normalized_exact_match'][-1]}  "
                f"NES={metrics_dict['normalized_edit_similarity'][-1]}"
            )
        i += 1
    return {
        "exact_match": float(np.mean(metrics_dict["exact_match"])),
        "normalized_exact_match": float(np.mean(metrics_dict["normalized_exact_match"])),
        "normalized_edit_similarity": float(np.mean(
            metrics_dict["normalized_edit_similarity"]
        )),
    }


eval_generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=False,
)

run_name = "1"
training_args = Seq2SeqTrainingArguments(
    output_dir=f"outputs/{run_name}",
    eval_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,
    eval_steps=100,
    eval_on_start=True,
    gradient_checkpointing=True,
    bf16=True,
    remove_unused_columns=False,
    predict_with_generate=True,
    generation_config=eval_generation_config,
)

trainer = Pdf2LatexTrainer(
    model=model,
    args=training_args,
    processing_class=processor,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=vlm_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
