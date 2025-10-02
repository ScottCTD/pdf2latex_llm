import torch
from transformers import TrainingArguments, Trainer, AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset

raw_datasets = load_dataset("datasets/latex80m_en_10k")

raw_datasets["train"] = load_dataset("datasets/latex80m_en_10k", split="train[:0.9%]")
raw_datasets["validation"] = load_dataset("datasets/latex80m_en_10k", split="train[0.9%:]")

# this processor is responsible for processing both images and texts for the model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", use_fast=True)

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

def build_messages(image, latex_formula):
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Transcribe the given image to LaTeX."}
        ]
    }
    assistant = {
        "role": "assistant",
        "content": [{"type": "text", "text": latex_formula}]
    }
    return [user], [user, assistant]

def pre_process(example):
    image = example["image"]
    answer = example["latex_formula"]
    
    user_only_msgs, full_msgs = build_messages(image, answer)
    
    prompt_text = processor.apply_chat_template(
        user_only_msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_text = processor.apply_chat_template(
        full_msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # calculate the number of tokens belong to the prompt
    prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    
    enc = processor(
        text=full_text,
        images=image,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    
    input_ids = enc["input_ids"][0]
    labels = input_ids.clone()
    
    ...

training_args = TrainingArguments(
    output_dir="outputs",
    eval_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=10,
    save_strategy="no",
    save_steps=1000,
    eval_steps=100,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processor=processor,
)

trainer.train()
