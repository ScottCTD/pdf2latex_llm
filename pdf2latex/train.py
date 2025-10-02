import torch
from transformers import TrainingArguments, Trainer, AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

raw_dataset_file = "datasets/latex80m_en_10k.parquet"
train_dataset = load_dataset("parquet", data_files=raw_dataset_file, split="train[:90%]")
validation_dataset = load_dataset("parquet", data_files=raw_dataset_file, split="train[90%:]")

# this processor is responsible for processing both images and texts for the model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", use_fast=True)

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
    device_map="auto",
)

target_modules = [
    # text attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    # text MLP
    "gate_proj", "up_proj", "down_proj",
    # vision attention
    "qkv", "proj",
    # vision MLP
    "fc1", "fc2",
    # merger: target only the Linear leaves (avoid the container)
    "visual.merger.mlp.0",
    "visual.merger.mlp.2",
]


peft_config = LoraConfig(
    target_modules=target_modules,
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

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

def vlm_collator(examples):
    user_only_msgs, full_msgs = [], []
    for ex in examples:
        u, f = build_messages(ex["image"], ex["latex_formula"])
        user_only_msgs.append(u)
        full_msgs.append(f)
    
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
    for i, prompt_len in enumerate(prompt_lens):
        labels[i, :int(prompt_len)] = -100

    labels[attn_mask == 0] = -100
    
    return {
        **full_inputs,
        "labels": labels,
    }

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
    bf16=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=vlm_collator,
)

trainer.train()
