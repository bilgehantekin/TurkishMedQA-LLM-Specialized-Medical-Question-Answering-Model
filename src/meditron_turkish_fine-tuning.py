import json
import os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login

# Hugging Face login (gerekirse)
login("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")

# Ortam değişkenleri
os.environ["TRITON_CACHE_DIR"] = "/home/data/triton_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/data/hf_cache"
os.environ["HF_HOME"] = "/home/data/hf_cache"

# Veri Yükleme
with open("/home/data/cleaned_output.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

dataset = Dataset.from_list([
    {"question": r["question"].strip(), "answer": r["answer"].strip()}
    for r in raw if r.get("question") and r.get("answer")
])

# Tokenizer ve Model
MODEL_ID = "malhajar/Mistral-7B-v0.2-meditron-turkish"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")

# PEFT Ayarları (LoRA)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# Split
split_1 = dataset.train_test_split(test_size=0.15, seed=42)
train_val = split_1["train"]
test_ds = split_1["test"]
split_2 = train_val.train_test_split(test_size=0.1765, seed=42)
train_ds, eval_ds = split_2["train"], split_2["test"]

# Tokenizasyon
def tok_fn(batch):
    texts = [f"Soru: {q}\nYanıt: {a}" for q, a in zip(batch["question"], batch["answer"])]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    enc["labels"] = enc["input_ids"].copy()
    return enc

train_ds = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(tok_fn, batched=True, remove_columns=eval_ds.column_names)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Eğitim Ayarları
args = TrainingArguments(
    output_dir="/home/data/mistral7b_meditron_peft_ckpts_1",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=8,
    learning_rate=5e-5,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained("/home/data/mistral7b_meditron_peft_ckpts_1")
