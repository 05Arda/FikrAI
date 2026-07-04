import torch
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from core import MODEL_ID, OUTPUT_DIR
from datetime import datetime, UTC

# --- AYARLAR ---
DATASET_PATH = "processed_data/fikri_egitim_verisi_2.json"
MAX_LENGTH = 512

# 1. Model ve Tokenizer Yükleme
print("Model yükleniyor...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_length=MAX_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# 2. LoRA Adaptörleri
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,         # r'nin 2 katı
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",   # Unsloth'un kendi optimize edilmiş checkpointing'i
    random_state=3407,
)

# 3. Veri Setini Yükleme ve Prompt/Completion Formatına Çevirme
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def to_prompt_completion(example):
    messages = example["messages"]
    prompt = [m for m in messages if m["role"] != "assistant"]
    completion = [m for m in messages if m["role"] == "assistant"]
    return {"prompt": prompt, "completion": completion}

dataset = raw_dataset.map(to_prompt_completion, remove_columns=raw_dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 4. Eğitim Hiperparametreleri
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=10,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    eval_strategy="epoch",
    max_length=MAX_LENGTH,
    completion_only_loss=True,   # bu template'te {% generation %} etiketi yok, assistant_only_loss çalışmaz
    report_to="none",
)

# 5. Eğiticiyi Başlatma
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    args=training_args,
)

# 6. Eğitimi Çalıştır
print("Eğitim başlıyor...")
trainer.train()

# 7. Kaydet
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_lora_weights")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora_weights")
print("Eğitim tamamlandı ve ağırlıklar kaydedildi.")