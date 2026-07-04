import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import gc

gc.collect()
torch.cuda.empty_cache()

# --- AYARLAR ---
MODEL_ID = "vngrs-ai/Kumru-2B"
DATASET_PATH = "processed_data/fikri_egitim_verisi.json"
OUTPUT_DIR = "./fikri_persona_model_2"

# 1. Veri Setini Yükleme
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 1b. messages formatını prompt/completion formatına çevir
def to_prompt_completion(example):
    messages = example["messages"]
    # ilk mesaj (user) -> prompt, ikinci mesaj (assistant) -> completion
    prompt = [m for m in messages if m["role"] != "assistant"]
    completion = [m for m in messages if m["role"] == "assistant"]
    return {"prompt": prompt, "completion": completion}

dataset = raw_dataset.map(to_prompt_completion, remove_columns=raw_dataset.column_names)

# İsteğe bağlı: train/eval split
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 2. T4 GPU Uyumlu 4-Bit Yapılandırması
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 3. Model ve Tokenizer Yükleme
print("Model yükleniyor...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,   # ekle — bnb_4bit_compute_dtype ile eşleşsin
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, clean_up_tokenization_spaces=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 4. LoRA Yapılandırması
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Eğitim Hiperparametreleri
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=20,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=5,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    eval_strategy="epoch",
    max_length=128,
    completion_only_loss=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none"
)
# 6. Eğiticiyi Başlatma (formatting_func YOK, data_collator YOK — TRL otomatik hallediyor)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)

# 7. Eğitimi Çalıştır
print("Eğitim başlıyor...")
trainer.train()

# 8. Kaydet
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_lora_weights")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora_weights")
print("Eğitim tamamlandı ve ağırlıklar kaydedildi.")