import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gc

gc.collect()
torch.cuda.empty_cache()

from core import MODEL_ID, OUTPUT_DIR

LORA_PATH = OUTPUT_DIR / "final_lora_weights"


def model_yukle():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Model yükleniyor...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    print("Model hazır!\n")
    return model, tokenizer


def sohbet(model, tokenizer, message, history, max_new_tokens=200, temperature=0.7):
    history.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        history,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    input_len = inputs["input_ids"].shape[-1]
    yanit = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    history.append({"role": "assistant", "content": yanit})
    return yanit

SYSTEM_PROMPT = "Senin adın Fikri. Sen, Rizeli ve İTÜ'lü, çoook uzun süreler çay demleyen, evrak işleriyle uğraşmayı seven bir yapay zeka asistanısın. Uzun ve samimi cevaplar ver."


def main():
    model, tokenizer = model_yukle()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]  # buraya ekle

    print("=" * 50)
    print("FikrAI ile sohbet başladı")
    print("Çıkmak için: 'q' veya 'exit'")
    print("Geçmişi temizlemek için: 'temizle'")
    print("=" * 50 + "\n")

    while True:
        try:
            message = input("Sen: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGörüşürüz!")
            break

        if not message:
            continue

        if message.lower() in ("q", "exit", "quit"):
            print("Görüşürüz!")
            break

        if message.lower() == "temizle":
            history = []
            print("(Geçmiş temizlendi)\n")
            continue

        response = sohbet(model, tokenizer, message, history)
        print(f"Fikri: {response}\n")


if __name__ == "__main__":
    main()