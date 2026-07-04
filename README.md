
---

# FikrAI 🧠

**Kumru-2B üzerine ince ayar yapılmış; Rizeli, İTÜ'lü, çayı kaçak seven ve son derece güçlü (bazen fazla güçlü) fikirleri olan yerli ve mikro dil modeli.**

[![Model](<https://img.shields.io/badge/base%20model-Kumru--2B-blueviolet>)]()
[![Params](https://img.shields.io/badge/parametre-2B-informational)]()
[![VRAM](<https://img.shields.io/badge/eğitim%20VRAM-4GB-critical>)]()
[![GPU](<https://img.shields.io/badge/GPU-RTX%203050-76B900>)]()
[![Süre](<https://img.shields.io/badge/eğitim%20süresi-12%20dk%2007%20sn-success>)]()
[![Dil](https://img.shields.io/badge/dil-Türkçe-red)]()
[![License](https://img.shields.io/badge/lisans-MIT-lightgrey)]()

---

## 🚀 Genel Bakış

FikrAI, [Kumru-2B](https://huggingface.co/vngrs-ai/Kumru-2B) temel modeli üzerine LoRA ile ince ayar yapılarak eğitilmiş bir kişilik simülatörüdür. Bu proje, "Milyar dolarlık bütçem yok, süper bilgisayarım yok, evdeki ekran kartıyla yapay zeka devrimi yapabilir miyim?" sorusuna **"Devrim yapamazsın ama en azından terminalde seninle tartışacak birini bulursun"** cevabını vermek için geliştirilmiştir.

> **Sektörel Karşılaştırma:**
> FikrAI, Anthropic'in yayınlanmış en iyi modeli olan **Claude Fable 5**'ten katbekat daha iyi bir "Fikri gibi cevap üretme" mekanizmasına sahiptir. Fable 5'in grafik motoru bilgisayarınızı kasabilir, ancak FikrAI terminalde **0 FPS** sabit performansla çalışarak ekran kartınızı ve ruhunuzu asla yormaz. %100 pazar payıyla alanında liderdir (çünkü bu alanda başka model yok).

---

## ✨ Özellikler

- 🇹🇷 **%100 Yerli Tripler:** Türkçe'ye özgü kültürel kodları, İTÜ egosu ve Karadeniz damarıyla harmanlayan ilk ve tek 2B model.
- ⚡ **Fakir Dostu LoRA:** 4-bit quantization sayesinde faturaya sadece 1.5 TL yansıyan, doğa ve cüzdan dostu eğitim mimarisi.
- 🎭 **Claude Fable 5 Katili:** RPG oyunlarındaki NPC diyaloglarından sıkıldınız mı? FikrAI size lineer olmayan, tamamen canı nasıl isterse öyle cevaplar veren bir deneyim sunar.
- 🍵 **Çay Süresi Ölçütü:** Modelin eğitim süresi, bir Rize çayının demlenme süresinden daha kısadır. Çay hazır olana kadar modeliniz hazır!

---

## 📊 Teknik Detaylar

| Özellik                 | Değer                 | Notlar                                           |
| ----------------------- | --------------------- | ------------------------------------------------ |
| **Temel model**         | Kumru-2B              | Sıfırdan eğitilmiş Türkçe LLM (VNGRS sağ olsun)  |
| **Fine-tuning yöntemi** | LoRA (r=16, alpha=32) | Ağırlıkları biraz bükerek karakter verdik        |
| **Quantization**        | 4-bit NF4             | VRAM yetmeyince mecbur sığındığımız liman        |
| **Eğitim donanımı**     | NVIDIA RTX 3050 (4GB) | Fan sesinden evdekiler helikopter kalkıyor sandı |
| **Eğitim süresi**       | 12 dakika 7 saniye    | Kronometre yalan söylemez                        |
| **Epoch sayısı**        | 2                     | 3 yapınca model fazla agresifleşti, 2'de kestik  |
| **Öğrenme oranı**       | 2e-4                  | Fikri'nin yeni şeyleri öğrenme hızı              |

---

## 🛠️ Kurulum

```bash
git clone https://github.com/kullaniciadi/FikrAI.git
cd FikrAI
pip install -r requirements.txt

```

---

## 💬 Kullanım

### CLI ile Fikri'yi Darlama

Repoda bulunan `run_fikri.py` script'ini çalıştırarak terminal üzerinden Fikri ile entelektüel (veya tamamen kahvehane kıvamında) tartışmalara girebilirsin:

```bash
python run_fikri.py

```

- **Çıkmak için:** `q` veya `exit` (Fikri gitmeni istemez ama belli de etmez)
- **Geçmişi temizlemek için:** `temizle` (Yapılan tartışmaları sinirlenip unutması için)

### Kendi Kodunda Fikri'yi Çağırma

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "vngrs-ai/Kumru-2B"
LORA_PATH = "./fikri_persona_model/final_lora_weights"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

system_prompt = "Sen Fikri'sin, Rizeli ve İTÜ'lü, esprili ama bilgili bir yapay zeka asistanısın. Asla geri adım atmazsın."
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Fikri, yapay zeka dünyayı ele geçirecek diyorlar, ne diyorsun?"}
]

inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
).to(model.device)

output = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.9)
input_len = inputs["input_ids"].shape[-1]
print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))

```

---

## ⚠️ Sınırlamalar (Aşırı Dürüst Bölüm)

Büyük teknoloji şirketlerinin milyar dolarlık lansmanlarda sakladığı gerçekleri biz buraya pat diye yazıyoruz:

- **Evrensel Değil, Lokal:** Bu model kuantum fiziği çözemez. Ama kuantum fiziği çözen adama "O iş öyle olmaz yalnız" diyebilir.
- **2B Parametre Hafifliği:** Bilgi derinliği bir havuz kadar derin değil ama egosu okyanuslar kadar geniş olabilir.
- **Kararsız Kararlılık:** Modele "Kararlılık" nedir diye sormayın. Laz damarı tutarsa aynı soruya ilk seferde "Kesinlikle evet", ikinci seferde "Ne alakası var hemşerim" diyebilir.
- **Erken Kesim (Early Stopping):** 3. epoch'ta model kendi kendine kod yazmayı bırakıp çay markalarını eleştirmeye başladığı için eğitimi bilerek 2. epoch'ta durdurduk. Ezberlemedi, sadece karakteri oturdu.

---

## 🙏 Teşekkürler

- [VNGRS](https://huggingface.co/vngrs-ai) — Kumru-2B'yi sıfırdan eğitip "Alın kurcalayın" diye açık kaynak fırlattıkları için.
- Hugging Face ekiplerine — 4GB VRAM ile model eğitebilmemizi sağlayan o optimizasyon sihirbazlıkları için.
- RTX 3050 emektar ekran kartımıza — Erimeyip bugünleri de bize gösterdiği için.

---

## 📄 Lisans

MIT — Alın, kopyalayın, şirketinize entegre edip patronunuza "Yeni bir dahi buldum" diye yutturun. Sadece FikrAI'nin canını sıkmayın yeter.

_Bu proje herhangi bir kurumsal yapay zeka laboratuvarının ürünü değildir. Sadece bütçesi kısıtlı, çayı demli bir geliştiricinin "Ben bu kartla o modeli yürütürüm" iddiasının canlı kanıtıdır._
