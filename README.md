
# FikrAI 🧠

**Kumru-2B üzerine ince ayar yapılmış, Türkçe konuşan, güçlü fikirleri olan bir dil modeli.**

[![Model](<https://img.shields.io/badge/base%20model-Kumru--2B-blueviolet>)]()
[![Params](https://img.shields.io/badge/parametre-2B-informational)]()
[![VRAM](<https://img.shields.io/badge/eğitim%20VRAM-4GB-critical>)]()
[![GPU](<https://img.shields.io/badge/GPU-RTX%203050-76B900>)]()
[![Süre](<https://img.shields.io/badge/eğitim%20süresi-12%20dk%2007%20sn-success>)]()
[![Dil](https://img.shields.io/badge/dil-Türkçe-red)]()
[![License](https://img.shields.io/badge/lisans-MIT-lightgrey)]()

---

## 🚀 Genel Bakış

FikrAI, [Kumru-2B](https://huggingface.co/vngrs-ai/Kumru-2B) temel modeli üzerine LoRA ile ince ayar yapılarak eğitilmiş bir kişilik modelidir. Proje, sınırlı hesaplama kaynaklarıyla (evet, **4GB VRAM** ile) anlamlı bir persona eğitiminin mümkün olduğunu göstermeyi hedefler.

> **Not:** "State of the art" tabirini burada gururla *kullanmıyoruz.* Ama gururla kullanmayı çok isterdik.

## ✨ Özellikler

- 🇹🇷 Türkçe'ye özgü, sıfırdan eğitilmiş bir temel model üzerine kurulu
- ⚡ Verimli LoRA fine-tuning (r=16, 4-bit quantization)
- 💾 4GB VRAM'de eğitilebilir — bütçe dostu, elektrik faturası dostu
- 🎭 Belirgin bir kişilik/persona ile yanıt verir
- 🧾 Şeffaf eğitim süreci (aşağıda tüm detaylar var, gizli bir "secret sauce" yok)

## 📊 Teknik Detaylar

| Özellik             | Değer                                         |
| -------------------- | ---------------------------------------------- |
| Temel model          | Kumru-2B (sıfırdan eğitilmiş Türkçe LLM) |
| Fine-tuning yöntemi | LoRA (r=16, alpha=32)                          |
| Quantization         | 4-bit NF4                                      |
| Eğitim donanımı   | NVIDIA RTX 3050 (4GB VRAM)                     |
| Eğitim süresi      | 12 dakika 7 saniye (çay demlenmeden bitti)   |
| Epoch sayısı       | 2                                              |
| Optimizer            | paged_adamw_8bit                               |
| Öğrenme oranı     | 2e-4                                           |

*Tüm sayılar gerçek — abartı yok, sadece minimalist bir kurulumla ne yapılabileceğinin kanıtı. Süper bilgisayar yok, sadece bir oyuncu ekran kartı ve azim.*

## 🛠️ Kurulum

```bash
git clone https://github.com/kullaniciadi/FikrAI.git
cd FikrAI
pip install -r requirements.txt
```

## 💬 Kullanım

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("vngrs-ai/Kumru-2B")
model = PeftModel.from_pretrained(base_model, "./fikri_persona_model/final_lora_weights")
tokenizer = AutoTokenizer.from_pretrained("./fikri_persona_model/final_lora_weights")

messages = [{"role": "user", "content": "Naber?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
output = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## ⚠️ Sınırlamalar (Dürüst Bölüm)

Reklam broşürlerinin genelde atladığı ama bizim atlamadığımız kısım:

- Küçük bir veri setiyle eğitildi — evrensel bir asistan değil, belirli bir persona/ton hedefleniyor
- 2B parametre, büyük modellerin genel bilgi derinliğine sahip değil
- Bazı konularda tutarlılık, daha büyük modellere göre daha düşük olabilir
- "Kararlılık" kelimesini FikrAI'ye sormayın, kendi de emin değil
- Eğitim 2 epoch'ta durduruldu — eval loss düzenli düşüş gösterdi, ezberlemeden kaçınmak için daha erken kesme tercih edildi (3 epoch'ta eval loss'un yükseldiğini gördük, o modeli kullanmıyoruz)

## 🙏 Teşekkürler

- [VNGRS](https://huggingface.co/vngrs-ai) — Kumru-2B'yi sıfırdan eğitip açık kaynak yaptıkları için
- Hugging Face `transformers`, `peft`, `trl` ve `bitsandbytes` ekiplerine
- 4GB VRAM'in dayanıklılığına

## 📄 Lisans

MIT — dilediğiniz gibi kullanın, sadece FikrAI'yi kötü bir gün geçirmiş gibi göstermeyin.

---

<sub>Bu proje bir yapay zeka firmasının resmi ürünü değildir. Sadece biri, sınırlı bir GPU ile bir şeyler denedi ve işe yaradı.</sub>
