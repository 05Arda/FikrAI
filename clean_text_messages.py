import re
import json
import os

# Ayarlar
TARGET_USER = "Fikri İTÜ Cevher"
INPUT_FILES = ["raw_sources/WhatsApp-Chat mit Fikri İTÜ Cevher.txt", "raw_sources/WhatsApp-Chat mit aleyküm selam lan berkay or.txt", "raw_sources/WhatsApp-Chat mit tenischain.txt", "raw_sources/WhatsApp-Chat mit Yozgat Ya.txt"]
OUTPUT_FILE = "processed_data/fikri_egitim_verisi.json"

# Hariç tutulacak metinler
ignored_texts = [
    "<Diese Nachricht wurde bearbeitet.>", 
    "<Medien ausgeschlossen>", 
    "Nachrichten und Anrufe sind", 
    "http://", 
    "https://",
    "Du hast diese Nachricht gelöscht.",
    "Diese Nachricht wurde gelöscht."
]

# Grup karmaşasını önlemek için Fikri'den önce bakılacak maksimum mesaj sayısı
MAX_HISTORY = 5

# WhatsApp satır yakalama regex'i
pattern = re.compile(r'^(\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}) - ([^:]+): (.*)$')

parsed_messages = []
# 1. Dosyaları Oku ve Filtrele
for file_path in INPUT_FILES:
    if not os.path.exists(file_path):
        continue
        
    with open(file_path, "r", encoding="utf-8") as f:
        current_message = None
        for line in f:
            line = line.strip()
            
            # Kara listedeki veya silinmiş mesaj satırlarını atla
            if any(ignored in line for ignored in ignored_texts):
                current_message = None
                continue
                
            match = pattern.match(line)
            if match:
                if current_message:
                    parsed_messages.append(current_message)
                date_time, sender, message = match.groups()
                current_message = {"sender": sender, "text": message}
            else:
                # Alt satıra sarkan mesajları bağla
                if current_message:
                    current_message["text"] += " " + line
                    
        if current_message:
            parsed_messages.append(current_message)

# 2. Aynı Kişinin Peş Peşe Attığı Mesajları Birleştir
grouped_messages = []
for msg in parsed_messages:
    if grouped_messages and grouped_messages[-1]["sender"] == msg["sender"]:
        grouped_messages[-1]["text"] += " " + msg["text"]
    else:
        grouped_messages.append(msg)

# 3. Dataset Oluşturma
dataset = []
others_buffer = []

for msg in grouped_messages:
    if msg["sender"] == TARGET_USER:
        if others_buffer:
            relevant_context = others_buffer[-MAX_HISTORY:]
            
            # Tire işareti kaldırıldı, mesajlar sadece alt satırla ayrılır
            user_content = "\n".join([m['text'] for m in relevant_context])
            
            dataset.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": msg["text"]}
                ]
            })
            others_buffer = []
    else:
        others_buffer.append(msg)

# 4. JSON Olarak Kaydet
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"İşlem tamamlandı. {len(dataset)} adet temizlenmiş senaryo '{OUTPUT_FILE}' dosyasına kaydedildi.")