import os
import sys
import numpy as np
import soundfile as sf
from chatterbox.tts import ChatterboxTTS

OUTPUT_PATH = '/content/output.wav'
TEXT_PATH = '/content/poem_text.txt'

# 1. စာသားဖိုင်ကို စစ်ဆေးပါ
if not os.path.exists(TEXT_PATH):
    print(f"FATAL: {TEXT_PATH} not found", flush=True)
    sys.exit(1)

with open(TEXT_PATH, 'r', encoding='utf-8') as f:
    text = f.read().strip()

if not text:
    print("FATAL: poem_text.txt is empty", flush=True)
    sys.exit(1)

print(f"Text length: {len(text)} chars", flush=True)

# 2. မော်ဒယ်ကို Load လုပ်ပါ
print("Loading Chatterbox model...", flush=True)
model = ChatterboxTTS.from_pretrained(device='cuda')

# 3. အသံ ဖန်တီးပါ
print("Generating audio...", flush=True)
wav = model.generate(text, exaggeration=0.7, cfg_weight=0.3)

# 4. torchaudio အစား soundfile နဲ့ သိမ်းပါ (torchcodec error ကို ရှောင်ဖို့)
audio = wav.squeeze().cpu().numpy()
sf.write(OUTPUT_PATH, audio, model.sr)
print(f"Saved to {OUTPUT_PATH}", flush=True)

# 5. ဖိုင် တကယ်ရှိမရှိ စစ်ဆေးပါ
if not os.path.exists(OUTPUT_PATH):
    print(f"FATAL: {OUTPUT_PATH} was not created", flush=True)
    sys.exit(1)

size = os.path.getsize(OUTPUT_PATH)
print(f"File size: {size} bytes", flush=True)

# 6. /content ထဲမှာ ဘာဖိုင်တွေ ရှိလဲ ပြပါ (debug အတွက်)
print("Files in /content:", os.listdir('/content'), flush=True)

print("Audio generated successfully!", flush=True)