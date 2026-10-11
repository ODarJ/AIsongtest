import os
import sys
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# ဖိုင်ကနေ စာသားကို ဖတ်ပါ
text_path = '/content/poem_text.txt'
if not os.path.exists(text_path):
    print(f"FATAL: {text_path} not found", flush=True)
    sys.exit(1)

with open(text_path, 'r', encoding='utf-8') as f:
    text = f.read().strip()

if not text:
    print("FATAL: poem_text.txt is empty", flush=True)
    sys.exit(1)

print(f"Text length: {len(text)} chars", flush=True)

print("Loading Chatterbox model...", flush=True)
model = ChatterboxTTS.from_pretrained(device='cuda')

print("Generating audio...", flush=True)
wav = model.generate(text, exaggeration=0.7, cfg_weight=0.3)

ta.save('output.wav', wav, model.sr)
print("Audio generated successfully!", flush=True)