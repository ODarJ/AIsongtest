# colab_job/generate_audio.py
import os
import sys
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

text = os.environ.get('POEM_TEXT', '')

if not text:
    print("FATAL: POEM_TEXT environment variable is empty", flush=True)
    sys.exit(1)

print("Loading Chatterbox model...", flush=True)
model = ChatterboxTTS.from_pretrained(device='cuda')

print("Generating audio...", flush=True)
wav = model.generate(text, exaggeration=0.7, cfg_weight=0.3)

ta.save('output.wav', wav, model.sr)
print("Audio generated successfully!", flush=True)