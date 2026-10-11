import os
import sys
import traceback

OUTPUT_PATH = '/content/output.wav'
TEXT_PATH = '/content/poem_text.txt'

try:
    import soundfile as sf
    from chatterbox.tts import ChatterboxTTS

    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError(f"{TEXT_PATH} not found")

    with open(TEXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        raise ValueError("poem_text.txt is empty")
    print(f"Text length: {len(text)} chars", flush=True)

    print("Loading Chatterbox model...", flush=True)
    model = ChatterboxTTS.from_pretrained(device='cuda')

    print("Generating audio...", flush=True)
    wav = model.generate(text, exaggeration=0.7, cfg_weight=0.3)

    sf.write(OUTPUT_PATH, wav.squeeze().cpu().numpy(), model.sr)
    if not os.path.exists(OUTPUT_PATH):
        raise RuntimeError("output.wav was not created")

    print(f"Saved: {os.path.getsize(OUTPUT_PATH)} bytes", flush=True)
    print("Files in /content:", os.listdir('/content'), flush=True)
    print("Audio generated successfully!", flush=True)
except Exception:
    traceback.print_exc()
    print("FATAL: generate_audio.py failed", flush=True)
    sys.exit(1)