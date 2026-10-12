import os
import re
import sys
import subprocess
import traceback

import numpy as np

OUTPUT_PATH = '/content/output.wav'
RAW_PATH = '/content/output_raw.wav'
TEXT_PATH = '/content/poem_text.txt'
VOICE_PATH = '/content/voice.wav'

# ညှိစရာတွေ (တည်ငြိမ်ပြီး စိတ်ဝင်စားဖွယ်)
SPEED = 0.88          # 1.0 = မူလ၊ 0.8 အောက် မဆင်းပါနဲ့
EXAGGERATION = 0.4    # နိမ့်လေ တည်ငြိမ်လေ၊ မြင့်လေ မြန်လေ
CFG_WEIGHT = 0.3      # official အကြံပြုတန်ဖိုး
PAUSE_LINE = 0.8      # ကော်မာ/စာကြောင်းဆုံး ရပ်ချိန် (စက္ကန့်)
PAUSE_END = 1.3       # . ! ? ။ ပြီးရင်
PAUSE_STANZA = 2.2    # | ပြီးရင် (ပိုဒ်ကြား)
LEAD = 0.5            # အစ တိတ်ဆိတ်ချိန်
TAIL = 1.2            # အဆုံး တိတ်ဆိတ်ချိန်
SEED = 42

print("SCRIPT VERSION: v3-voice", flush=True)


def split_chunks(stanza):
    lines = [l.strip() for l in stanza.splitlines() if l.strip()]
    if len(lines) == 1:
        lines = [c.strip() for c in re.split(r'(?<=[,;:.!?။])\s+', lines[0]) if c.strip()]
    return lines


try:
    import torch
    import soundfile as sf
    from chatterbox.tts import ChatterboxTTS

    if not os.path.exists(TEXT_PATH):
        raise FileNotFoundError(f"{TEXT_PATH} not found")
    with open(TEXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        raise ValueError("poem_text.txt is empty")

    stanzas = [s.strip() for s in text.split('|') if s.strip()]
    print(f"Stanzas: {len(stanzas)}", flush=True)

    print("Loading Chatterbox model...", flush=True)
    model = ChatterboxTTS.from_pretrained(device='cuda')
    sr = model.sr
    torch.manual_seed(SEED)

    if os.path.exists(VOICE_PATH):
        model.prepare_conditionals(VOICE_PATH, exaggeration=EXAGGERATION)
        print("Using reference voice", flush=True)
    else:
        print("Using default voice", flush=True)

    parts = [np.zeros(int(sr * LEAD), dtype=np.float32)]
    total = sum(len(split_chunks(s)) for s in stanzas)
    n = 0
    for si, stanza in enumerate(stanzas):
        chunks = split_chunks(stanza)
        for ci, chunk in enumerate(chunks):
            n += 1
            print(f"[{n}/{total}] {chunk}", flush=True)
            wav = model.generate(chunk, exaggeration=EXAGGERATION, cfg_weight=CFG_WEIGHT)
            parts.append(wav.squeeze().cpu().numpy())
            last_in_stanza = (ci == len(chunks) - 1)
            last_stanza = (si == len(stanzas) - 1)
            if last_in_stanza and not last_stanza:
                pause = PAUSE_STANZA
            elif chunk[-1] in '.!?။':
                pause = PAUSE_END
            else:
                pause = PAUSE_LINE
            parts.append(np.zeros(int(sr * pause), dtype=np.float32))

    parts.append(np.zeros(int(sr * TAIL), dtype=np.float32))
    audio = np.concatenate(parts)
    sf.write(RAW_PATH, audio, sr)

    r = subprocess.run(
        ['ffmpeg', '-y', '-i', RAW_PATH, '-filter:a', f'atempo={SPEED}', OUTPUT_PATH],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
        raise RuntimeError("ffmpeg atempo failed")

    if not os.path.exists(OUTPUT_PATH):
        raise RuntimeError("output.wav was not created")

    print(f"Saved: {os.path.getsize(OUTPUT_PATH)} bytes", flush=True)
    print("Audio generated successfully!", flush=True)
except Exception:
    traceback.print_exc()
    print("FATAL: generate_audio.py failed", flush=True)
    sys.exit(1)