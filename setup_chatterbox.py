# colab_job/setup_chatterbox.py
import subprocess
import sys

def run(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"FATAL: command failed: {' '.join(cmd)}", flush=True)
        sys.exit(1)

run(['pip', 'install', 'chatterbox-tts', 'soundfile'])

import torch
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
print("Setup complete!", flush=True)