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

# torch version ကိုက်ညီမှုမရှိတဲ့ torchvision ကို ဖယ်ရှားပါ (chatterbox က မသုံးပါ)
subprocess.run(['pip', 'uninstall', '-y', 'torchvision'], text=True)

# Verify: process အသစ်မှာ import စစ်ပါ
check = subprocess.run(
    [sys.executable, '-c',
     'import torch, torchaudio; print("torch", torch.__version__, "| torchaudio", torchaudio.__version__, "| CUDA", torch.cuda.is_available()); '
     'from transformers import LlamaModel; from chatterbox.tts import ChatterboxTTS; print("imports OK")'],
    capture_output=True, text=True)
print(check.stdout, flush=True)
print(check.stderr[-3000:], flush=True)
if check.returncode != 0:
    print("FATAL: import check failed after setup", flush=True)
    sys.exit(1)

print("Setup complete!", flush=True)