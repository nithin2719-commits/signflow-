import sys
import shutil
import subprocess
from pathlib import Path

# Paths
BASE = Path(r"c:\Users\Asus\project\New Msasl")
# Use combined data (84k+ existing + 4k msasl extracted)
DATA_COMBINED = BASE / "landmark_data_combined"
DATA_FALLBACK = BASE / "landmark_data_random_split"
DATA = DATA_COMBINED if (DATA_COMBINED / "train").exists() else DATA_FALLBACK
print(f"[INFO] Using data from: {DATA}")
TRAIN_SCRIPT = BASE / "train_landmark_transformer.py"
OUTPUT_DIR = BASE / "outputs" / "landmark_transformer_common_v1"
PRETRAINED = BASE / "outputs" / "landmark_transformer_common_v1" / "best_model.pth"

# Priority/common words (from your earlier prompt)
PRIORITY_WORDS = {
    'hi','hello','bye','good_morning','how_are_you','how','what','where','when',
    'who','why','yes','no','ok','fine','please','sorry','thankyou','my','name',
    'i','you','we','love','i_love_you','help','want','need','not_understand',
    'deaf','happy','sad','tired','hungry','mother','father','family','friend',
    'today','tomorrow','now','later','eat','drink','go','come','stop','finish',
    'understand','think','nice_to_meet_you','good','morning','night','welcome',
    'excited','angry','afraid','nervous','proud','lonely','confused','bored',
    'brother','sister','baby','child','children','husband','wife','uncle','aunt',
    'grandfather','grandmother','cousin','son','daughter','boyfriend','girlfriend',
    'man','woman','girl','boy','person','people','home','school','hospital',
    'water','food','dog','cat','phone','computer','book','money','music',
    'one','two','three','four','five','six','seven','eight','nine','ten',
    'red','blue','green','yellow','orange','black','white','brown','purple','pink',
    'big','small','hot','cold','old','new','fast','slow','beautiful','cute','right',
    'wrong','same','here','there','up','down','sick','work','learn','make','give',
    'take','talk','ask','tell','listen','read','write','play','wait','meet','buy',
    'see','feel','have','do','remember','forget','find','teach','use','try'
}

# Filter classes by sample count
train_dir = DATA / 'train'
val_dir = DATA / 'val'

def is_num(n):
    return n.replace('_','').replace('-','').isdigit()

words = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not is_num(d.name)])
total_train = 0
total_val = 0
kept = []
skipped = []

for w in words:
    n = len(list((train_dir / w).glob('*.npy')))
    is_p = w.lower() in PRIORITY_WORDS
    # Keep as many words as possible - low threshold
    thresh = 5 if is_p else 15
    if n >= thresh:
        kept.append((w, n, is_p))
        total_train += n
        vd = val_dir / w
        if vd.exists():
            total_val += len(list(vd.glob('*.npy')))
    else:
        skipped.append(w)

print()
print('=' * 60)
print(f'WORDS TO TRAIN: {len(kept)}')
print(f'SKIPPED (too few samples): {len(skipped)}')
print(f'TOTAL TRAIN SAMPLES: {total_train}')
print(f'TOTAL VAL   SAMPLES: {total_val}')
print('=' * 60)
print()
print('--- ALL TRAINING WORDS ---')
for w, n, p in kept:
    tag = ' [PRIORITY]' if p else ''
    print(f'  {w}: {n} samples{tag}')
print()
print(f'TOTAL: {len(kept)} words')
print('=' * 60)

# Start training
print("[INFO] Starting training now...")
cmd = [
    sys.executable, str(TRAIN_SCRIPT),
    "--data-dir", str(DATA),
    "--output-dir", str(OUTPUT_DIR),
    "--epochs", "200",
    "--batch-size", "64",
    "--lr", "5e-5",
    "--weight-decay", "0.08",
    "--warmup-epochs", "3",
    "--units", "512",
    "--num-blocks", "8",
    "--num-heads", "8",
    "--dropout", "0.35",
    "--label-smoothing", "0.1",
    "--mixup-alpha", "0.2",
    "--patience", "50",
    "--num-workers", "4",
    "--seed", "42",
]

# Load pretrained weights (finetune: load weights, reset optimizer for fresh LR schedule)
if PRETRAINED.exists():
    cmd.extend(["--resume", str(PRETRAINED), "--finetune"])
    print(f"[INFO] Fine-tuning from pretrained {PRETRAINED.name} (85% model)")
    print(f"[INFO] Fresh optimizer with lr=3e-4, reduced augmentation, SWA+TTA enabled")


print("\n" + "="*60)
print("Training v3 — target: 90%+ validation accuracy")
print("Key fixes: lr 3e-4→5e-5 (safe for finetune), warmup 8→3 epochs")
print("Features: reduced lips dropout, SWA + TTA enabled")
print("Model: 8-block Transformer, 512-dim, 3D landmarks (x,y,z)")
print("="*60 + "\n")
result = subprocess.run(cmd, cwd=str(BASE))

if result.returncode == 0:
    print("\n" + "="*60)
    print("Training DONE!")
    best = OUTPUT_DIR / "best_model.pth"
    log = OUTPUT_DIR / "training_log.csv"
    if log.exists():
        import csv
        best_acc = 0.0
        with open(log) as f:
            for row in csv.DictReader(f):
                try: best_acc = max(best_acc, float(row.get("val_acc", 0)))
                except: pass
        print(f"  Best val accuracy : {best_acc:.2f}%")
    print(f"  Best model saved  : {best}")
    print("\nDeploy to overlay:")
    dst = BASE / "SignFlow-Core" / "models"
    print(f'  python -c "import shutil; shutil.copy2(r\'{best}\', r\'{dst / "best_model.pth"}\'); '
          f'shutil.copy2(r\'{OUTPUT_DIR / "class_map.json"}\', r\'{dst / "class_map.json"}\'); print(\'Done\')"')

sys.exit(result.returncode)
