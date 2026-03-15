# SignFlow — Real-time ASL Sign Language Recognition

Real-time American Sign Language (ASL) recognition with **sentence formation** and **LLM-powered grammar correction**. Uses MediaPipe landmarks and a multi-stream Transformer model.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)

## Features

- **Real-time webcam inference** at 30+ FPS
- **58 common ASL signs** with ~95% validation accuracy
- **Sentence formation** — hold a sign → drop hands → word added to sentence
- **LLM grammar correction** — fixes ASL word order to natural English
- **Next word prediction** — suggests likely next signs
- **Two LLM backends**: Local (Ollama + Mistral 7B) or Online (Google Gemini)
- **Multi-stream Transformer** (26.4M params)
- **GPU accelerated** with CUDA + TF32 (works on CPU too)

## Supported Signs (58 classes)

hello · thank you · help · sorry · please · yes · no · what is your name · I love you · I · he/she/it · what · how · when · where · stop · want · love · need · eat · drink · make · see · talk · wake · open · cry · stay · buy · cook · sad · hungry · hot · old · sick · thirsty · mad · cute · better · fine · day · night · yesterday · tomorrow · later · time · dad · mom · brother · man · boy · grandma · grandpa · water · food · milk · apple · home

## Quick Start

### 1. Clone

```bash
git clone https://github.com/nithin2719-commits/signflow-.git
cd signflow-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU recommended:** Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/)
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Run

```bash
# Basic (no LLM)
python sign_inference.py

# With local LLM (Ollama + Mistral — offline, best privacy)
python sign_inference.py --llm local

# With Google Gemini (online, free API)
python sign_inference.py --llm gemini --api-key YOUR_GEMINI_KEY
```

## LLM Setup

### Option 1: Local LLM (Ollama + Mistral 7B)

```bash
# Install Ollama from https://ollama.com
ollama pull mistral
python sign_inference.py --llm local
```

### Option 2: Google Gemini (Free API)

1. Get free API key: https://aistudio.google.com/apikey
2. Run:
```bash
python sign_inference.py --llm gemini --api-key YOUR_KEY
# Or set environment variable:
set GEMINI_API_KEY=YOUR_KEY
python sign_inference.py --llm gemini
```

## Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **B** | Backspace — remove last word from sentence |
| **C** | Clear entire sentence |
| **R** | Reset landmark buffer |

## How It Works

1. **MediaPipe** extracts 92 3D landmarks per frame (40 lip + 21 left hand + 21 right hand + 10 pose)
2. Rolling buffer of ~40 recent frames is maintained
3. Frames are downsampled to 64 and fed to the **multi-stream Transformer**
4. Model outputs probabilities with light exponential smoothing
5. Hold a sign ~0.5s → drop hands → word added to sentence
6. **LLM** corrects grammar and suggests next words in real-time (background thread)

## Display

- **Gray text** — raw signed words (as detected)
- **Green text** — LLM-corrected sentence (natural English)
- **Cyan text** — next word suggestions from LLM

## Architecture

```
Lips (40×3) ──→ LandmarkEmbedding ──┐
Left Hand (21×3) → LandmarkEmbedding ─┤  Learned Weighted
Right Hand (21×3) → LandmarkEmbedding ─┤  Fusion (softmax)
Pose (10×3) ──→ LandmarkEmbedding ──┘       │
                                             ↓
                                   MLP → 512d features
                                             ↓
                         8-Block Transformer (pre-norm, 8 heads)
                                             ↓
                         Masked Average Pooling → Dropout
                                             ↓
                                  Linear(512, 58) → Prediction
```

**26.4M parameters** | 512-dim | 8 attention heads | 0.35 dropout

## Project Structure

```
signflow-/
├── sign_inference.py              # Main inference + sentence formation
├── llm_helper.py                  # LLM backends (Ollama + Gemini)
├── train_demo_20.py               # Train 58-class focused model
├── train_common_words.py          # Train common words model
├── train_landmark_transformer.py  # Full transformer training
├── models/
│   ├── best_model.pth             # Trained 58-class model (~95% val acc)
│   └── class_map.json             # Class index → display name mapping
├── mediapipe_models/
│   ├── face_landmarker.task
│   ├── hand_landmarker.task
│   └── pose_landmarker_heavy.task
├── requirements.txt
├── .gitattributes                 # Git LFS for large files
└── README.md
```

## Requirements

- Python 3.9+
- Webcam
- NVIDIA GPU recommended (works on CPU but slower)
- ~200MB disk space (model + MediaPipe files)
- Optional: Ollama (local LLM) or Gemini API key (online LLM)

## Training

Trained on MS-ASL + WLASL datasets using transfer learning:
- Pre-trained 621-class base model (79.7% val accuracy)
- Fine-tuned to 58 common signs (~95% val accuracy)
- AdamW optimizer with cosine annealing
- Mixup augmentation + label smoothing

## License

MIT License — for educational and research purposes.
