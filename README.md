# 🤖 Cobot Detection & Segmentation

A modular real-time cobot perception pipeline supporting:

- Vision-based object detection & segmentation (YOLO-based)

- Natural-language object command parsing using on-device LLaMA (GGUF)

- Optional voice-controlled object selection (STT + LLM)
  
- GPU-accelerated inference with CPU fallback
  
- Clean separation between core vision+language and voice dependencies

---

## ✨ Features

- Zero-shot object detection via **text or speech**
  
- YOLO-based real-time segmentation
  
- Temporal stability filtering to avoid false positives
  
- Edge-aware bounding box validation
  
- Offline command parsing using **LLaMA (GGUF)**
  
- Offline speech-to-text using **Faster-Whisper**
  
- CUDA-enabled inference (optional)
  
- Clean dependency separation (`requirements.txt` vs `requirements_stt.txt`)

---

## 📁 Repository Structure

```text
cobot_detection_segmentation/
├── cobot.py                  # Vision + LLM-based command parsing (text input)
├── cobot_stt.py              # Voice-controlled detection (STT + LLM + Vision)

├── requirements.txt          # Vision + LLM dependencies (no audio)
├── requirements_stt.txt      # Full dependencies for cobot_stt.py

├── weights/                  # (Not tracked) Model weights directory
│   ├── yoloe-26l-seg.pt       # YOLO segmentation model (user-provided)
│   └── *.gguf                 # LLaMA GGUF model (auto-downloaded)

├── images/                   # (Not tracked) Output images from cobot.py
├── output_images/             # (Not tracked) Output images from cobot_stt.py

└── README.md
```

Note:

weights/, images/, and output_images/ are runtime-generated and are not tracked in this repository

## 🔧 Installation & Setup

1. Clone the repository
   
git clone https://github.com/Sinam36/cobot_detection_segmentation.git

cd cobot_detection_segmentation

2. Create and activate a virtual environment

Using conda (recommended):

conda create -n cobot python=3.10 -y

conda activate cobot

3. Install dependencies
   
#### For Text-based command parsing (LLM + Vision) (cobot.py):

pip install -r requirements.txt


#### For voice-controlled mode (STT + LLM + Vision) (cobot_stt.py):

pip install -r requirements_stt.txt


##### ⚠️ Dependency files are intentionally separated.

Install only what you need.

4. (Optional but recommended) Enable GPU support

If you have an NVIDIA GPU with CUDA installed:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Verify:

python -c "import torch; print(torch.cuda.is_available())"

## 📦 Model Weights

### YOLO Segmentation Model (Required)

Download the YOLO segmentation model from the official Ultralytics release:

👉 [Download yoloe-26l-seg-pf.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg-pf.pt)

Place the file at:

weights/yoloe-26l-seg.pt


The weights/ directory is created automatically if missing.


### LLaMA Model (Required for LLM Parsing)

For both cobot.py and cobot_stt.py, the LLaMA GGUF model is automatically downloaded on first run from Hugging Face:

Repository: bartowski/Llama-3.2-3B-Instruct-GGUF

File: Llama-3.2-3B-Instruct-Q4_K_M.gguf

No manual download is required.

## ▶️ Running the Code

#### Text-based command parsing (Vision + LLM)
python cobot.py

You will be prompted to enter a command such as:

```text
pick up the red cup
grab the bottle
```

The LLM extracts the target object and triggers detection.


#### Voice-controlled detection (STT + LLM + Vision)
python cobot_stt.py


Instructions:

```text
Press ENTER to speak

Speak a command (e.g. “pick up the bottle”)
```

The system parses the object and runs detection

## 🧠 How It Works (High-Level)


#### Vision Pipeline

Live camera feed → YOLO segmentation

Zero-shot class prompting

Temporal stability check

Edge-safe object capture

#### Language Pipeline

Text or speech command

LLaMA extracts object noun phrase

Parsed object passed to YOLO

#### Voice Pipeline (cobot_stt.py)

Microphone audio → Faster-Whisper (STT)

STT output → LLaMA parsing

Parsed object → Vision pipeline


##### All inference runs fully offline.


## ⚙️ System Requirements

OS: Windows 10/11 (tested)

Python: 3.10

Webcam (DirectShow compatible)

Microphone (for cobot_stt.py)

NVIDIA GPU recommended (CUDA 12.x)

CPU-only mode is supported but slower.

## 📌 Notes

Large model files are not committed

Output images are runtime artifacts

First run may take time due to model download

DirectShow (cv2.CAP_DSHOW) is used on Windows
