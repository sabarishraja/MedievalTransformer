# 🏰 CanterburyGPT

A custom **Transformer language model** (GPT-like) built from scratch in PyTorch.  
It is trained on *The Canterbury Tales* (Middle English) and fine-tuned on a second dataset to generate text in **two distinct styles**.  
This project was built as part of the **Open Avenues Build Fellowship** under mentor **Kacper Raczy**.

---

## 📖 Project Overview
The project demonstrates:
- Building a **Transformer model from scratch** (multi-head self-attention, feed-forward layers, positional embeddings).
- Training on **The Canterbury Tales** dataset (style = medieval English).
- Fine-tuning on a second dataset to enable **style-conditioned text generation**.
- Evaluating generations using **Perplexity, ROUGE, BERTScore, Accuracy**.

---
## ⚡ Installation & Setup

### 1. Clone the repo
```bash
git clone https://github.com/sabarishraja/MedievalTransformer
cd MedievalTransformer

### 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Train on Canterbury Tales
 python train.py --input input.txt --finetune-input finetune_input.txt --batch-size 32 --context-size 256 --n-embd 384 --n-head 6 --n-layer 6 --dropout 0.2 finetune --load model.pth --save model_finetuned.pth --steps 5000 --report 500 --lr 5e-5

### 5. Fine-tune on 2nd style along with prompt for evaluation
python train.py --input input.txt --finetune-input finetune_input.txt --batch-size 32 --context-size 256 --n-embd 384 --n-head 6 --n-layer 6 --dropout 0.2 eval --load model_finetuned.pth --prompt "WHAN that Aprille with his shoures soote, " --token-count 300 --style 0
