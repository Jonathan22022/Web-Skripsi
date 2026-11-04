import os
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel

# =====================================
# 1. KONFIGURASI
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    config = json.load(f)

MODEL_NAME = config["embedding_model"]
MAX_LEN = config["max_len"]
PREFIX = config["prefix"]

device = torch.device("cpu")

# =====================================
# 2. LOAD E5-SMALL-V2
# =====================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

embedding_model = AutoModel.from_pretrained(MODEL_NAME)
embedding_model.to(device)
embedding_model.eval()

# =====================================
# 3. LOAD XGBOOST
# =====================================

predictor = xgb.XGBRegressor()

predictor.load_model(
    os.path.join(MODEL_DIR, "xgb_prompt_quality.json")
)

# =====================================
# 4. FASTAPI
# =====================================

app = FastAPI(title="Prompt Quality Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100000)

# =====================================
# 5. PREPROCESSING
# =====================================

def clean_prompt(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# =====================================
# 6. AVERAGE POOLING
# =====================================

def average_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(
        last_hidden_state.dtype
    )

    summed = (last_hidden_state * mask).sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts

# =====================================
# 7. TEXT EMBEDDING
# =====================================

@torch.inference_mode()
def create_embedding(prompt):

    text = PREFIX + prompt

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    hidden = embedding_model(**encoded).last_hidden_state

    embedding = average_pool(
        hidden,
        encoded["attention_mask"]
    )

    embedding = F.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding.cpu().numpy().astype(np.float32)

# =====================================
# 8. PREDICTION API
# =====================================

@app.post("/predict")
def predict_quality(data: PromptRequest):

    prompt = clean_prompt(data.prompt)

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Prompt tidak boleh kosong."
        )

    try:
        # Generate embedding
        embedding = create_embedding(prompt)

        # Validasi dimensi
        if embedding.shape[1] != config["embedding_dim"]:
            raise ValueError("Dimensi embedding tidak sesuai.")

        # Predict menggunakan XGBoost
        prediction = float(predictor.predict(embedding)[0])

        return {
            "prompt": prompt,
            "score": prediction,
            "score_percentage": prediction * 100
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/")
def home():
    return {
        "message": "Prompt Quality Evaluation API is running"
    }