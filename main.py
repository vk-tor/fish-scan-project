# ════════════════════════════════════════════════════════════
# main.py — TáFresco API v1.0
# Rodar: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ════════════════════════════════════════════════════════════
import os
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from datetime import datetime

# ── Configurações ──────────────────────────────────────────
IMG_SIZE     = (224, 224)
MODEL_PATH   = "/home/tafresco/projeto_tafresco/models/tafresco_efficientb0_v1_final.keras"
LOG_DIR      = "logs_campo"
os.makedirs(LOG_DIR, exist_ok=True)

# ── Classe de frescor com emoji e cor para o frontend ──────
CLASSES = {
    0: {"nome": "Fresh",        "emoji": "🟡", "cor": "#FFC107"},
    1: {"nome": "Highly_Fresh", "emoji": "🟢", "cor": "#4CAF50"},
    2: {"nome": "Not_Fresh",    "emoji": "🔴", "cor": "#F44336"},
}
# ATENÇÃO: confirme a ordem real das suas classes com:
# tf.keras.utils.image_dataset_from_directory(...).class_names

# ── Carrega modelo uma única vez na inicialização ──────────
print("🧠 Carregando modelo TáFresco...")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo pronto.")

app = FastAPI(
    title="TáFresco API",
    description="Classificação de frescor de peixes — MVP v1.0",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Em produção: restrinja ao domínio do app
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Utilitário de preparação de imagem ────────────────────
def preparar_imagem(imagem_bytes: bytes) -> np.ndarray:
    """
    Abre, converte para RGB e redimensiona para 224x224.
    EfficientNetB0 normaliza internamente — sem preprocess_input.
    """
    try:
        img = Image.open(io.BytesIO(imagem_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)       # Shape: (224, 224, 3) em [0, 255]
        return np.expand_dims(arr, axis=0)           # Shape: (1, 224, 224, 3)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {str(e)}")

# ══════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "projeto": "TáFresco",
        "versao":  "1.0.0",
        "modelo":  "EfficientNetB0",
        "status":  "online",
        "endpoints": ["/health", "/predict", "/predict_e_logar", "/docs"]
    }

@app.get("/health")
def health():
    """Verifica se o servidor e o modelo estão ativos."""
    return {
        "status":        "online",
        "modelo_carregado": MODEL is not None,
        "input_shape":   str(MODEL.input_shape),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recebe uma imagem e retorna a classificação de frescor.
    
    Retorna:
    - classe: nome da classe vencedora
    - confianca: probabilidade da classe vencedora (0.0 a 1.0)
    - breakdown: probabilidade de cada classe
    - alerta_baixa_confianca: True se confiança < 65%
    """
    # Valida o tipo do arquivo antes de processar
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Envie apenas arquivos de imagem.")

    imagem_bytes = await file.read()
    tensor       = preparar_imagem(imagem_bytes)
    predicoes    = MODEL.predict(tensor, verbose=0)[0]   # Array com 3 probabilidades

    classe_idx = int(np.argmax(predicoes))
    confianca  = float(predicoes[classe_idx])
    info_classe = CLASSES[classe_idx]

    return {
        "classe":     info_classe["nome"],
        "emoji":      info_classe["emoji"],
        "cor":        info_classe["cor"],
        "confianca":  round(confianca, 4),
        "breakdown": {
            CLASSES[i]["nome"]: round(float(p), 4)
            for i, p in enumerate(predicoes)
        },
        # Sinal útil para o app: "tire outra foto com melhor luz"
        "alerta_baixa_confianca": confianca < 0.65,
    }

@app.post("/predict_e_logar")
async def predict_e_logar(file: UploadFile = File(...)):
    """
    Igual ao /predict, mas salva a imagem em logs_campo/.
    Cada foto vira dado de treino para a v2 automaticamente.
    Nome do arquivo codifica: data_hora_CLASSE_id.jpg
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Envie apenas arquivos de imagem.")

    imagem_bytes = await file.read()
    tensor       = preparar_imagem(imagem_bytes)
    predicoes    = MODEL.predict(tensor, verbose=0)[0]

    classe_idx  = int(np.argmax(predicoes))
    confianca   = float(predicoes[classe_idx])
    info_classe = CLASSES[classe_idx]

    # Salva com nome descritivo — facilita revisão manual depois
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{LOG_DIR}/{timestamp}_{info_classe['nome']}_{uuid.uuid4().hex[:6]}.jpg"
    with open(nome_arquivo, "wb") as f:
        f.write(imagem_bytes)

    return {
        "classe":          info_classe["nome"],
        "emoji":           info_classe["emoji"],
        "confianca":       round(confianca, 4),
        "breakdown": {
            CLASSES[i]["nome"]: round(float(p), 4)
            for i, p in enumerate(predicoes)
        },
        "alerta_baixa_confianca": confianca < 0.65,
        "arquivo_salvo":   nome_arquivo,   # Confirmação visual de que logou
    }

@app.get("/estatisticas_campo")
def estatisticas_campo():
    """
    Conta quantas fotos foram coletadas por classe no campo.
    Útil para acompanhar a coleta de dados para a v2.
    """
    if not os.path.exists(LOG_DIR):
        return {"total": 0, "por_classe": {}}

    arquivos = [f for f in os.listdir(LOG_DIR) if f.endswith(".jpg")]
    contagem = {info["nome"]: 0 for info in CLASSES.values()}

    for arquivo in arquivos:
        for info in CLASSES.values():
            if f"_{info['nome']}_" in arquivo:
                contagem[info["nome"]] += 1
                break

    return {
        "total":      len(arquivos),
        "por_classe": contagem,
        "diretorio":  os.path.abspath(LOG_DIR)
    }