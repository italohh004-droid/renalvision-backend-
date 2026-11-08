from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import torch.nn as nn
import cv2
import numpy as np


app = FastAPI(
    title="RENALVISION API",
    description="API para identificación de cálculos renales",
    version="1.0"
)


# =========================
# MODELO
# =========================

class CNNRenal(nn.Module):
    def __init__(self):
        super().__init__()

        def bloque(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.f = nn.Sequential(
            bloque(1, 32),
            bloque(32, 64),
            bloque(64, 128)
        )

        self.pool = nn.AdaptiveAvgPool2d(4)

        self.c = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.c(self.pool(self.f(x)))


device = torch.device("cpu")

modelo = CNNRenal().to(device)

MODEL_PATH = Path(__file__).resolve().parent / "cnn_renal.pt"

estado = torch.load(
    MODEL_PATH,
    map_location=device
)

modelo.load_state_dict(estado)
modelo.eval()


# =========================
# PREPROCESAMIENTO
# =========================

def preprocesar_imagen(gray):
    gray = cv2.medianBlur(gray, 3)

    gray = cv2.resize(
        gray,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    x = torch.from_numpy(gray[None]).float().div(255).unsqueeze(1)

    x = (x - 0.5) / 0.5

    return x.to(device)


# =========================
# ENDPOINTS
# =========================

@app.get("/")
def inicio():
    return {
        "status": "ok",
        "message": "RENALVISION API funcionando"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes JPG o PNG."
        )

    contenido = await file.read()

    array = np.frombuffer(contenido, np.uint8)

    gray = cv2.imdecode(
        array,
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer la imagen."
        )

    x = preprocesar_imagen(gray)

    with torch.no_grad():
        salida = modelo(x)
        probabilidades = torch.softmax(salida, dim=1)[0]

    prob_normal = float(probabilidades[0])
    prob_calculo = float(probabilidades[1])

    if prob_calculo >= 0.5:
        clase = "CALCULO_RENAL"
        etiqueta = "Cálculo renal"
    else:
        clase = "NORMAL"
        etiqueta = "Normal"

    return {
        "success": True,
        "filename": file.filename,
        "classification": clase,
        "label": etiqueta,
        "probability_normal": prob_normal,
        "probability_stone": prob_calculo
    }
