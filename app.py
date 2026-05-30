"""
app.py
Ponto de entrada da aplicação AgroVision AI.
Responsabilidade: orquestrar serviços e expor rotas HTTP.
- Não acessa banco diretamente (usa event_repository).
- Não roda detecção diretamente (usa vision_service).
- Não faz scraping diretamente (usa weather_scraper).
"""
import os
import cv2
import time
import threading
import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

from services.event_repository import init_db, save_event, list_recent_events
from services.monitoring_agent import build_agent_messages, get_agent_status
from services.vision_service import run_detection, update_state, check_alerts
from services.weather_scraper import fetch_weather
from services import ollama_client
from services.config import AGENT_EVENT_LIMIT
from services.schemas import ChatRequest

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuração ──────────────────────────────────────────────────────────────
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.45
SAVE_DIR = "static/captures"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AgroVision AI")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLO(MODEL_PATH)

last_frame = None
last_frame_lock = threading.Lock()


# ── Loop de captura ───────────────────────────────────────────────────────────
def process_stream() -> None:
    """
    Captura frames da câmera em loop.
    Delega detecção ao vision_service e persistência ao event_repository.
    """
    global last_frame

    source: int | str = CAMERA_SOURCE
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # mantém como string (URL de stream)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"[câmera] Não foi possível abrir: {source}")
        return

    logger.info(f"[câmera] Iniciada: {source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.warning("[câmera] Frame não lido. Reconectando...")
            time.sleep(int(os.getenv("CAMERA_RECONNECT_SECONDS", 5)))
            cap = cv2.VideoCapture(source)
            continue

        # 1. Detectar objetos no frame
        found_labels, best_conf = run_detection(frame, model, CONFIDENCE_THRESHOLD)

        # 2. Atualizar contadores de frames consecutivos
        update_state(found_labels)

        # 3. Verificar quais labels geram alerta e salvar imagem
        events = check_alerts(found_labels, best_conf, frame)

        # 4. Persistir eventos no banco (único ponto de acesso ao DB)
        for event in events:
            save_event(
                event_id=event["id"],
                label=event["label"],
                confidence=event["confidence"],
                image_path=event["image_path"],
            )

        # 5. Atualizar frame compartilhado com a rota /frame
        with last_frame_lock:
            last_frame = frame.copy()

        time.sleep(0.05)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event() -> None:
    init_db()
    thread = threading.Thread(target=process_stream, daemon=True)
    thread.start()
    logger.info("[app] AgroVision AI iniciado.")


# ── Rotas ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    events = list_recent_events(20)
    return templates.TemplateResponse("index.html", {"request": request, "events": events})


@app.get("/health")
def health():
    return {"status": "ok", "service": "AgroVision AI"}


@app.get("/events")
def get_events():
    return JSONResponse(content=list_recent_events(50))


@app.get("/frame")
def get_frame():
    global last_frame
    with last_frame_lock:
        if last_frame is None:
            return JSONResponse(
                content={"message": "Ainda sem frame disponível."},
                status_code=503,
            )
        success, buffer = cv2.imencode(".jpg", last_frame)
        if not success:
            return JSONResponse(
                content={"message": "Erro ao processar frame."},
                status_code=500,
            )
        return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/chat")
def chat(req: ChatRequest):
    events = list_recent_events(limit=AGENT_EVENT_LIMIT)
    messages = build_agent_messages(
        question=req.message,
        history=req.history or [],
        events=events,
    )
    answer = ollama_client.chat(messages)
    return {"answer": answer}


@app.get("/agent/status")
def agent_status():
    events = list_recent_events(limit=AGENT_EVENT_LIMIT)
    return get_agent_status(events)


@app.get("/weather")
def get_weather():
    """
    Retorna dados climáticos coletados via web scraping do Open-Meteo.
    Relevância: condições climáticas afetam o risco operacional das detecções.
    Cache de 10 minutos — não sobrecarrega a fonte externa.
    """
    data = fetch_weather()
    if data is None:
        return JSONResponse(
            content={"error": "Dados climáticos indisponíveis no momento."},
            status_code=503,
        )
    return JSONResponse(content=data)