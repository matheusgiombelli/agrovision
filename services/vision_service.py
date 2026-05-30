"""
services/vision_service.py
Responsabilidade única: tudo relacionado à detecção YOLO.
- Não acessa banco de dados.
- Não conhece rotas HTTP.
- Recebe um frame, devolve resultados estruturados.
"""
import cv2
import time
import uuid
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

TARGET_CLASSES = {"person", "car", "motorcycle", "truck", "bus"}
MIN_CONSECUTIVE_FRAMES = 3
ALERT_COOLDOWN_SECONDS = 20
SAVE_DIR = "static/captures"

# Estado interno do serviço (isolado do app.py)
_detection_state: dict[str, int] = defaultdict(int)
_last_alert_time: dict[str, float] = defaultdict(lambda: 0.0)


def draw_box(frame, x1: int, y1: int, x2: int, y2: int, label: str, conf: float) -> None:
    text = f"{label} {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame, text, (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )


def _should_alert(label: str) -> bool:
    return (time.time() - _last_alert_time[label]) > ALERT_COOLDOWN_SECONDS


def run_detection(frame, model, confidence_threshold: float) -> tuple[set, dict]:
    """
    Executa o modelo YOLO no frame.
    Desenha as caixas no frame (in-place).
    Retorna: (labels encontrados, melhor confiança por label).
    """
    results = model(frame, conf=confidence_threshold, verbose=False)
    found_labels: set[str] = set()
    best_conf: dict[str, float] = {}

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.names[cls_id]
            if label not in TARGET_CLASSES:
                continue
            found_labels.add(label)
            if label not in best_conf or conf > best_conf[label]:
                best_conf[label] = conf
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            draw_box(frame, x1, y1, x2, y2, label, conf)

    return found_labels, best_conf


def update_state(found_labels: set) -> None:
    """
    Atualiza o contador de frames consecutivos por label.
    Labels não detectados no frame atual têm contador zerado.
    """
    for label in TARGET_CLASSES:
        if label in found_labels:
            _detection_state[label] += 1
        else:
            _detection_state[label] = 0


def check_alerts(found_labels: set, best_conf: dict, frame) -> list[dict]:
    """
    Verifica quais labels atingiram o limiar de frames consecutivos
    e respeitam o cooldown. Para cada alerta:
    - Salva o frame como imagem .jpg
    - Registra o tempo do alerta
    - Retorna lista de dicts prontos para persistir no banco

    NÃO salva no banco — responsabilidade do chamador (app.py).
    """
    events: list[dict] = []

    for label in found_labels:
        consecutive = _detection_state.get(label, 0)
        if consecutive >= MIN_CONSECUTIVE_FRAMES and _should_alert(label):
            event_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{label}_{event_id}.jpg"
            filepath = f"{SAVE_DIR}/{filename}"

            cv2.imwrite(filepath, frame)
            _last_alert_time[label] = time.time()

            logger.info(f"[ALERTA] {label} detectado — salvo em {filepath}")

            events.append({
                "id": event_id,
                "label": label,
                "confidence": best_conf.get(label, 0.0),
                "image_path": f"/static/captures/{filename}",
            })

    return events