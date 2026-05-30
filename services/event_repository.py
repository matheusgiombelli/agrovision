"""
services/event_repository.py
Camada exclusiva de acesso ao banco de dados.
- Nenhum outro arquivo deve chamar sqlite3 diretamente.
- Toda lógica de persistência passa por aqui.
"""
import os
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
DB_PATH = os.getenv("DB_PATH", "detections.db")


def init_db() -> None:
    """Cria a tabela de eventos se ainda não existir."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                event_time  TEXT NOT NULL,
                label       TEXT NOT NULL,
                confidence  REAL NOT NULL,
                image_path  TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("[db] Banco inicializado.")
    except Exception as e:
        logger.error(f"[db] Erro ao inicializar banco: {e}")


def save_event(event_id: str, label: str, confidence: float, image_path: str) -> None:
    """Persiste um evento de detecção no banco."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO events (id, event_time, label, confidence, image_path)
            VALUES (?, ?, ?, ?, ?)
        """, (
            event_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            round(confidence, 4),
            image_path,
        ))
        conn.commit()
        conn.close()
        logger.info(f"[db] Evento salvo: {event_id} — {label}")
    except sqlite3.IntegrityError:
        logger.warning(f"[db] Evento duplicado ignorado: {event_id}")
    except Exception as e:
        logger.error(f"[db] Erro ao salvar evento {event_id}: {e}")


def list_recent_events(limit: int = 12) -> list[dict]:
    """Retorna os eventos mais recentes do banco."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            SELECT id, event_time, label, confidence, image_path
            FROM events
            ORDER BY event_time DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "event_time": r[1],
                "label": r[2],
                "confidence": r[3],
                "image_path": r[4],
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"[db] Erro ao consultar eventos: {e}")
        return []