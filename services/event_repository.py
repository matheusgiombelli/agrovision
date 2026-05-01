import sqlite3

DB_PATH = "detections.db"


def list_recent_events(limit: int = 12) -> list[dict]:
    """Retorna os eventos mais recentes do banco."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, event_time, label, confidence, image_path
            FROM events
            ORDER BY event_time DESC
            LIMIT ?
            """,
            (limit,),
        )
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
        print(f"[event_repository] Erro ao consultar banco: {e}")
        return []