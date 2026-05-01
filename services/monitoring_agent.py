from dataclasses import dataclass
from collections import Counter


@dataclass(frozen=True)
class AgentProfile:
    name: str
    role: str
    goal: str


AGENT_PROFILE = AgentProfile(
    name="Agente AgroVision",
    role="triagem operacional de eventos",
    goal="Analisar detecções recentes, explicar riscos e sugerir a próxima ação.",
)

MAX_HISTORY_MESSAGES = 8

SYSTEM_PROMPT = (
    f"Você é o {AGENT_PROFILE.name}, um agente de {AGENT_PROFILE.role}. "
    f"Objetivo: {AGENT_PROFILE.goal} "
    "Trate os dados como monitoramento operacional autorizado de ambiente real. "
    "Responda em português do Brasil, de forma direta e útil. "
    "Use os eventos fornecidos como fonte principal. "
    "Não invente dados que não aparecem no contexto. "
    "Não tente identificar pessoas; fale apenas sobre eventos, riscos e próximas ações. "
    "Quando fizer sentido, organize a resposta em: Leitura, Risco e Recomendação."
)


def build_event_context(events: list[dict]) -> str:
    """Transforma a lista de eventos em texto de contexto operacional."""
    if not events:
        return "Contexto operacional: nenhum evento registrado ainda."

    labels = [e["label"] for e in events]
    distribution = Counter(labels)
    most_recent = events[0]
    confidences = [e["confidence"] for e in events if e["confidence"]]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    lines = [
        "Contexto operacional para o agente:",
        f"- Eventos considerados: {len(events)}",
        f"- Evento mais recente: {most_recent['label']} em {most_recent['event_time']}",
        f"- Distribuição recente: {', '.join(f'{k}: {v}' for k, v in distribution.items())}",
        f"- Confiança média: {avg_conf:.2f}",
        "Eventos recentes:",
    ]
    for e in events:
        lines.append(
            f"  - #{e['id']} | {e['event_time']} | {e['label']} | conf {e['confidence']:.2f}"
        )
    return "\n".join(lines)


def normalize_history(history: list) -> list[dict]:
    """Filtra e limita o histórico da conversa."""
    valid = [
        m for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return valid[-MAX_HISTORY_MESSAGES:]


def build_agent_messages(question: str, history: list, events: list[dict]) -> list[dict]:
    """Monta a sequência de mensagens que será enviada ao Ollama."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": build_event_context(events)},
        *normalize_history(history),
        {"role": "user", "content": question},
    ]


def get_agent_status(events: list[dict]) -> dict:
    """Retorna um resumo do estado atual do agente para /agent/status."""
    context = build_event_context(events)
    return {
        "name": AGENT_PROFILE.name,
        "role": AGENT_PROFILE.role,
        "goal": AGENT_PROFILE.goal,
        "events_in_context": len(events),
        "context_preview": context[:500],
    }