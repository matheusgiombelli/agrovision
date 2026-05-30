"""
services/ollama_client.py
Cliente HTTP para o Ollama (LLM local).
- Erros técnicos são logados internamente.
- Usuário recebe apenas mensagens genéricas e amigáveis.
"""
import httpx
import logging
from services.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


def chat(messages: list[dict]) -> str:
    """
    Envia uma lista de mensagens ao Ollama e retorna o texto da resposta.
    Em caso de erro, loga o detalhe técnico e retorna mensagem amigável.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            response = client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "Sem resposta do modelo.")

    except httpx.ConnectError:
        logger.error(f"[ollama] Falha de conexão com {OLLAMA_URL}")
        return "Não foi possível conectar ao agente. Verifique se o Ollama está ativo."

    except httpx.TimeoutException:
        logger.error(f"[ollama] Timeout após {OLLAMA_TIMEOUT}s")
        return "O agente demorou para responder. Tente novamente em instantes."

    except httpx.HTTPStatusError as e:
        logger.error(f"[ollama] Erro HTTP {e.response.status_code}")
        return "O agente retornou um erro. Tente novamente."

    except Exception as e:
        logger.error(f"[ollama] Erro inesperado: {e}")
        return "Ocorreu um erro interno. Tente novamente em instantes."