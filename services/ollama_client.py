import httpx
from services.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


def chat(messages: list[dict]) -> str:
    """
    Envia uma lista de mensagens ao Ollama e retorna o texto da resposta.
    Usa stream=False para simplicidade (compatível com o front atual).
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
        return (
            "Não foi possível conectar ao Ollama. "
            "Verifique se ele está rodando em http://127.0.0.1:11434"
        )
    except httpx.TimeoutException:
        return "O Ollama demorou demais para responder. Tente novamente."
    except Exception as e:
        return f"Erro ao consultar o Ollama: {e}"