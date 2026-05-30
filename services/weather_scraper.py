"""
services/weather_scraper.py
Camada de web scraping — previsão do tempo via Open-Meteo.

Por que este dado é relevante para o AgroVision?
- O sistema monitora movimentação em ambiente rural via câmera.
- Condições climáticas afetam diretamente o risco operacional:
  chuva forte à noite com veículo detectado = risco elevado.
- A previsão é exibida no dashboard e pode ser incluída
  no contexto do agente de IA para respostas mais precisas.

Fonte: https://open-meteo.com (gratuita, sem chave de API)
Limite: cache local de 10 minutos para não sobrecarregar a fonte.
"""
import httpx
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 600  # 10 minutos entre requisições reais

# Cache em memória — evita requisições desnecessárias
_cache: dict = {"data": None, "fetched_at": 0.0}

# Mapeamento dos códigos WMO para descrições legíveis
WEATHER_CODES: dict[int, str] = {
    0: "Céu limpo",
    1: "Predominantemente limpo",
    2: "Parcialmente nublado",
    3: "Nublado",
    45: "Neblina",
    48: "Neblina com geada",
    51: "Garoa leve",
    53: "Garoa moderada",
    55: "Garoa intensa",
    61: "Chuva leve",
    63: "Chuva moderada",
    65: "Chuva forte",
    71: "Neve leve",
    73: "Neve moderada",
    75: "Neve forte",
    80: "Pancadas de chuva leves",
    81: "Pancadas de chuva moderadas",
    82: "Pancadas de chuva fortes",
    95: "Tempestade",
    96: "Tempestade com granizo leve",
    99: "Tempestade com granizo forte",
}


def fetch_weather(lat: float = -24.55, lon: float = -54.06) -> Optional[dict]:
    """
    Busca previsão do tempo para as coordenadas informadas.
    Padrão: Marechal Cândido Rondon, PR (-24.55, -54.06).

    Retorna dict estruturado ou None se falhar e não houver cache.
    Em caso de falha com cache disponível, retorna o cache mesmo expirado.
    """
    now = time.time()

    # Retorna cache se ainda válido
    if _cache["data"] and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
        logger.debug("[weather] Retornando dados do cache.")
        return _cache["data"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "precipitation_probability",
        "forecast_days": 1,
        "timezone": "America/Sao_Paulo",
    }

    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        code = int(data["current_weather"]["weathercode"])

        # A probabilidade de chuva vem como um array horário (24 valores).
        # Selecionamos o índice da hora atual — não o [0], que é meia-noite.
        current_time = data["current_weather"]["time"]          # ex.: "2026-05-29T20:30"
        current_hour = current_time[:13]                        # ex.: "2026-05-29T20"
        hourly_times = data["hourly"]["time"]
        hourly_rain = data["hourly"]["precipitation_probability"]
        rain_index = next(
            (i for i, t in enumerate(hourly_times) if t[:13] == current_hour),
            0,  # fallback para meia-noite se não encontrar a hora
        )

        result = {
            "temperature_c": data["current_weather"]["temperature"],
            "wind_speed_kmh": data["current_weather"]["windspeed"],
            "weather_code": code,
            "weather_description": WEATHER_CODES.get(code, "Condição desconhecida"),
            "is_day": bool(data["current_weather"]["is_day"]),
            "rain_probability_pct": hourly_rain[rain_index],
            "fetched_at": time.strftime("%H:%M:%S", time.localtime(now)),
        }

        _cache["data"] = result
        _cache["fetched_at"] = now
        logger.info("[weather] Dados climáticos atualizados com sucesso.")
        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"[weather] Erro HTTP {e.response.status_code}: {e.request.url}")
    except httpx.TimeoutException:
        logger.error("[weather] Timeout ao consultar Open-Meteo.")
    except httpx.RequestError as e:
        logger.error(f"[weather] Falha de conexão: {e}")
    except KeyError as e:
        logger.error(f"[weather] Campo ausente na resposta da API: {e}")
    except Exception as e:
        logger.error(f"[weather] Erro inesperado: {e}")

    # Se falhou mas tem cache (mesmo expirado), retorna o cache com aviso
    if _cache["data"]:
        logger.warning("[weather] Retornando cache expirado por falha na requisição.")
        return _cache["data"]

    return None