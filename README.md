# AgroVision AI

Sistema de monitoramento agrícola com visão computacional em tempo real, dados climáticos via web scraping e um agente conversacional com LLM local. A câmera detecta eventos no campo (pessoas, veículos), registra as ocorrências e permite consultar em linguagem natural o que está acontecendo na propriedade.

---

## Funcionalidades

- **Feed ao vivo** — frame da câmera com as caixas de detecção, atualizado a cada 2 segundos no dashboard.
- **Detecção de objetos** — YOLOv8 (`yolov8n.pt`) analisa cada frame e dispara alertas para classes de interesse (`person`, `car`, `motorcycle`, `truck`, `bus`).
- **Registro de eventos** — cada detecção relevante é salva no SQLite com horário, label, confiança e imagem capturada.
- **Condições climáticas** — dados do Open-Meteo (temperatura, vento, chuva da hora atual) com cache de 10 minutos.
- **Agente conversacional** — chat em linguagem natural via Ollama (LLM local). O agente recebe o contexto dos eventos recentes para responder com Leitura, Risco e Recomendação.

---

## Arquitetura

O projeto é dividido em camadas com responsabilidade única — rotas não acessam banco nem o modelo diretamente.

```
agrovision/
├── app.py                     # Entrada FastAPI + loop de captura da câmera
├── compose.yml                # Docker apenas para o Ollama (LLM)
├── requirements.txt
├── .env                       # Variáveis de ambiente (não versionado)
├── .env.example               # Modelo de referência
├── templates/
│   └── index.html             # Dashboard (Tailwind via CDN)
├── static/
│   └── captures/              # Imagens salvas dos eventos
└── services/
    ├── config.py              # Configurações do Ollama / agente
    ├── schemas.py             # Pydantic models (ChatRequest)
    ├── event_repository.py    # Única camada de acesso ao SQLite
    ├── vision_service.py      # Detecção YOLO e lógica de alertas
    ├── monitoring_agent.py    # Monta o contexto/mensagens do agente
    ├── weather_scraper.py     # Web scraping do clima (Open-Meteo)
    └── ollama_client.py       # Cliente HTTP para o Ollama
```

**Decisão de deploy:** o Docker é usado **apenas para o Ollama**. O app roda localmente (fora do container) porque a webcam USB não é acessível de dentro do Docker no Windows/macOS. Assim, câmera e LLM funcionam juntos.

---

## Pré-requisitos

- Python 3.10–3.12
- Docker + Docker Compose (para o Ollama)
- Webcam (índice `0`) ou uma URL de stream RTSP/HTTP

---

## Como rodar

### 1. Suba o Ollama no Docker

O `compose.yml` sobe o servidor Ollama e **baixa o modelo automaticamente** (`llama3.2:3b` por padrão) num serviço efêmero.

```powershell
docker compose up -d
```

Acompanhe o download do modelo, se quiser:

```powershell
docker compose logs -f ollama-pull
```

### 2. Rode o app localmente

```powershell
# Ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1
# Caso apareça erro de permissão:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Dependências
pip install -r requirements.txt

# Configuração (o .env já aponta para o Ollama em localhost)
Copy-Item .env.example .env

# Servidor
uvicorn app:app --reload --port 8000
```

No Linux/macOS, troque a ativação do venv por `source .venv/bin/activate`.

Acesse: **http://localhost:8000**

---

## Variáveis de Ambiente

Lidas via `.env` (`python-dotenv`). Veja `.env.example`.

| Variável                   | Padrão                              | Descrição                                                        |
|----------------------------|-------------------------------------|------------------------------------------------------------------|
| `OLLAMA_URL`               | `http://127.0.0.1:11434/api/chat`   | Endpoint de chat do Ollama.                                      |
| `OLLAMA_MODEL`             | `llama3.2:3b`                       | Modelo usado pelo agente.                                        |
| `OLLAMA_TIMEOUT`           | `120`                               | Timeout (s) das chamadas ao Ollama.                             |
| `AGENT_EVENT_LIMIT`        | `12`                                | Nº de eventos enviados como contexto ao agente.                 |
| `CAMERA_SOURCE`            | `0`                                 | Índice da webcam ou URL de stream (RTSP/HTTP).                  |
| `CAMERA_RECONNECT_SECONDS` | `5`                                 | Espera antes de reconectar a câmera após falha.                |
| `DB_PATH`                  | `detections.db`                     | Caminho do banco SQLite.                                         |

> `OLLAMA_MODEL` também é lido pelo `compose.yml`: `OLLAMA_MODEL=qwen2.5:3b docker compose up -d` troca o modelo baixado.

---

## Câmera: local vs. Docker

A webcam USB **só funciona com o app rodando localmente** (como acima). Para colocar o app inteiro no Docker, é preciso usar um **stream de rede** em vez da webcam:

```
CAMERA_SOURCE=http://192.168.0.10:8080/video        # MJPEG (ex.: app "IP Webcam")
CAMERA_SOURCE=rtsp://user:senha@192.168.0.10:554/...  # câmera IP / RTSP
```

O código aceita índice numérico (webcam) ou URL (stream) na mesma variável.

---

## Rotas da API

| Método | Rota            | Descrição                                              |
|--------|-----------------|--------------------------------------------------------|
| `GET`  | `/`             | Dashboard com os eventos mais recentes.               |
| `GET`  | `/frame`        | Frame atual da câmera em JPEG (`503` se ainda não há). |
| `GET`  | `/events`       | Últimos 50 eventos em JSON.                            |
| `GET`  | `/weather`      | Dados climáticos em JSON (cache 10 min).               |
| `POST` | `/chat`         | Pergunta ao agente; resposta contextualizada.         |
| `GET`  | `/agent/status` | Resumo do estado do agente e do contexto.             |
| `GET`  | `/health`       | Health check.                                          |

### `POST /chat`

```json
// Request
{ "message": "Qual o risco com base nos eventos de hoje?", "history": [] }

// Response
{ "answer": "Leitura: ...\nRisco: moderado ...\nRecomendação: ..." }
```

### `GET /weather`

```json
{
  "temperature_c": 18.2,
  "wind_speed_kmh": 12.2,
  "weather_code": 3,
  "weather_description": "Nublado",
  "is_day": false,
  "rain_probability_pct": 53,
  "fetched_at": "21:17:54"
}
```

> A probabilidade de chuva corresponde à **hora atual** (selecionada do array horário do Open-Meteo), não a um valor fixo do início do dia.

---

## Como funciona a detecção

1. A thread `process_stream` (em `app.py`) lê frames da câmera continuamente.
2. Cada frame passa pelo **YOLOv8** com confiança mínima de `0.45`; `vision_service` desenha as caixas e retorna os labels de interesse.
3. Um contador exige **3 frames consecutivos** com a mesma classe para evitar falsos positivos, e há **cooldown de 20s** por classe para não duplicar alertas.
4. Ao disparar, a imagem é salva em `static/captures/` e o evento é persistido no SQLite via `event_repository` (única camada que toca o banco).
5. O frame mais recente fica em memória para a rota `/frame`.

---

## Camada de Web Scraping (clima)

`services/weather_scraper.py` coleta a previsão do **Open-Meteo** (fonte pública e gratuita, sem chave de API). Boas práticas aplicadas:

- **Cache em memória de 10 minutos** para não sobrecarregar a fonte.
- **Tratamento de erros** (timeout, conexão, HTTP, campo ausente) com fallback para o último cache válido.
- Dados organizados em **JSON estruturado** e integrados ao dashboard.

**Relevância:** condições climáticas afetam o risco operacional das detecções — ex.: veículo detectado à noite com alta probabilidade de chuva indica risco maior.

---

## Stack

| Pacote             | Versão    | Uso                                  |
|--------------------|-----------|--------------------------------------|
| `fastapi`          | 0.115.0   | Framework web e roteamento           |
| `uvicorn`          | 0.30.6    | Servidor ASGI                        |
| `opencv-python`    | 4.10.0.84 | Captura de câmera e encode JPEG      |
| `ultralytics`      | 8.3.0     | YOLOv8 para detecção                 |
| `jinja2`           | 3.1.4     | Template do dashboard                |
| `httpx`            | 0.27.0    | Requisições HTTP (clima e Ollama)    |
| `python-dotenv`    | 1.0.1     | Leitura do `.env`                    |
| `python-multipart` | 0.0.9     | Suporte a form data no FastAPI       |

Infra: **Ollama** (LLM local) em container Docker · **SQLite** para persistência.
