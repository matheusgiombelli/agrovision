# Relatório Técnico — AgroVision AI

Revisão de arquitetura, segurança e qualidade do código, com a implementação de uma camada de web scraping. O objetivo não é apenas "fazer funcionar", mas avaliar se o projeto está bem estruturado, seguro e preparado para crescer.

**Sumário**

- [Parte 1 — Revisão da Arquitetura](#parte-1--revisão-da-arquitetura)
- [Parte 2 — Revisão de Segurança](#parte-2--revisão-de-segurança)
- [Parte 3 — Melhoria do Código Gerado com IA](#parte-3--melhoria-do-código-gerado-com-ia)
- [Parte 4 — Camada de Web Scraping](#parte-4--camada-de-web-scraping)

---

# Parte 1 — Revisão da Arquitetura

O projeto adota uma separação em camadas, com o `app.py` atuando apenas como ponto de entrada e orquestração, e cada responsabilidade isolada em um módulo dentro de `services/`.

```
agrovision/
├── app.py                     # Entrada FastAPI + orquestração (não acessa banco nem modelo direto)
├── templates/index.html       # Frontend (dashboard)
└── services/
    ├── config.py              # Configurações
    ├── schemas.py             # Validação de entrada (Pydantic)
    ├── event_repository.py    # Única camada de acesso ao banco (SQLite)
    ├── vision_service.py      # Camada de IA/modelo (YOLO)
    ├── monitoring_agent.py    # Regra de negócio do agente
    ├── weather_scraper.py     # Camada de web scraping (Open-Meteo)
    └── ollama_client.py       # Integração externa (LLM local)
```

### Divisão das camadas

| Camada | Onde está | Situação |
|---|---|---|
| Frontend | `templates/index.html` | Separado; consome a API por `fetch`. |
| Backend / API | `app.py` | Concentra rotas e orquestra os serviços. |
| Banco de dados | `services/event_repository.py` | Isolado — único arquivo que usa `sqlite3`. |
| Serviços internos | `services/vision_service.py`, `monitoring_agent.py` | Lógica de detecção e do agente isoladas. |
| IA / modelo | `services/vision_service.py` | Chamada ao YOLO separada da regra de negócio. |
| Integração externa | `services/ollama_client.py` | Cliente HTTP do Ollama isolado. |
| Web scraping | `services/weather_scraper.py` | Serviço separado (detalhado na Parte 4). |

### Respostas à análise proposta

**A interface está apenas exibindo dados ou também possui regra de negócio indevida?**
A interface é majoritariamente de apresentação: busca dados nos endpoints e renderiza. Há apenas uma pequena lógica de exibição no JavaScript (ex.: marcar ⚠️ quando a probabilidade de chuva ≥ 70%), o que é aceitável por ser formatação visual e não decisão de negócio. Nenhuma persistência ou cálculo crítico acontece no front.

**O backend concentra a lógica principal do sistema?**
Sim. O `app.py` orquestra o fluxo (captura → detecção → persistência → exposição via rotas), mas delega a lógica concreta aos serviços, mantendo as rotas enxutas.

**O acesso ao banco está isolado em uma camada própria ou aparece espalhado pelo código?**
Isolado. `event_repository.py` é o único módulo que importa `sqlite3` e expõe `init_db()`, `save_event()` e `list_recent_events()`. Nenhuma rota executa SQL diretamente.

**A chamada ao modelo de IA/YOLO está separada da regra de negócio?**
Sim. `vision_service.py` recebe um frame e devolve resultados estruturados; não conhece banco nem rotas HTTP. A regra de alerta (frames consecutivos, cooldown) também vive nessa camada, separada do roteamento.

**A camada de scraping é um serviço separado ou ficou misturada em rotas/telas?**
Serviço separado (`weather_scraper.py`). A rota `/weather` apenas chama `fetch_weather()` e devolve o resultado.

### Pontos fortes e limitações

**Pontos fortes**
- Responsabilidade única bem definida por módulo.
- Acesso ao banco centralizado — trocar SQLite por outro SGBD exige mudar um arquivo.
- Modelo de IA e scraping desacoplados das rotas.

**Limitações identificadas (oportunidades de evolução)**
- O laço de captura (`process_stream`) roda como *thread* dentro do mesmo processo da API. Funciona, mas acopla câmera e servidor web; para escalar, um *worker* separado (ou fila de mensagens) seria mais robusto.
- Estado compartilhado em memória (`last_frame` com lock) impede rodar múltiplas instâncias do app atrás de um balanceador.
- Ausência de testes automatizados e de autenticação (ver Parte 2).

---

# Parte 2 — Revisão de Segurança

> As soluções abaixo são **recomendações** a partir da revisão; ainda não estão aplicadas no código atual. O objetivo desta parte é identificar os riscos e indicar como tratá-los.

## 1. Rotas da API abertas sem autenticação

### Problema encontrado

Todas as rotas estão acessíveis sem autenticação. Qualquer pessoa com acesso à rede onde o servidor roda pode consultar dados, assistir ao feed da câmera e interagir com o agente de IA sem se identificar.

| Rota | Método | Risco |
|---|---|---|
| `/events` | GET | Expõe histórico completo de detecções com horários e imagens |
| `/frame` | GET | Transmite o frame atual da câmera em tempo real |
| `/chat` | POST | Permite interagir com o agente de IA sem restrição |
| `/weather` | GET | Dado público, risco baixo, mas segue o mesmo padrão inseguro |
| `/agent/status` | GET | Expõe configuração interna do agente e contexto operacional |

### Código original (problemático)

```python
# app.py — rota completamente aberta
@app.get("/events")
def get_events():
    return JSONResponse(content=list_recent_events(50))

@app.get("/frame")
def get_frame():
    ...
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
```

Não há verificação de identidade antes de retornar os dados.

### Como seria resolvido

Proteger as rotas com um token de API simples usando o sistema de dependências do FastAPI:

```python
# services/security.py — como ficaria
from fastapi import Header, HTTPException
from services.config import API_TOKEN

def verify_token(x_api_token: str = Header(...)):
    if x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")
```

```python
# app.py — rota protegida
from fastapi import Depends
from services.security import verify_token

@app.get("/events", dependencies=[Depends(verify_token)])
def get_events():
    return JSONResponse(content=list_recent_events(50))
```

```env
# .env — token configurado por variável de ambiente
API_TOKEN=meu_token_secreto_aqui
```

### Por que esta solução é melhor

- O token fica no `.env` e nunca vai para o repositório.
- O FastAPI rejeita automaticamente requisições sem o header correto.
- Não exige banco de usuários nem sistema de login completo.
- Pode evoluir para JWT no futuro sem mudar a estrutura.

---

## 2. Rota `/chat` sem limite de tamanho de mensagem

### Problema encontrado

A rota `/chat` aceita qualquer mensagem sem validar o tamanho do conteúdo, o que abre dois riscos:

**Risco 1 — Sobrecarga do Ollama:** uma mensagem muito longa é enviada diretamente ao modelo, que pode travar, demorar muito ou consumir todos os recursos do servidor.

**Risco 2 — Abuso do sistema:** sem limite, um usuário pode automatizar envios massivos com mensagens gigantes, tornando o sistema inacessível para os demais (negação de serviço simples).

### Código original (problemático)

```python
# services/schemas.py
class ChatRequest(BaseModel):
    message: str                       # aceita qualquer tamanho
    history: Optional[list] = []
```

### Como seria resolvido

Validar o tamanho diretamente no schema Pydantic e limitar o histórico:

```python
# services/schemas.py — como ficaria
from pydantic import BaseModel, Field
from typing import Optional

MAX_MESSAGE_LENGTH = 1000   # caracteres
MAX_HISTORY_ITEMS  = 10     # mensagens no histórico

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Pergunta ao agente (máximo 1000 caracteres).",
    )
    history: Optional[list] = Field(default=[], max_length=MAX_HISTORY_ITEMS)
```

Com isso, o FastAPI rejeita automaticamente mensagens acima do limite com status 422, sem nem chegar ao Ollama.

### Por que esta solução é melhor

- A validação acontece antes de qualquer processamento.
- Não exige código extra — o Pydantic resolve sozinho.
- Protege o Ollama de sobrecarga.
- O erro retornado ao usuário é claro e padronizado.

---

## 3. `CAMERA_SOURCE` sem validação

### Problema encontrado

A variável `CAMERA_SOURCE` é lida e repassada ao `cv2.VideoCapture()` sem verificação do valor recebido.

```python
# app.py
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
```

Riscos:

**Risco 1 — Apontamento para fonte maliciosa:** quem alterar a variável (acesso ao servidor, `.env` exposto ou CI/CD comprometido) pode redirecionar a câmera para qualquer URL externa arbitrária.

**Risco 2 — Travamento silencioso:** um valor inválido não causa erro imediato — o `cv2.VideoCapture()` falha em abrir e o sistema fica sem câmera sem avisar claramente o motivo.

**Risco 3 — Consumo de recursos:** uma URL de stream muito pesada ou infinita pode consumir banda e memória sem controle.

### Como seria resolvido

Validar se o valor é um índice de câmera local ou uma URL com protocolo permitido:

```python
# services/config.py — como ficaria
ALLOWED_STREAM_PROTOCOLS = ("rtsp://", "http://", "https://")

def validate_camera_source(value: str) -> str | int:
    """Aceita índice de câmera local (0–10) ou URL com protocolo conhecido.
    Qualquer outro valor cai em fallback seguro (câmera 0)."""
    try:
        index = int(value)
        if 0 <= index <= 10:
            return index
        raise ValueError("Índice fora do intervalo permitido.")
    except ValueError:
        pass

    if any(value.startswith(p) for p in ALLOWED_STREAM_PROTOCOLS):
        return value

    logger.warning(
        f"[config] CAMERA_SOURCE inválido: '{value}'. Usando câmera local 0 como fallback."
    )
    return 0
```

```python
# app.py — com validação
from services.config import validate_camera_source
CAMERA_SOURCE = validate_camera_source(os.getenv("CAMERA_SOURCE", "0"))
```

### Por que esta solução é melhor

- Impede que valores arbitrários cheguem ao OpenCV.
- Define explicitamente quais protocolos de stream são aceitos.
- Em caso de valor inválido, o sistema não trava — usa fallback seguro.
- O log deixa claro o que aconteceu e por quê.

---

## Resumo dos Riscos e Soluções

| # | Risco | Severidade | Solução recomendada |
|---|---|---|---|
| 1 | Rotas abertas sem autenticação | Alta | Token de API via header + `Depends()` do FastAPI |
| 2 | `/chat` sem limite de mensagem | Média | `max_length` no schema Pydantic |
| 3 | `CAMERA_SOURCE` sem validação | Média | `validate_camera_source()` com allowlist de protocolos |

---

# Parte 3 — Melhoria do Código Gerado com IA

### Melhoria 1 — Banco de dados duplicado

**O que fazia originalmente.** As funções `init_db`, `save_event` e `list_events` existiam dentro do `app.py` e eram chamadas diretamente nas rotas. O `event_repository.py` existia em paralelo, com uma versão diferente da mesma função `list_recent_events`, causando duplicação.

**Problema encontrado.** Dois arquivos acessavam o SQLite com lógicas distintas. Qualquer manutenção precisava ser feita em dois lugares, sem garantia de qual versão era chamada em cada parte do sistema.

**O que foi melhorado.** Todas as funções de banco foram removidas do `app.py`. O `event_repository.py` passou a ser o único ponto de acesso ao SQLite, com três funções bem definidas: `init_db()`, `save_event()` e `list_recent_events()`. O `app.py` apenas importa e chama.

**Por que a nova versão é melhor.** Qualquer mudança no banco (trocar SQLite por PostgreSQL, por exemplo) exige alteração em um único arquivo. O código fica previsível — sabe-se exatamente onde está a lógica de persistência.

---

### Melhoria 2 — Função `process_stream` com responsabilidade única

**O que fazia originalmente.** A função `process_stream` no `app.py` fazia tudo em sequência: capturava o frame, rodava o YOLO, desenhava as caixas, contava frames consecutivos, decidia o alerta, salvava a imagem e persistia no banco.

**Problema encontrado.** Uma função com sete responsabilidades é impossível de testar de forma isolada. Alterar a lógica de alerta obrigava mexer no mesmo bloco que controla a câmera. O `app.py`, que deveria ser só o ponto de entrada, concentrava lógica operacional complexa.

**O que foi melhorado.** A lógica de detecção foi extraída para `services/vision_service.py` em três funções:

- `run_detection(frame, model, confidence)` — executa o YOLO e desenha as caixas
- `update_state(found_labels)` — atualiza o contador de frames consecutivos
- `check_alerts(found_labels, best_conf, frame)` — decide o alerta e salva a imagem

O `app.py` passou a apenas chamar essas funções em sequência.

**Por que a nova versão é melhor.** Cada função tem responsabilidade clara e pode ser testada e modificada de forma independente. Adicionar suporte a uma nova câmera ou trocar o modelo YOLO não exige tocar no roteamento da API.

---

### Melhoria 3 — Erro técnico exposto ao usuário no `ollama_client.py`

**O que fazia originalmente.**

```python
except Exception as e:
    return f"Erro ao consultar o Ollama: {e}"
```

**Problema encontrado.** A exceção completa era devolvida ao usuário final, expondo informações internas (endereço do servidor Ollama, tipo de erro de rede, detalhes da infraestrutura). Além do risco de segurança, a mensagem era técnica demais para um usuário comum.

**O que foi melhorado.**

```python
except httpx.ConnectError:
    logger.error(f"[ollama] Falha de conexão com {OLLAMA_URL}")
    return "Não foi possível conectar ao agente. Verifique se o Ollama está ativo."

except httpx.TimeoutException:
    logger.error(f"[ollama] Timeout após {OLLAMA_TIMEOUT}s")
    return "O agente demorou para responder. Tente novamente em instantes."

except Exception as e:
    logger.error(f"[ollama] Erro inesperado: {e}")
    return "Ocorreu um erro interno. Tente novamente em instantes."
```

**Por que a nova versão é melhor.** O detalhe técnico vai para o log do servidor; o usuário recebe uma mensagem clara e amigável; a infraestrutura interna não é exposta; cada tipo de erro tem mensagem específica.

---

### Melhoria 4 — `print()` substituído por `logging`

**O que fazia originalmente.**

```python
print(f"[câmera] Erro ao abrir fonte: {source}")
print(f"[ALERTA] {label} detectado. Evidência salva em {filepath}")
```

**Problema encontrado.** O `print()` não registra timestamp, nível de severidade nem o módulo de origem. Em produção, não é possível filtrar apenas erros ou alertas, e os logs se perdem ao reiniciar o processo.

**O que foi melhorado.**

```python
import logging
logger = logging.getLogger(__name__)

logger.error(f"[câmera] Não foi possível abrir: {source}")
logger.info(f"[ALERTA] {label} detectado — salvo em {filepath}")
```

Com configuração centralizada no `app.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
```

**Por que a nova versão é melhor.** Cada linha de log tem timestamp, nível (INFO/WARNING/ERROR) e módulo de origem. É possível filtrar por severidade em produção e o formato é compatível com ferramentas como Grafana Loki e Datadog.

---

### Melhoria 5 — Probabilidade de chuva incorreta no scraper

**O que fazia originalmente.** O `weather_scraper.py` lia a probabilidade de chuva do índice fixo `[0]` do array horário retornado pela API:

```python
"rain_probability_pct": data["hourly"]["precipitation_probability"][0],
```

**Problema encontrado.** A Open-Meteo devolve `precipitation_probability` como um array de 24 valores — um por hora do dia. O índice `[0]` corresponde **sempre à meia-noite (00:00)**, não à hora atual. Na prática, o dashboard exibia a chance de chuva do início do dia. Em um teste às 20:30, por exemplo, a API trazia 55% para a hora corrente, mas o sistema mostrava 0% (valor da meia-noite) — um dado climático enganoso justamente na informação usada para avaliar risco operacional.

**O que foi melhorado.** O índice passou a ser localizado pela hora atual (a partir de `current_weather.time`), com fallback seguro:

```python
current_time = data["current_weather"]["time"]   # ex.: "2026-05-29T20:30"
current_hour = current_time[:13]                  # ex.: "2026-05-29T20"
hourly_times = data["hourly"]["time"]
hourly_rain  = data["hourly"]["precipitation_probability"]
rain_index = next(
    (i for i, t in enumerate(hourly_times) if t[:13] == current_hour),
    0,  # fallback para meia-noite se não encontrar a hora
)
result["rain_probability_pct"] = hourly_rain[rain_index]
```

**Por que a nova versão é melhor.** O valor exibido passa a refletir a hora atual, que é o que importa para a decisão operacional. É um caso típico do papel do desenvolvedor: o código gerado "funcionava" (não dava erro), mas estava semanticamente errado — só uma revisão atenta detecta esse tipo de bug.

---

# Parte 4 — Camada de Web Scraping

### Dado coletado

Previsão do tempo via API pública **Open-Meteo** (`https://api.open-meteo.com`), sem necessidade de cadastro ou chave de API.

### Dados retornados

- Temperatura atual (°C)
- Velocidade do vento (km/h)
- Probabilidade de chuva (%) **da hora atual**
- Condição do céu (céu limpo, nublado, tempestade, etc.)
- Se é dia ou noite no momento da consulta

### Por que este dado melhora o projeto

O AgroVision monitora movimentação em ambiente rural por câmera, e as condições climáticas afetam diretamente o risco operacional de cada detecção:

- Um veículo detectado à noite com 80% de probabilidade de chuva representa risco diferente de uma detecção em dia claro.
- O agente de IA recebe o contexto dos eventos e pode cruzá-lo com as condições climáticas para respostas mais precisas ao operador.
- Em ambiente agrícola, saber se há tempestade ou ventos fortes ajuda a distinguir operação normal de situação de risco.

### Implementação técnica

Serviço isolado em `services/weather_scraper.py`, atendendo a todos os requisitos:

| Requisito | Como foi atendido |
|---|---|
| Serviço separado | Arquivo exclusivo `weather_scraper.py` |
| Fonte pública e gratuita | Open-Meteo, sem chave de API |
| Tratamento de erro | `try/except` para HTTP, timeout, conexão e campo ausente |
| Limite de requisições | Cache local de 10 minutos em memória |
| Dados estruturados | Retorna dicionário JSON com campos nomeados |
| Integração com o sistema | Rota `/weather` na API + card no dashboard |
| Finalidade clara | Contexto climático para análise de risco operacional |

### Cache implementado

```python
CACHE_TTL_SECONDS = 600  # 10 minutos
_cache: dict = {"data": None, "fetched_at": 0.0}

if _cache["data"] and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
    return _cache["data"]  # retorna cache sem nova requisição
```

Se a fonte externa estiver fora do ar, o sistema retorna o último dado disponível no cache (mesmo expirado) em vez de falhar completamente.

### Integração com o dashboard

Os dados são exibidos em um card dedicado no `index.html`, atualizado automaticamente a cada 10 minutos via JavaScript, sem recarregar a página.

### Oportunidade de evolução

Hoje o clima é exibido no dashboard, mas **ainda não é injetado no contexto do agente** (`monitoring_agent.py` usa apenas os eventos). Um próximo passo natural é incluir o resumo climático nas mensagens enviadas ao Ollama, tornando fundamentadas as perguntas do tipo "qual o risco considerando a chuva?".

---

*Projeto: AgroVision AI — Relatório Técnico (Partes 1 a 4)*
