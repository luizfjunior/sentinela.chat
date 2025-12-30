# Sentinela — README

> **Autor:** Brandon Franco Ferreira  
> **Stack:** FastAPI + Uvicorn + Planner (LLM) + Ferramentas de dados (PostgreSQL) + Frontend React/Vite  
> **Arquivos principais:** `server.py` (launcher/CLI), `agent_app.py` (aplicação FastAPI) e `frontend/` (React)

---

## 1) Visão geral

O **Sentinela** é uma API de assistente de dados que:

- expõe endpoints HTTP (FastAPI) para **saúde/diagnóstico**, **chat/planejamento**, e **ferramentas de dados** (listar CSVs, pré-visualizar conteúdo, ingerir em SQLite, etc.);
- faz **roteamento de modelos** (um PRIMÁRIO e, opcionalmente, um **FREE**/barato para tarefas simples/curtas) via um **planner**;
- mantém um **catálogo** de tabelas/colunas/tags após ingestão para facilitar consultas e sugestões.

Separação por arquivos:

- `server.py` — inicia a aplicação, lida com variáveis de ambiente e CLI, imprime URLs úteis e sobe o Uvicorn apontando para `agent_app:app`.
- `agent_app.py` — define `FastAPI`, rotas de status/diagnóstico, rotas de ferramentas de dados, endpoints de chat/planner, catálogo, UI do chat, etc.

---

## 2) Arquitetura

```
┌───────────────────────────┐
│          client           │
│  (curl/HTTP/Swagger/UI)   │
└─────────────┬─────────────┘
              │ HTTP
      ┌───────▼─────────────────────────────────────────┐
      │                   FastAPI                       │
      │                 (agent_app.py)                  │
      │  - /status, /ping, /stats, /free_health         │
      │  - /tool/* (CSV → SQLite)                       │
      │  - /upload_csv                                  │
      │  - (chat/planner/outside)                       │
      └───────┬─────────────────────────────────────────┘
              │ chama
      ┌───────▼─────────────────────────────────────────┐
      │                Planner / Router                 │
      │  - Infere intenção do pedido                    │
      │  - Decide usar PRIMARY_MODEL ou FREE_MODEL_ID   │
      │  - Prefixos: !primary / !free                   │
      └───────┬─────────────────────────────────────────┘
              │ usa
      ┌───────▼───────────────────────┐
      │  Ferramentas de Dados         │
      │  - CSV preview, ingestão      │
      │  - SQLite: head/filter/agg    │
      │  - sql_count, sku_intersection│
      │  - Catálogo (tabelas/colunas) │
      └───────────────────────────────┘
```

---

## 3) Como executar

### Opção 1: Script completo (Backend + Frontend)

```bash
# Windows - clique duplo ou execute:
iniciar_completo.bat
```

### Opção 2: Manual (separado)

**Backend (FastAPI):**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8000 --reload
```

**Frontend (React/Vite):**
```bash
cd frontend
npm install
npm run dev
```

### Configurar variáveis de ambiente do Frontend

Copie `frontend/.env.example` para `frontend/.env` e configure:
```env
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=sua_url_supabase
VITE_SUPABASE_ANON_KEY=sua_chave_anonima
```

**URLs úteis**  
- Frontend (React): `http://localhost:5173/`
- Backend API: `http://localhost:8000/`  
- Swagger: `http://localhost:8000/docs`

---

## 4) Variáveis de ambiente

Obrigatória:

- `OPENAI_API_KEY` — usada para o modelo primário.

O launcher pode perguntar e definir modelos/chaves em tempo de execução (`--ask-models`). Variáveis suportadas:

- `PRIMARY_MODEL` (padrão: `gpt-4o-mini`)
- `FREE_MODEL_ID` (ex.: `llama-3.1-8b-instant`)
- `FREE_API_BASE` (ex.: `https://api.groq.com/openai/v1`)
- `FREE_API_KEY` (opcional)

Diretórios/banco:

- `DATA_DIR` (default: `data/`)
- `DB_PATH`  (default: `data/sentinela.db`)

**Novas flags úteis**

- `SEMANTIC_FALLBACK` = `on|off` (padrão: on) → habilita o modo **outside** para perguntas conceituais/ambíguas.
- `OUT_OF_TOOL_MAXTOKENS` (padrão: 1400) → limite de tokens do outside (respostas longas).
- `ACTIVE_WINDOW_SECONDS` (padrão: 300) → janela para contagem de sessões ativas em `/stats`.
- `OUTSIDE_AUTOEXEC` = `1|0` (padrão: 1) → permite **execução automática** de comandos sugeridos pelo outside (ver §10).

---

## 5) CLI do `server.py`

O launcher prepara o ambiente, imprime os links úteis (UI/Swagger) e sobe o Uvicorn, com opção de perguntar os modelos (`--ask-models`).

---

## 6) Endpoints principais

### 6.1 Saúde/Diagnóstico

- `GET /status` → `{ "status": "up" }`
- `GET /ping`   → `{ "ok": true }`
- `GET /stats`  → métricas em tempo real (requisições ativas, sessões nos últimos N segundos).
- `GET /free_health` → valida o provider FREE (`/models` + uma `chat/completions` curta).

### 6.2 CSV/SQLite

- `GET  /tool/list_csvs` → lista CSVs em `data/`.
- `POST /tool/head_csv`  → preview (auto-detecção de separador).
- `POST /upload_csv`     → upload/ingestão (cria/atualiza tabela no SQLite + índices úteis).
- `POST /tool/ingest_all` → percorre `data/` (recursivo opcional) e (re)ingere todos os CSVs, depois **rebuild_catalog**.

### 6.3 SQL (SQLite)

- `GET /tool/sql_head?table=…&n=…`
- `GET /tool/sql_filter?table=…&where=JSON&limit=…&order_by=…&order=…&offset=…`
- `GET /tool/sql_aggregate?table=…&by=…&value=…&op=sum|count|avg|min|max&top=…`
- `GET /tool/sql_count?table=…&where=JSON` (novo)
- `GET /tool/sku_intersection?...` (ajustes ∩ devoluções com mesmos filtros de loja e período; diagnósticos e amostras quando vazio)

> As ferramentas foram pensadas para **não** gerar SQL bruto via LLM: a LLM escolhe a ferramenta e **nunca “inventa números”** no plano; os dados vêm do SQLite/CSV.

---

## 7) Catálogo e mapeamento semântico

Ao ingerir CSVs, o sistema:

- detecta colunas;
- cria **tags semânticas** (ex.: `sku`, `store`, `date`, `amount`, `quantity`, `return`, `cancel`, `sales`);
- mantém um catálogo `{ tabela → { columns, col_tags, table_tags, tokens } }`.

Isso permite:

- **resolver tabela** a partir do texto (“cancelamento”, “devolução”, etc.);
- **descobrir colunas** de agrupamento/valor ao agregar;
- **inferir período** (datas no padrão PT-BR e ISO);
- **normalizar loja** (zeros à esquerda) e **datas**.

---

## 8) Chat/Planner/Router

### 8.1 Fast-paths determinísticos

- Atalhos para frases como “**mostre X com N linhas**” e “**filtre … em TABELA limite N**” são resolvidos **sem LLM** (rápidos e baratos).
- Também existe uma resposta de **help** e **smalltalk** mínima.

### 8.2 Planner + roteamento de modelos

- Heurística de intenção (regex) → escolhe **FREE** (tarefas simples: head/filter/aggregate curtas) ou **PRIMARY** (tarefas complexas).
- Prefixes no prompt para forçar: `!free …` ou `!primary …`.
- Se o planner retorna plano pouco confiável ou incompleto, desviamos para o **outside** (abaixo).

### 8.3 Modo **outside** (explicativo, sem ferramentas)

Para perguntas **conceituais** ou quando não há parâmetros suficientes, o outside:

- escreve **explicação** clara de *como* obter o dado de forma segura;
- **não inventa números**;
- pode **sugerir um comando executável** (linguagem natural) dentro de um bloco `[[RUN]] … [[/RUN]]`, por exemplo:

```
[[RUN]]
agregue em devolucao por SKU somando VALOR top 20
[[/RUN]]
```

Com `OUTSIDE_AUTOEXEC=1`, o **/chat** e **/chat_html**:

1) mostram a explicação em Markdown;  
2) extraem o **comando sugerido**;  
3) tentam **executá-lo automaticamente** via fast-path (ou exibem “Comando sugerido não reconhecido para execução automática” se não for claro).

> A UI mostra um *badge* com meta-informações (rota/fornecedor/modelo) para depurar rapidamente qual caminho foi tomado.

---

## 9) UI Web (/) — Chat leve embutido

- HTML/CSS/JS simples integrado ao app.
- **Sticky header** da tabela, **zebra**, overflow horizontal, e *badge* de métricas ao vivo.
- Cada request envia um `X-Client-Id` persistido no `localStorage` para contagem de sessões em `/stats`.
- A UI de boas-vindas orienta a usar `/upload_csv` e dá exemplos de prompts.

---

## 10) Ferramentas SQL (detalhes e dicas)

### 10.1 `sql_filter`

- **Operadores**: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `contains`, `icontains`, `between`, `date_between`.
- **Normalização numérica**: strings numéricas são comparadas ignorando zeros à esquerda (`"00017"` ≡ `"17"`).
- `icontains` faz `lower(col) LIKE %valor%`.
- `date_between` aceita datas ISO (`YYYY-MM-DD`) ou PT-BR (com conversão automática).

### 10.2 `sql_aggregate`

- Campos: `table, by, value?, op=sum|count|avg|min|max, top, date_col?, start?, end?`.
- Se `op != count`, `value` precisa existir e é convertido para `REAL` internamente.
- Quando `start/end` são fornecidos, o sistema tenta descobrir `date_col` automaticamente e padroniza formatos de data.

### 10.3 `sql_count` (novo)

- Conta linhas com `where` arbitrário (mesmos operadores do `sql_filter`).
- Quando há um `COL=VAL` em `where`, a resposta gera uma **frase legível** do tipo “`COL = VAL` aparece **X** vezes em **TABELA** [loja/período se aplicável]”.

### 10.4 `sku_intersection`

- Encontra SKUs **presentes em duas tabelas** (default: “ajustes” e “devoluções”) com **mesmos filtros** de loja e período.
- Se a interseção vier vazia, traz **diagnóstico** (contagens por filtro) e **amostras** de cada lado para depuração.

---

## 11) Exemplos rápidos

Status e Swagger:

```bash
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/ping
# abrir no navegador
http://127.0.0.1:8000/docs
```

Prévia de CSV já em `data/`:

```bash
curl -X POST http://127.0.0.1:8000/tool/head_csv   -H "Content-Type: application/json"   -d '{"filename":"meu.csv","n":5}'
```

Upload + ingestão:

```bash
curl -F "file=@/caminho/arquivo.csv" -F "table=nome_tabela" http://127.0.0.1:8000/upload_csv
```

Checagem do provider free:

```bash
curl http://127.0.0.1:8000/free_health
```

Forçar modelo primário no chat:

```
!primary Faça um resumo das métricas do mês por categoria.
```

---

## 12) Boas práticas de prompts (no chat)

- “**mostre TABELA com N linhas**” → amostra rápida.
- “**filtre COL=VAL e COL2 in 123 456 em TABELA limite 50**” → filtros determinísticos.
- “**agregue em TABELA por COL somando VALOR top 10**” → ranking numérico.
- Perguntas abertas (“quais usuários e produtos…?”) podem ir para **outside**; quando houver comando `[[RUN]]…[[/RUN]]`, a execução automática tentará disparar a ferramenta correta.

---

## 13) Erros comuns & soluções

- **`OPENAI_API_KEY ausente` no startup:** defina a variável antes de subir; o app aborta de propósito para deixar claro o requisito.
- **CSV com separador “estranho”**: usamos auto-detecção e *fallbacks* `; , \t |`.
- **Sem resultados no filtro**: a resposta traz um **diagnóstico por condição** (`isolated_counts`) indicando qual filtro está “zerando” a busca.
- **Outside sem comando autoexecutável**: a explicação aparece, mas com aviso “comando não reconhecido para execução automática”.

---

## 14) Estrutura de repositório

```
.
├── agent_app.py          # Backend FastAPI + Agente IA
├── server.py             # Launcher/CLI
├── requirements.txt      # Dependências Python
├── iniciar.bat           # Inicia apenas o backend
├── iniciar_completo.bat  # Inicia backend + frontend
├── .env                  # Variáveis de ambiente (backend)
├── frontend/             # Frontend React/Vite
│   ├── src/
│   │   ├── pages/        # Páginas (Chat, Dashboard, Auth, etc.)
│   │   ├── components/   # Componentes UI (Shadcn)
│   │   ├── contexts/     # AuthContext (Supabase)
│   │   └── hooks/        # Hooks customizados
│   ├── .env              # Variáveis de ambiente (frontend)
│   └── package.json
└── README.md
```

---

## 15) Changelog resumido (vs. doc anterior)

- **Novo**: `sql_count` com frase legível.
- **Novo**: `sku_intersection` com diagnóstico + amostras.
- **Novo**: `OUTSIDE_AUTOEXEC` + parser `[[RUN]]…[[/RUN]]` no **outside**.
- **Melhorias**:
  - `sql_filter`: `icontains` robusto, normalização de números com zeros à esquerda, `in` com misto string/número, `date_between` mais tolerante.
  - **Roteamento**: `!primary` / `!free`, heurística de intenção e fallback seguro.
  - **UI**: cabeçalho fixo, zebra, overflow horizontal e badge de métricas/sessões com `X-Client-Id`.
  - **/stats**: ativa contagem de sessões por janela (configurável).
  - **/free_health**: verifica provider FREE e realiza *probe* de chat.

---

## 16) FAQ

**Sem `OPENAI_API_KEY` dá erro?**  
Sim; o launcher orienta a definir a chave e pode perguntar interativamente (`--ask-models`).

**CSV com separador “estranho” falha?**  
A ingestão tenta detectar; se não conseguir, a resposta explicará o motivo.

**Posso forçar o modelo FREE?**  
Sim — prefixe o prompt com `!free`.

---

## 17) Licença

(Defina aqui a licença do projeto, p.ex. MIT/Apache-2.0.)
