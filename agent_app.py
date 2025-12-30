# agent_app.py
# Sentinela – Chat + Tools (PostgreSQL Data Mart) com planner, roteamento e UI
# Requisitos principais: fastapi, uvicorn, agno, pydantic, pandas, numpy, markdown, starlette, psycopg

import os, traceback
import sys

from typing import Any, Dict, Optional, Literal
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env (override=True força o .env a sobrescrever variáveis de ambiente)
load_dotenv(override=True)
import numpy as np
import pandas as pd
import re, unicodedata
import json
import requests
import calendar
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from starlette.middleware.base import BaseHTTPMiddleware

from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, ValidationError
import html

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
import psycopg
from psycopg.rows import dict_row
import markdown

# =========================
# App & rotas básicas
# =========================
app = FastAPI(title="Sentinela")

# CORS - permite requisições do frontend React (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _missing_key_banner() -> str:
    return (
        "\n"
        "============================================================\n"
        "  ❌ OPENAI_API_KEY não definida\n"
        "  Defina a variável no terminal OU use o launcher recomendado:\n\n"
        "    Windows PowerShell:\n"
        '      $env:OPENAI_API_KEY = "SUA_CHAVE_AQUI"\n'
        "      python server.py --host 0.0.0.0 --port 8000 --reload\n\n"
        "    Linux/macOS:\n"
        '      export OPENAI_API_KEY=\"SUA_CHAVE_AQUI\"\n'
        "      python server.py --host 0.0.0.0 --port 8000 --reload\n\n"
        "  Links após iniciar:\n"
        "    • UI (Chat):   http://<host>:<port>/\n"
        "    • Swagger:     http://<host>:<port>/docs\n"
        "============================================================\n"
    )

@app.get("/status")
def status():
    return {"status": "up"}

@app.get("/ping")
def ping():
    return {"ok": True}

@asynccontextmanager
async def app_lifespan(app):
    # STARTUP
    if not os.environ.get("OPENAI_API_KEY"):
        print(_missing_key_banner(), flush=True)
        raise RuntimeError("OPENAI_API_KEY ausente")

    # Verifica conexão PostgreSQL
    try:
        print("[*] Conectando ao PostgreSQL Data Mart...", flush=True)
        print(f"[*] Host: {DB_HOST}:{DB_PORT} | Database: {DB_NAME}", flush=True)
        conn = _get_pg_connection()
        conn.close()
        print("[OK] Conexao com PostgreSQL estabelecida!", flush=True)
    except Exception as e:
        print(f"[ERRO] Erro ao conectar ao PostgreSQL: {e}", flush=True)
        print("[AVISO] Verifique as variaveis DB_* no arquivo .env", flush=True)
        raise RuntimeError(f"Falha na conexao com PostgreSQL: {e}")

    # Constrói catálogo de tabelas
    try:
        print("[*] Construindo catalogo de tabelas...", flush=True)
        rebuild_catalog()
        tables = list(CATALOG.keys())
        print(f"[OK] Catalogo construido! Tabelas disponiveis: {', '.join(tables)}", flush=True)
    except Exception as e:
        print(f"[AVISO] Erro ao construir catalogo: {e}", flush=True)

    yield

app.router.lifespan_context = app_lifespan

ACTIVE_REQUESTS = 0                   # requisições sendo atendidas agora
SESSIONS_LAST_SEEN = {}               # sid -> epoch seconds
METRICS_STARTED_AT = time.time()
ACTIVE_WINDOW_SECONDS = int(os.environ.get("ACTIVE_WINDOW_SECONDS", "300"))  # 5 min

def _extract_client_id(request):
    sid = request.headers.get("x-client-id")
    if not sid:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            sid = fwd.split(",")[0].strip()
        else:
            sid = (request.client.host if request.client else "unknown")
    return sid

class TrafficMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        global ACTIVE_REQUESTS
        sid = _extract_client_id(request)
        now = time.time()
        SESSIONS_LAST_SEEN[sid] = now
        ACTIVE_REQUESTS += 1
        try:
            response = await call_next(request)
            return response
        finally:
            ACTIVE_REQUESTS -= 1

app.add_middleware(TrafficMiddleware)

@app.get("/stats")
def stats():
    now = time.time()
    active_window = sum(1 for t in SESSIONS_LAST_SEEN.values() if now - t <= ACTIVE_WINDOW_SECONDS)
    return {
        "started_at": METRICS_STARTED_AT,
        "active_requests_now": ACTIVE_REQUESTS,
        "active_sessions_last_minutes": active_window,
        "window_seconds": ACTIVE_WINDOW_SECONDS,
        "unique_sessions_total": len(SESSIONS_LAST_SEEN),
    }

@app.get("/free_health")
def free_health():
    base = FREE_API_BASE or ""
    key  = FREE_API_KEY or ""
    mid  = FREE_MODEL_ID or ""
    if not (base and key and mid):
        return {"ok": False, "reason": "envs ausentes", "FREE_API_BASE": base, "FREE_MODEL_ID": mid, "has_key": bool(key)}

    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    try:
        r_models = requests.get(f"{base}/models", headers=h, timeout=10)
        models_ok = (r_models.status_code, r_models.text[:200])

        r_chat = requests.post(f"{base}/chat/completions", headers=h, json={
            "model": mid,
            "messages": [{"role":"user","content":"Responda 'pong'."}],
            "max_tokens": 5
        }, timeout=15)
        chat_ok = (r_chat.status_code, r_chat.text[:200])

        return {"ok": True, "models": models_ok, "chat": chat_ok}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# =========================
# Constantes e Configuração do Banco
# =========================
DATA_DIR = os.environ.get("DATA_DIR", "data")
DB_PATH  = os.path.join(DATA_DIR, "sentinela.db")  # Mantido para compatibilidade, mas não usado

# PostgreSQL Configuration
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "estoque")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_SCHEMA = os.environ.get("DB_SCHEMA", "sentinela")

# Connection String (usa variável ou constrói)
POSTGRES_CONNECTION_STRING = os.environ.get(
    "POSTGRES_CONNECTION_STRING",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

USE_PLANNER = True
OUTSIDE_AUTOEXEC = os.environ.get("OUTSIDE_AUTOEXEC", "on").lower() in {"on","1","true","yes"}
RUN_OPEN = "[[RUN]]"
RUN_CLOSE = "[[/RUN]]"
# ===== Fallback fora de tools =====
SEMANTIC_FALLBACK = os.environ.get("SEMANTIC_FALLBACK", "on").lower() in {"on","1","true","yes"}
OUT_OF_TOOL_MAXTOKENS = int(os.environ.get("OUT_OF_TOOL_MAXTOKENS", "5000"))

# =========================
# PostgreSQL Connection Helper (psycopg3)
# =========================
def _get_pg_connection():
    """Cria uma conexão com o PostgreSQL Data Mart usando psycopg3"""
    if not DB_PASSWORD:
        raise RuntimeError(
            "Configuração do PostgreSQL incompleta no arquivo .env\n"
            "Configure: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD"
        )

    # Connection string no formato PostgreSQL
    conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Retry em caso de erro de DNS/rede temporário
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = psycopg.connect(conn_str, row_factory=dict_row)
            return conn
        except psycopg.OperationalError as e:
            if attempt < max_retries - 1:
                print(f"[AVISO] Tentativa {attempt + 1}/{max_retries} falhou: {e}")
                print(f"[*] Tentando novamente em 2 segundos...")
                time.sleep(2)
            else:
                print(f"[ERRO] Falha ao conectar apos {max_retries} tentativas")
                raise

def _pg_exec(q: str, params: list | tuple = ()):
    """Executa uma query no PostgreSQL e retorna lista de dicts"""
    conn = _get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(q, params)
            if cur.description:  # Se há resultados
                rows = cur.fetchall()
                return [dict(row) for row in rows]
            return []
    finally:
        conn.close()

def _pg_cols(table: str) -> list[str]:
    """Retorna lista de colunas de uma tabela PostgreSQL"""
    q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    rows = _pg_exec(q, (DB_SCHEMA, table,))
    return [r["column_name"] for r in rows]

def _table_exists(table: str) -> bool:
    """Verifica se uma tabela existe no PostgreSQL"""
    q = """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        ) as exists
    """
    rows = _pg_exec(q, (DB_SCHEMA, table,))
    return rows[0]["exists"] if rows else False

def _fq_table(table: str) -> str:
    """Retorna nome qualificado da tabela com schema (ex: sentinela.ajustes_estoque)"""
    return f'"{DB_SCHEMA}"."{table}"'

# ===== Model routing (planner barato + fallback) =====
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "gpt-4o-mini").strip()
FREE_MODEL_ID = os.environ.get("FREE_MODEL_ID", "").strip()  # ex.: "llama-3.1-8b-instant"
FREE_API_BASE = os.environ.get("FREE_API_BASE", "").strip()  # ex.: "https://api.groq.com/openai/v1"
FREE_API_KEY  = os.environ.get("FREE_API_KEY", "").strip()   # ex.: sua GROQ_API_KEY

MODEL_KW = dict(temperature=1.0)  # zero criatividade; queremos determinismo

import contextlib
import re as _route_re

# Heurística simples de intenção -> decide quando tentar o modelo "free"
INTENT_PATTERNS = [
    ("sql_head",        _route_re.compile(r"\b(cabeçalho|primeir[ao]s?\s+\d+|head|amostra|mostre)\b", _route_re.I)),
    ("sql_filter",      _route_re.compile(r"\b(filtr|onde|where|igual|maior|menor|contém|contains)\b", _route_re.I)),
    ("sql_aggregate",   _route_re.compile(r"\b(sum|soma|média|avg|count|agreg|top\s*\d+)\b", _route_re.I)),
    ("sku_intersection",_route_re.compile(r"\b(interse(c|ç)[aã]o|aparecem\s+em\s+ambas|ajuste.*devol|devol.*ajuste)\b", _route_re.I)),
]

def _intent_hint(msg: str) -> str:
    if _route_re.search(r"\bsku\b.*\bdevol|\bdevol\b.*\bsku", msg or "", _route_re.I):
        return "sku_intersection"
    for name, pat in INTENT_PATTERNS:
        if pat.search(msg or ""):
            return name
    return "generic"

def _make_model(model_id: str, provider: str = "primary"):
    from agno.models.openai import OpenAIChat
    if provider == "free" and FREE_API_BASE and FREE_API_KEY:
        return OpenAIChat(id=model_id, base_url=FREE_API_BASE, api_key=FREE_API_KEY, **MODEL_KW)
    return OpenAIChat(id=model_id, **MODEL_KW)

def _route_model(msg: str) -> tuple[str, str]:
    intent = _intent_hint(msg or "")
    if FREE_MODEL_ID and intent in {"sql_head","sql_filter","sql_aggregate","sku_intersection"} and len(msg or "") <= 700:
        return ("free", FREE_MODEL_ID)
    return ("primary", PRIMARY_MODEL)

_FORCE_RE = _route_re.compile(r'^\s*!(free|primary)\s+', _route_re.I)

def _extract_forced_provider(msg: str) -> tuple[Optional[str], str]:
    if not msg:
        return None, msg
    m = _FORCE_RE.match(msg)
    if not m:
        return None, msg
    prov = m.group(1).lower()
    clean = msg[m.end():]
    return prov, clean

def _needs_fallback(answer_text: str) -> bool:
    if not answer_text or len(answer_text.strip()) == 0:
        return True
    low = answer_text.lower()
    bad_markers = [
        "não reconheci a tabela",
        "tabela não encontrada",
        "coluna não existe",
        '"status": "error"',
        "status\":\"error",
        "não consigo acessar",
        "não tenho dado suficiente",
    ]
    return any(b in low for b in bad_markers)

def _run_with_retry(agent_obj, message: str, max_attempts: int = 2):
    last_err = None
    for _ in range(max_attempts):
        try:
            resp = agent_obj.run(message)
            # Extrai nomes das tools usadas
            tools_used = []
            if hasattr(resp, 'tools') and resp.tools:
                tools_used = [t.tool_name for t in resp.tools if hasattr(t, 'tool_name') and t.tool_name]
            return True, (resp.content or ""), None, tools_used
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if any(x in msg for x in ["rate limit", "429", "temporarily unavailable", "timeout", "service unavailable", "overloaded"]):
                time.sleep(0.4)
                continue
            break
    return False, "", last_err, []

def _coerce_int_any(v) -> Optional[int]:
    import re
    if v is None:
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, (list, tuple)) and len(v) > 0:
        return _coerce_int_any(v[0])
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        if m:
            return int(m.group(0))
    return None

_PT_MONTHS = {
    "janeiro":1, "fevereiro":2, "marco":3, "março":3, "abril":4, "maio":5, "junho":6,
    "julho":7, "agosto":8, "setembro":9, "outubro":10, "novembro":11, "dezembro":12
}

def _extract_date_range(text: str) -> tuple[Optional[str], Optional[str]]:
    import re
    if not text:
        return (None, None)

    iso = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if len(iso) >= 2:
        return iso[0], iso[1]

    dmy = re.findall(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    if len(dmy) >= 2:
        def dmy2iso(s):
            d,m,y = s.split("/")
            return f"{y}-{m}-{d}"
        return dmy2iso(dmy[0]), dmy2iso(dmy[1])

    m = re.search(
        r"(janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s*(?:a|até|-|e)\s*"
        r"(janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)"
        r".*?(\d{4})",
        text, re.I
    )
    if m:
        m1 = _PT_MONTHS[m.group(1).lower()]
        m2 = _PT_MONTHS[m.group(2).lower()]
        y  = int(m.group(3))
        if m1 > m2:
            m1, m2 = m2, m1
        start = f"{y}-{m1:02d}-01"
        last_day = calendar.monthrange(y, m2)[1]
        end   = f"{y}-{m2:02d}-{last_day:02d}"
        return start, end

    return (None, None)

def _extract_loja_from_text(text: str) -> Optional[int]:
    import re
    m = re.search(r"\b(?:loja|filial|store)\s*0*([0-9]{1,5})\b", text or "", re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b0*([0-9]{1,3})\b", text or "")
    if m2:
        return int(m2.group(1))
    return None

# =========================
# CSV (legacy) - funções + endpoints
# =========================
def _set_resp_headers(resp: Response | None, *, provider: str, model_id: str, path: str, fallback: bool=False, tools_used: list=None):
    if resp is None:
        return
    resp.headers["X-Model-Provider"] = provider or ""
    resp.headers["X-Model-Id"] = model_id or ""
    resp.headers["X-Responder-Path"] = path or ""
    if fallback:
        resp.headers["X-Model-Fallback"] = "primary"
    if tools_used:
        resp.headers["X-Tools-Used"] = ", ".join(tools_used)

def _set_router_debug_headers(resp: Response, *, msg: str, provider: str):
    try:
        intent = _intent_hint(msg)
    except Exception:
        intent = "error"
    resp.headers["X-Router-Intent"] = intent
    resp.headers["X-Free-Enabled"] = "1" if bool(FREE_MODEL_ID) else "0"
    resp.headers["X-Forced-Provider"] = provider or ""

# Funções de CSV desabilitadas - dados vêm direto do PostgreSQL
def head_csv_fn(filename: str, n: int = 5):
    return {"status": "error", "message": "Função desabilitada. Os dados vêm direto do PostgreSQL."}

def list_csvs_fn():
    return {"status": "error", "message": "Função desabilitada. Use list_tables para ver as tabelas disponíveis no PostgreSQL."}

def list_tables_fn():
    """Lista todas as tabelas disponíveis no PostgreSQL Data Mart"""
    try:
        q = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        rows = _pg_exec(q, (DB_SCHEMA,))
        tables = [r["table_name"] for r in rows]
        return {"status": "ok", "tables": tables, "count": len(tables)}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}

ALLOWED_TOOLS = {"sql_head", "sql_aggregate", "sku_intersection", "list_tables", "none", "sql_count", "sku_tre_table", "pandas_analysis"}
TOOL_ALIASES = {
    "head": "sql_head",
    "sample": "sql_head",
    "preview": "sql_head",
    "table_head": "sql_head",

    "filter": "sql_filter",
    "where": "sql_filter",
    "query": "sql_filter",

    "aggregate": "sql_aggregate",
    "groupby": "sql_aggregate",
    "group_by": "sql_aggregate",

    "sql_list_tables": "list_tables",
    "tables": "list_tables",
    "list_csvs": "list_tables",  # Redirecionamento para compatibilidade

    "sku_intersect": "sku_intersection",
    "sku_overlap": "sku_intersection",
    "count": "sql_count",
    "contar": "sql_count",
    "conte": "sql_count",
    "quantas_vezes": "sql_count",
    "ocorrencias": "sql_count",

    "cross_analysis": "sku_tre_table",
    "sku_cross": "sku_tre_table",
    "cross_sku": "sku_tre_table",
    "cruzar_sku": "sku_tre_table",
    "cruzamento": "sku_tre_table",
    "tree_table": "sku_tre_table",
    "arvore_sku": "sku_tre_table",

    "pandas": "pandas_analysis",
    "analise_avancada": "pandas_analysis",
    "python_analysis": "pandas_analysis",
    "advanced_analysis": "pandas_analysis",
}

class Plan(BaseModel):
    mode: Literal["tool", "answer", "delegate"] = "tool"
    tool: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    rewrite: Optional[str] = None
    confidence: float = 0.0

class HeadIn(BaseModel):
    filename: str
    n: Optional[int] = 5

@app.post("/tool/head_csv")
def call_head_csv(in_: HeadIn):
    return head_csv_fn(filename=in_.filename, n=in_.n or 5)

def list_csvs_fn():
    os.makedirs(DATA_DIR, exist_ok=True)
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    return {"status": "ok", "files": files}

@app.get("/tool/list_csvs")
def call_list_csvs():
    return list_csvs_fn()

# =========================
# PostgreSQL helpers (antigo SQLite)
# =========================
def _unaccent_sql(expr: str) -> str:
    # remove alguns acentos comuns (minúsc/maiúsc). adicione mais se precisar.
    repl = [
        ("á","a"),("à","a"),("â","a"),("ã","a"),("ä","a"),
        ("Á","A"),("À","A"),("Â","A"),("Ã","A"),("Ä","A"),
        ("é","e"),("ê","e"),("É","E"),("Ê","E"),
        ("í","i"),("ï","i"),("Í","I"),("Ï","I"),
        ("ó","o"),("ô","o"),("õ","o"),("ö","o"),("Ó","O"),("Ô","O"),("Õ","O"),("Ö","O"),
        ("ú","u"),("ü","u"),("Ú","U"),("Ü","U"),
        ("ç","c"),("Ç","C"),
    ]
    for a,b in repl:
        expr = f"replace({expr}, '{a}', '{b}')"
    return expr

# Aliases para manter compatibilidade com código existente
_sqlite_exec = _pg_exec
_sqlite_cols = _pg_cols
# _table_exists já foi definida acima

def _sniff_sep(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            head = f.read(4096).decode("latin-1", "ignore")
        first = head.splitlines()[0] if head else ""
    except Exception:
        first = ""
    candidates = [";", ",", "\t", "|"]
    counts = {s: first.count(s) for s in candidates}
    sep = max(counts, key=counts.get) if any(counts.values()) else ";"
    return sep

# Funções de importação CSV desabilitadas - dados vêm direto do PostgreSQL
def _import_csv_to_sqlite(temp_csv_path: str, table: str) -> dict:
    return {"status": "error", "message": "Importação de CSV desabilitada. Dados vêm direto do PostgreSQL."}

def _import_csv_to_sqlite_OLD(temp_csv_path: str, table: str) -> dict:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.row_factory = sqlite3.Row

    first = True
    total = 0
    try:
        primary_sep = _sniff_sep(temp_csv_path)
        tried = set()

        def _read_chunks(sep_value):
            return pd.read_csv(
                temp_csv_path,
                sep=sep_value,
                engine="python",
                encoding="latin-1",
                on_bad_lines="skip",
                dtype=str,
                chunksize=50_000,
            )

        last_err = None
        for sep in [primary_sep, ";", ",", "\t", "|"]:

            if sep in tried:
                continue
            tried.add(sep)
            try:
                for chunk in _read_chunks(sep):
                    chunk.to_sql(table, con, if_exists="replace" if first else "append", index=False)
                    total += len(chunk)
                    first = False
                last_err = None
                break
            except Exception as e:
                last_err = e
                continue

        if last_err is not None and first:
            raise last_err

        cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
        for col in ("LOJA", "DATACANCELAMENTO"):
            if col in cols:
                con.execute(
                    f'CREATE INDEX IF NOT EXISTS idx_{table}_{col.lower()} ON "{table}" ("{col}");'
                )

        con.commit()
        return {"rows_imported": total, "columns": cols}
    finally:
        con.close()

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), table: str = Form(None)):
    """Endpoint desabilitado - dados vêm direto do PostgreSQL."""
    return {
        "status": "error",
        "message": "Upload de CSV desabilitado. Os dados são carregados diretamente do PostgreSQL."
    }

def _safe_table_name(stem: str) -> str:
    name = re.sub(r'[^A-Za-z0-9_]+', '_', stem).strip('_').lower()
    return name or "tabela"

def ingest_all_fn(recursive: bool = False):
    """Função desabilitada - dados vêm direto do PostgreSQL"""
    return {
        "status": "error",
        "message": "Ingestão de CSVs desabilitada. Os dados são carregados diretamente do PostgreSQL."
    }

def ingest_all_fn_OLD(recursive: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)
    imported = []

    files = []
    if recursive:
        for root, _, fns in os.walk(DATA_DIR):
            for fn in fns:
                if fn.lower().endswith(".csv"):
                    files.append(os.path.join(root, fn))
    else:
        files = [os.path.join(DATA_DIR, fn) for fn in os.listdir(DATA_DIR) if fn.lower().endswith(".csv")]

    for path in files:
        stem = pathlib.Path(path).stem
        table = _safe_table_name(stem)
        try:
            info = _import_csv_to_sqlite_OLD(path, table)
            imported.append({"file": os.path.relpath(path, DATA_DIR), "table": table, **info})
        except Exception as e:
            imported.append({"file": os.path.relpath(path, DATA_DIR), "table": table,
                             "error": f"{type(e).__name__}: {e}"})

    return {"status": "ok", "db": DB_PATH, "count": len(imported), "results": imported}

@app.post("/tool/ingest_all")
def call_ingest_all(recursive: bool = Query(False, description="Se true, percorre subpastas de data/")):
    res = ingest_all_fn(recursive=recursive)
    rebuild_catalog()
    return res

# --- normalização / tokenização leves ---
_PT_STOP = {"de","da","do","das","dos","e","em","no","na","nos","nas","com","por","para","a","o","os","as","por","entre"}

_SYNONYMS = {
    "estoque": {"estoque","stock"},
    "ajuste": {"ajuste","ajustes","ajust"},
    "devolucao": {"devolucao","devolucoes","devoluções","devol","troca","retorno","return"},
    "cancelamento": {"cancelamento","cancel","canc","cancelados","cancelar"},
    "venda": {"venda","vendas","sale","sales","faturamento","receita","revenue","billing"},
    "inventario": {"inventario","inventário","invent"},
    "saida": {"saida","saída","out"},
    "sku": {"sku","produto","prod","codigo_produto","codigo","id_sku","idproduto","codprod","cod_prod"},
    "loja": {"loja","filial","store","cod_loja","codfilial"},
    "cliente": {"cliente","consumidor","clienteid","idcliente"},
    "usuario": {"usuario","vendedor","idusuario","operador"},
    "categoria": {"categoria","grupo","subgrupo","departamento","segmento","familia"},
    "marca": {"marca","brand"},
    "data": {"data","dt","datacancelamento","data_devolucao","datadev","data_devolucoes","datavenda","dt_mov","dtmov","dtmovto"},
    "valor": {"valor","valorbruto","preco","preço","total","subtotal","receita","faturamento","amount"},
    "quantidade": {"qtd","qtde","quantidade","qte","qtdade"},
}

def _strip_accents(s:str)->str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def _norm_text(s:str)->str:
    return re.sub(r'[^a-z0-9]+',' ', _strip_accents(s.lower())).strip()

def _tokens(s:str)->set[str]:
    toks = {t for t in _norm_text(s).split() if t and t not in _PT_STOP}
    expanded = set(toks)
    for base, syns in _SYNONYMS.items():
        if base in toks or (toks & syns):
            expanded |= syns | {base}
    return expanded

def _tag_of_column(col: str)->set[str]:
    c = _norm_text(col)
    tags = set()
    if re.search(r'\b(loja|filial|store|cod[_]?loja|codfilial)\b', c): tags.add("store")
    if re.search(r'\b(sku|produto|prod|codigo|codigo[_]?produto|id[_]?sku|idproduto|codprod|cod[_]?prod)\b', c): tags.add("sku")
    if re.search(r'\b(data|dt|datacancelamento|data[_]?dev|datavenda|dt[_]?mov|dtmov|dtmovto)\b', c): tags.add("date")
    if re.search(r'\b(valor|preco|preço|total|subtotal|bruto|receita|faturamento|amount)\b', c): tags.add("amount")
    if re.search(r'\b(qtd|qtde|quant|quantidade|movimentada|ajuste)\b', c): tags.add("quantity")
    if re.search(r'\b(cliente|consumidor)\b', c): tags.add("customer")
    if re.search(r'\b(usuario|vendedor|operador)\b', c): tags.add("user")
    if re.search(r'\b(categoria|grupo|subgrupo|departamento|segmento|familia)\b', c): tags.add("category")
    if re.search(r'\b(marca|brand)\b', c): tags.add("brand")
    if re.search(r'\b(ajuste|estoque|stock)\b', c): tags.add("adjust")
    if re.search(r'\b(devol)\b', c): tags.add("return")
    if re.search(r'\b(cancel)\b', c): tags.add("cancel")
    if re.search(r'\b(venda|sales|faturamento|receita|revenue)\b', c): tags.add("sales")
    return tags

def _tags_from_tokens(tokens: set[str]) -> set[str]:
    tags = set()
    if tokens & _SYNONYMS.get("ajuste", set()):        tags.add("adjust")
    if tokens & _SYNONYMS.get("estoque", set()):       tags.add("adjust")
    if tokens & _SYNONYMS.get("devolucao", set()):     tags.add("return")
    if tokens & _SYNONYMS.get("cancelamento", set()):  tags.add("cancel")
    if tokens & _SYNONYMS.get("venda", set()):         tags.add("sales")
    for k, t in [("sku","sku"), ("data","date"), ("loja","store"),
                 ("valor","amount"), ("quantidade","quantity"),
                 ("cliente","customer"), ("usuario","user"),
                 ("categoria","category"), ("marca","brand")]:
        if tokens & _SYNONYMS.get(k, set()): tags.add(t)
    return tags

# catálogo global (evita NameError se usado antes de rebuild)
CATALOG: dict = {}

def rebuild_catalog():
    """Reconstrói o catálogo de tabelas do PostgreSQL Data Mart"""
    global CATALOG
    CATALOG = {}

    # Lista tabelas do PostgreSQL
    q = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    tabs = _pg_exec(q, (DB_SCHEMA,))

    for r in tabs:
        tname = r["table_name"]
        cols = _pg_cols(tname)
        t_tokens = _tokens(tname)
        col_tags = {c: _tag_of_column(c) for c in cols}
        name_tags = _tags_from_tokens(t_tokens)
        t_tags = (set().union(*col_tags.values()) if col_tags else set()) | name_tags
        CATALOG[tname] = {"tokens": t_tokens, "columns": cols, "col_tags": col_tags, "table_tags": t_tags}

def _first_col_with_tag(table: str, tag: str) -> str | None:
    if not CATALOG or table not in CATALOG:
        rebuild_catalog()
    info = CATALOG.get(table, {})
    for c, tags in (info.get("col_tags") or {}).items():
        if tag in tags:
            return c
    return None

def _key_to_col(table: str, k: str, meta: dict | None = None) -> str | None:
    kk = (k or "").strip().lower()
    meta = meta or _guess_cols(table)

    if kk in {"sku"}:
        return meta.get("sku")
    if kk in {"loja", "store", "filial"}:
        return meta.get("loja")
    if kk in {"date", "data", "dt"}:
        return meta.get("data")

    if kk in {"valor", "amount", "preco", "preco_venda", "valorbruto", "price"}:
        c = _first_col_with_tag(table, "amount")
        if c: return c
    if kk in {"qtd", "qtde", "quantidade", "quantity"}:
        c = _first_col_with_tag(table, "quantity")
        if c: return c

    cols = set(_pg_cols(table))
    if k in cols:
        return k
    m = {c.lower(): c for c in cols}
    return m.get(kk)

def list_tables() -> list[str]:
    return list(CATALOG.keys())

def resolve_table_from_text(msg: str, required_tags: set[str]|None=None) -> str|None:
    if not CATALOG:
        rebuild_catalog()
    qtok = _tokens(msg)
    best, best_score = None, -1
    qnorm = _norm_text(msg)
    for tname, info in CATALOG.items():
        base = len(qtok & info["tokens"])
        tag_bonus = 0
        if required_tags:
            tag_bonus = 2 * len(required_tags & info["table_tags"])
        name_hit = 1 if _norm_text(tname) in qnorm else 0
        score = base + tag_bonus + name_hit
        if score > best_score:
            best, best_score = tname, score
    return best if best_score >= 1 else None

def _extract_loja(text: str) -> int | None:
    m = re.search(r'\b(loja|filial|store)\s*[:=\-]?\s*0*([0-9]{1,4})\b', text, re.I)
    if not m:
        return None
    try:
        return int(m.group(2))
    except:
        return None

def _extract_time_range_pt(text: str, table_hint: str | None = None) -> tuple[str | None, str | None]:
    t = (text or "").lower().strip()

    yy = re.search(r"\b(19|20)\d{2}\b", t) or (re.search(r"(19|20)\d{2}", table_hint or "") if table_hint else None)
    base_year = int(yy.group(0)) if yy else datetime.now().year

    def _parse_date_any_local(s: str) -> str | None:
        s = s.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", s)
        if m:
            return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        return _parse_pt_date_words(s, base_year=base_year)

    m = re.search(r"entre\s+(.+?)\s+e\s+(.+)", t)
    if m:
        s = _parse_date_any_local(m.group(1)) or None
        e = _parse_date_any_local(m.group(2)) or None
        if s and e:
            return (s, e)

    months_long = {"janeiro":1,"fevereiro":2,"março":3,"marco":3,"abril":4,"maio":5,"junho":6,"julho":7,"agosto":8,"setembro":9,"outubro":10,"novembro":11,"dezembro":12}
    months_abbr = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}
    months_all  = {**months_long, **months_abbr}

    mm = re.search(r"\b(janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\b", t)
    if mm and yy:
        import calendar
        key = mm.group(1).replace("ç","c")
        mnum = months_all[key]
        y = int(yy.group(0))
        last = calendar.monthrange(y, mnum)[1]
        return (f"{y:04d}-{mnum:02d}-01", f"{y:04d}-{mnum:02d}-{last:02d}")

    mr = re.search(
        r"(?:de|entre)\s+(janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)"
        r"\s+(?:a|até|e|-)\s+"
        r"(janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)"
        r"(?:\s+(19|20)\d{2})?", t)
    if mr:
        import calendar
        m1 = months_all[mr.group(1).replace("ç","c")]
        m2 = months_all[mr.group(2).replace("ç","c")]
        y  = int(mr.group(3)) if mr.group(3) else base_year
        if m1 > m2: m1, m2 = m2, m1
        last = calendar.monthrange(y, m2)[1]
        return (f"{y:04d}-{m1:02d}-01", f"{y:04d}-{m2:02d}-{last:02d}")

    mmm = re.search(r"\b(19|20)\d{2}-(\d{2})\b\s+(?:a|até|-)\s+\b(19|20)\d{2}-(\d{2})\b", t)
    if mmm:
        import calendar
        y1,m1,y2,m2 = map(int, mmm.groups())
        last = calendar.monthrange(y2, m2)[1]
        return (f"{y1:04d}-{m1:02d}-01", f"{y2:04d}-{m2:02d}-{last:02d}")

    mr2 = re.search(
        r"(\d{1,2}\s*(?:de\s+)?(?:janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez))"
        r"\s+(?:a|até|e|-)\s+"
        r"(\d{1,2}\s*(?:de\s+)?(?:janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez))"
        r"(?:\s+(19|20)\d{2})?", t
    )
    if mr2:
        s = _parse_pt_date_words(mr2.group(1), base_year=base_year)
        e = _parse_pt_date_words(mr2.group(2), base_year=int(mr2.group(3)) if mr2.group(3) else base_year)
        if s and e:
            return (s, e)

    return (None, None)

def _parse_pt_date_words(s: str, base_year: int | None = None) -> str | None:
    import re
    from datetime import datetime

    months_long = {"janeiro":1,"fevereiro":2,"março":3,"marco":3,"abril":4,"maio":5,"junho":6,"julho":7,"agosto":8,"setembro":9,"outubro":10,"novembro":11,"dezembro":12}
    months_abbr = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}
    months_all  = {**months_long, **months_abbr}

    t = (s or "").lower().strip()
    m = re.search(
        r"\b(\d{1,2})\s*(?:de\s+)?(janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\b"
        r"(?:\s*(?:de)?\s*((?:19|20)\d{2}))?", t
    )
    if not m:
        return None

    d = int(m.group(1))
    mon_name = m.group(2).replace("ç","c")
    y = int(m.group(3)) if m.group(3) else (base_year or datetime.now().year)
    mon = months_all.get(mon_name)
    if not mon:
        return None

    last = calendar.monthrange(y, mon)[1]
    d = max(1, min(d, last))
    return f"{y:04d}-{mon:02d}-{d:02d}"

def _infer_table(user_msg: str, prefer_tags: set[str] | None = None) -> str | None:
    toks = _tokens(user_msg)
    req = set()
    if toks & _SYNONYMS["cancelamento"]: req.add("cancel")
    if toks & _SYNONYMS["devolucao"]:    req.add("return")
    if toks & _SYNONYMS["venda"]:        req.add("sales")
    if prefer_tags: req |= prefer_tags
    return resolve_table_from_text(user_msg, required_tags=req or None)

def _infer_by(table: str, toks: set[str]) -> str | None:
    pref = []
    if toks & _SYNONYMS["sku"]:       pref.append("sku")
    if toks & _SYNONYMS["loja"]:      pref.append("store")
    if toks & _SYNONYMS["cliente"]:   pref.append("customer")
    if toks & _SYNONYMS["usuario"]:   pref.append("user")
    if toks & _SYNONYMS["categoria"]: pref.append("category")
    if toks & _SYNONYMS["marca"]:     pref.append("brand")
    for tag in pref:
        col = _first_col_with_tag(table, tag)
        if col: return col
    cols = _pg_cols(table)
    info = CATALOG.get(table, {})
    bad = {"date","amount"}
    for c in cols:
        if tag := (info.get("col_tags", {}).get(c) or set()):
            if not (tag & bad): return c
    return cols[0] if cols else None

def _infer_op_and_value(table: str, toks: set[str]) -> tuple[str, str | None]:
    if re.search(r"\b(m[eé]dia|avg)\b", " ".join(toks), re.I):
        val = _first_col_with_tag(table, "amount") or _first_col_with_tag(table, "quantity")
        return ("avg", val)
    if (toks & _SYNONYMS["valor"]) or (toks & _SYNONYMS["venda"]):
        val = _first_col_with_tag(table, "amount")
        if val: return ("sum", val)
    if (toks & _SYNONYMS["quantidade"]):
        val = _first_col_with_tag(table, "quantity")
        if val: return ("sum", val)
    return ("count", None)

def _infer_top(user_msg: str, default: int = 10) -> int:
    m = re.search(r"\btop\s*(\d{1,3})\b", user_msg, re.I) or re.search(r"\b(\d{1,3})\b", user_msg)
    try:
        n = int(m.group(1)) if m else default
        return max(1, min(1000, n))
    except:
        return default

def _parse_date_any(s: str) -> str | None:
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): return s
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m: return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None

def _mini_schema_for_prompt(user_msg: str, max_tables: int = 5, max_cols: int = 12) -> dict:
    if not CATALOG:
        rebuild_catalog()
    toks = _tokens(user_msg)
    scored = []
    for tname, info in CATALOG.items():
        score = len(toks & info["tokens"]) + len(toks & info["table_tags"])
        scored.append((score, tname, info))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = []
    for _, tname, info in scored[:max_tables]:
        cols = info["columns"][:max_cols]
        top.append({"table": tname, "columns": cols, "tags": sorted(info["table_tags"])})
    return {"tables": top}

def _planner_prompt(user_msg: str) -> str:
    schema = _mini_schema_for_prompt(user_msg)
    return (
        "Você é um PLANEJADOR. Retorne SOMENTE JSON válido com este formato:\n"
        "{"
        "  \"mode\": \"tool|answer|delegate\","
        "  \"tool\": \"sql_head|sql_filter|sql_aggregate|sku_intersection|sql_count|list_csvs|pandas_analysis|sku_tre_table|null\","

        "  \"params\": { ... },"
        "  \"rewrite\": \"mensagem reescrita para o agente responder sem consultar DB\","
        "  \"confidence\": 0.0"
        "}\n\n"
        "REGRA CRÍTICA - CRUZAMENTO DE 3 TABELAS (Ajuste + Inventário + Troca):\n"
        "- Quando o usuário pedir SKUs que aparecem em ajuste, inventário E troca → use tool 'sku_tre_table'\n"
        "- Extraia: loja, data_ini (YYYY-MM-DD), data_fim (YYYY-MM-DD)\n"
        "- Converta meses PT-BR: janeiro=01, fevereiro=02, ..., dezembro=12\n"
        "- params para sku_tre_table: {\"p_loja\": \"002\", \"p_data_ini\": \"2025-01-01\", \"p_data_fim\": \"2025-06-30\", \"p_rule\": \"any\"}\n\n"
        "EXEMPLOS de sku_tre_table:\n"
        "- 'SKUs da loja 111 em abril de 2025 com ajuste, inventário e troca' → {\"p_loja\": \"111\", \"p_data_ini\": \"2025-04-01\", \"p_data_fim\": \"2025-04-30\"}\n"
        "- 'loja 002 entre janeiro e junho de 2025' → {\"p_loja\": \"002\", \"p_data_ini\": \"2025-01-01\", \"p_data_fim\": \"2025-06-30\"}\n\n"
        "Regras gerais:\n"
        "- Use mode \"tool\" quando o usuário quer números, contagens, filtros, amostras, agregações ou cruzar tabelas.\n"
        "- Use mode \"answer\" para perguntas conceituais/explicativas. NÃO invente números.\n"
        "- Use mode \"delegate\" se estiver incerto.\n"
        "- Não invente nomes de tabela; escolha entre as do catálogo.\n\n"
        f"Catálogo enxuto: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Pedido do usuário: {user_msg}\n"
    )

def planner_route(user_msg: str) -> tuple[str, Optional[str]]:
    try:
        provider, model_id = _route_model(user_msg)
        planner_model = _make_model(model_id, provider)

        planner = Agent(
            name="planner",
            system_message="Retorne APENAS JSON válido, sem texto extra.",
            model=planner_model,
        )
        prompt = _planner_prompt(user_msg)

        ok, out, _err, _tools = _run_with_retry(planner, prompt)
        if (not ok) or (not out or not out.strip()):
            if provider != "primary":
                planner = Agent(
                    name="planner",
                    system_message="Retorne APENAS JSON válido, sem texto extra.",
                    model=_make_model(PRIMARY_MODEL, "primary"),
                )
                ok, out, _err, _tools = _run_with_retry(planner, prompt)
            if (not ok) or (not out or not out.strip()):
                return ("fail", None)

        try:
            plan = Plan.model_validate_json(out)
        except Exception:
            m = re.search(r"\{.*\}", out or "", re.S)
            if not m:
                return ("fail", None)
            try:
                plan = Plan.model_validate_json(m.group(0))
            except Exception:
                return ("fail", None)

        plan, bad_reason = _sanitize_plan(plan, user_msg)

        conf = plan.confidence if (plan.confidence is not None) else 0.0
        if conf < 0.6:
            return ("delegate_outside", None)

        if bad_reason in {"tool_desconhecida", "sem_parametros", "delegate"}:
            return ("delegate_outside", None)

        try:
            html = _execute_plan_to_html(plan)
            return ("tool", html)
        except Exception as e:
            print(f"[planner_route] execute_plan_to_html erro: {e}", flush=True)
            return ("delegate_outside", None)

    except Exception as e:
        print(f"[planner_route] erro inesperado: {e}", flush=True)
        return ("delegate_outside", None)

def _sanitize_plan(plan: Plan, user_msg: str) -> tuple[Plan, Optional[str]]:
    p = dict(plan.params or {})

    for k, v in list(p.items()):
        if isinstance(v, bool):
            p.pop(k, None)
    for k in ("limit", "top", "n"):
        if k in p:
            try:
                p[k] = max(1, min(1000, int(p[k])))
            except Exception:
                p.pop(k, None)

    reason = None
    tool = plan.tool

    if plan.tool in {"null"}:
        plan.tool = "none"

    if plan.mode != "tool":
        return Plan(mode=plan.mode, tool=plan.tool, params=p,
                    rewrite=plan.rewrite, confidence=plan.confidence), "delegate"
    if tool not in ALLOWED_TOOLS or tool == "none":
        return Plan(mode=plan.mode, tool=plan.tool, params=p,
                    rewrite=plan.rewrite, confidence=plan.confidence), "tool_desconhecida"

    toks = _tokens(user_msg)

    if tool == "sql_head":
        if not p.get("table"):
            t = _infer_table(user_msg, None)
            if t: p["table"] = t
        if "n" not in p:
            p["n"] = 5
        if not p.get("table"):
            return Plan(mode=plan.mode, tool=tool, params=p,
                        rewrite=plan.rewrite, confidence=plan.confidence), "sem_parametros"

    elif tool == "sql_filter":
        if not p.get("table"):
            t = _infer_table(user_msg, None)
            if t: p["table"] = t
        where = dict(p.get("where") or {})
        loja_num = _extract_loja_from_text(user_msg)
        if loja_num is not None and "loja__eq" not in where and "store__eq" not in where and "filial__eq" not in where:
            where["loja__eq"] = loja_num
        start, end = _extract_time_range_pt(user_msg, p.get("table"))
        if start and end and "date__date_between" not in where:
            where["date__date_between"] = [start, end]
        p["where"] = where
        if "limit" not in p:
            p["limit"] = 100
        if not p.get("table"):
            return Plan(mode=plan.mode, tool=tool, params=p,
                        rewrite=plan.rewrite, confidence=plan.confidence), "sem_parametros"

    elif tool == "sql_aggregate":
        if not p.get("table"):
            t = _infer_table(user_msg, None)
            if t: p["table"] = t
        table = p.get("table")

        toks = _tokens(user_msg)
        want_user = bool(toks & _SYNONYMS.get("usuario", set()))
        want_prod = bool(toks & _SYNONYMS.get("sku", set()) or re.search(r"\bprodut", user_msg, re.I))
        want_same_day = bool(re.search(r"\bmesm[oa]\s+dia\b", user_msg, re.I))

        # BY
        if not p.get("by") and table:
            meta = _guess_cols(table)
            by_candidates = []
            if want_user:
                u = _first_col_with_tag(table, "user")
                if u: by_candidates.append(u)
            if want_prod:
                s = _first_col_with_tag(table, "sku")
                if s: by_candidates.append(s)

            # caso "mesmo dia": SKU + DATA
            if want_same_day and meta.get("data"):
                if not by_candidates:
                    s = _first_col_with_tag(table, "sku")
                    if s: by_candidates.append(s)
                by_candidates.append(meta["data"])

            if by_candidates:
                p["by"] = by_candidates if len(by_candidates) > 1 else by_candidates[0]
            else:
                by = _infer_by(table, toks)
                if by: p["by"] = by

        # OP / VALUE (frequência vs volume)
        txt = user_msg.lower()
        if ("volume" in txt) and (table) and (not p.get("value")):
            v = _first_col_with_tag(table, "amount")
            if v:
                p["op"] = "sum"
                p["value"] = v

        if ("frequência" in txt or "frequencia" in txt):
            p["op"] = "count"
            p.pop("value", None)

        if not p.get("op") or (p.get("op") != "count" and not p.get("value")):
            if table:
                op, val = _infer_op_and_value(table, toks)
                p.setdefault("op", op)
                if val: p.setdefault("value", val)

        if "top" not in p:
            p["top"] = _infer_top(user_msg, default=10)

        start, end = _extract_time_range_pt(user_msg, table)
        if start and end:
            if "date_col" not in p and table:
                p["date_col"] = _guess_cols(table).get("data")
            p["start"], p["end"] = start, end

        if not table or not p.get("by"):
            return Plan(mode=plan.mode, tool=tool, params=p,
                        rewrite=plan.rewrite, confidence=plan.confidence), "sem_parametros"
        if p.get("op") not in {"sum","count","avg","min","max"}:
            p["op"] = "count"
        if p["op"] != "count" and not p.get("value"):
            p["op"] = "count"

    elif tool == "sku_intersection":
        if p.get("loja") is None:
            loja_num = _extract_loja_from_text(user_msg)
            if loja_num is not None:
                p["loja"] = loja_num
        if not p.get("start") or not p.get("end"):
            s, e = _extract_time_range_pt(user_msg, None)
            if s and e:
                p["start"], p["end"] = s, e
        if "limit" not in p:
            p["limit"] = 100
        if p.get("loja") is None or not p.get("start") or not p.get("end"):
            return Plan(mode=plan.mode, tool=tool, params=p,
                        rewrite=plan.rewrite, confidence=plan.confidence), "sem_parametros"

    elif tool == "sql_count":
        if not p.get("table"):
            t = _infer_table(user_msg, None)
            if t: p["table"] = t

        where = dict(p.get("where") or {})
        where.update(_extract_eq_pairs(user_msg))

        loja_num = _extract_loja_from_text(user_msg)
        if loja_num is not None and "loja__eq" not in where and "store__eq" not in where and "filial__eq" not in where:
            where["loja__eq"] = loja_num
        start, end = _extract_time_range_pt(user_msg, p.get("table"))
        if start and end and "date__date_between" not in where:
            where["date__date_between"] = [start, end]

        p["where"] = where

        if not p.get("table") or not where:
            return Plan(mode=plan.mode, tool=tool, params=p,
                        rewrite=plan.rewrite, confidence=plan.confidence), "sem_parametros"

    elif tool == "list_csvs":
        pass

    new_plan = Plan(mode=plan.mode, tool=tool, params=p,
                    rewrite=plan.rewrite, confidence=plan.confidence)
    return new_plan, None

def _force_intersection_if_applicable(user_msg: str) -> Plan | None:
    if _intent_hint(user_msg) != "sku_intersection":
        return None
    loja = _extract_loja_from_text(user_msg)
    start, end = _extract_time_range_pt(user_msg, None)
    if loja is None or not (start and end):
        return None
    return Plan(
        mode="tool",
        tool="sku_intersection",
        params={"loja": loja, "start": start, "end": end, "limit": 500},
        rewrite=None,
        confidence=0.95
    )

_NEEDS_DATA_RE = re.compile(
    r"\b(top|soma|sum|conta|count|m[eé]dia|avg|min|max|total|quant|quanto|"
    r"valor|pre[çc]o|vendas?|entre\s+\d{4}|\d{4}\-\d{2}\-\d{2})\b", re.I)

OUTSIDE_AUTOEXEC = os.environ.get("OUTSIDE_AUTOEXEC", "on").lower() in {"on","1","true","yes"}

RUN_TAG_RE = re.compile(r"\[\[RUN\]\](.+?)\[\[/RUN\]\]", re.S | re.I)

def _extract_run_commands(text: str) -> list[str]:
    if not text:
        return []
    cmds = [re.sub(r"\s+", " ", m.strip()) for m in RUN_TAG_RE.findall(text)]
    return [c for c in cmds if c]

def _looks_like_data_question(msg: str) -> bool:
    return bool(_NEEDS_DATA_RE.search(msg or ""))

def _answer_outside_tools(msg: str) -> str:
    # modelo principal com teto maior de tokens
    big_model = _make_model(PRIMARY_MODEL, "primary")
    try:
        big_model.max_tokens = OUT_OF_TOOL_MAXTOKENS
    except Exception:
        pass

    # catálogo resumido para o modelo não inventar tabela/coluna
    schema = _mini_schema_for_prompt(msg)

    system = (
        "🔍 Você é um ANALISTA DE FRAUDES & SOLUCIONADOR do Sentinela.\n\n"

        "MISSÃO: Ajudar a detectar fraudes e inconsistências em dados de estoque através de análise cruzada.\n\n"

        "REGRAS DE OPERAÇÃO:\n\n"

        "1) PERGUNTAS CONCEITUAIS:\n"
        "   → Responda de forma clara e objetiva\n"
        "   → Explique padrões de fraude quando relevante\n\n"

        "2) PERGUNTAS QUE EXIGEM DADOS (contar/filtrar/agrupar/cruzar):\n"
        "   → NÃO invente números ou dados\n"
        "   → Proponha EXATAMENTE UM comando válido dentro de [[RUN]] ... [[/RUN]]\n"
        "   → SEM explicações DENTRO das tags\n\n"

        "   EXEMPLOS DE COMANDOS VÁLIDOS:\n"
        "   • Visualizar dados:\n"
        "     [[RUN]]mostre ajustes_estoque com 10 linhas[[/RUN]]\n"
        "     [[RUN]]mostre vendas_canceladas com 5 linhas[[/RUN]]\n\n"

        "   • Filtrar por loja/SKU:\n"
        "     [[RUN]]filtre loja=022 em ajustes_estoque limite 20[[/RUN]]\n"
        "     [[RUN]]filtre loja=01 e sku=12345 em inventario_saida limite 10[[/RUN]]\n\n"

        "   • Agregar/agrupar dados:\n"
        "     [[RUN]]agregue em ajustes_estoque por loja contando id top 10[[/RUN]]\n"
        "     [[RUN]]agregue em vendas_canceladas por sku somando valor_bruto top 20[[/RUN]]\n\n"

        "   • Análise por período:\n"
        "     [[RUN]]filtre data entre 2025-01-01 e 2025-03-31 em troca limite 50[[/RUN]]\n\n"

        "3) APÓS AS TAGS [[RUN]]...[[/RUN]]:\n"
        "   → Explique brevemente o que o comando faz\n"
        "   → Oriente como interpretar o resultado\n"
        "   → Se for análise de fraude, destaque o que procurar (padrões suspeitos)\n\n"

        "4) PROBLEMAS DE CODIFICAÇÃO:\n"
        "   → Considere acentos (ex.: 'Sa?a' vs 'Saída')\n"
        "   → Para filtros de texto, prefira 'icontains' (case-insensitive)\n\n"

        "5) TABELAS E COLUNAS DISPONÍVEIS:\n"
        "   → Use SOMENTE tabelas/colunas do catálogo abaixo\n"
        "   → NÃO invente nomes de tabelas ou colunas\n\n"

        "   TABELAS PRINCIPAIS:\n"
        "   • ajustes_estoque - Ajustes manuais de estoque\n"
        "   • vendas_canceladas - Cancelamentos de vendas\n"
        "   • inventario_saida - Saídas de inventário\n"
        "   • troca - Trocas de produtos\n\n"

        "6) ANÁLISE DE FRAUDES:\n"
        "   → Quando detectar padrões suspeitos, use emojis:\n"
        "     🚨 = Fraude altamente provável\n"
        "     ⚠️ = Padrão suspeito que requer investigação\n"
        "     ℹ️ = Informação relevante\n\n"

        "   PADRÕES SUSPEITOS COMUNS:\n"
        "   • SKU com múltiplos ajustes em curto período\n"
        "   • Ajuste seguido de cancelamento\n"
        "   • Inventário saída + troca no mesmo dia\n"
        "   • Usuário com alta frequência de ajustes\n\n"

        f"CATÁLOGO DE DADOS:\n{json.dumps(schema, ensure_ascii=False)}\n"
    )

    # se for claramente “pergunta de dados”, força a sugerir 1 comando
    if _looks_like_data_question(msg):
        prompt = (
            "O usuário quer um resultado que requer consulta ao banco.\n"
            "Proponha EXATAMENTE UM comando do Sentinela dentro de [[RUN]]...[[/RUN]] e depois explique.\n\n"
            f"Pedido do usuário: {msg}"
        )
    else:
        prompt = f"Pedido do usuário: {msg}\nSe precisar de dados, use [[RUN]]...[[/RUN]]."

    tmp_agent = Agent(name="explainer", model=big_model, system_message=system)
    ok, out, err, _tools = _run_with_retry(tmp_agent, prompt)
    return out or (f"Erro: {err}" if err else "Sem resposta.")
CMD_LINE_RE = re.compile(r'^\s*(mostre|agregue|filtr[ea])\b.+$', re.I | re.M)

def _extract_first_command(text: str) -> str | None:
    """Pega a primeira linha que parece um comando reconhecido pelos parsers."""
    if not text:
        return None
    m = CMD_LINE_RE.search(text)
    return m.group(0).strip() if m else None

def _autorun_if_command(text_from_outside: str):
    """
    Se o outside escreveu um comando válido (ex.: 'agregue ...'),
    reenvia esse comando para o planner e já renderiza o HTML de tool.
    """
    cmd = _extract_first_command(text_from_outside)
    if not cmd:
        return None  # nada para rodar

    mode, payload = planner_route(cmd)
    if mode == "tool" and payload:
        banner = (
            "<div class='muted'>Execução automática (outside): "
            f"<code>{html.escape(cmd)}</code></div>"
        )
        return banner + payload
    return None

# =========================
# SQL tools (consultas)
# =========================

def sql_count_fn(table: str, where: dict | None = None):
    """
    Conta linhas em uma tabela aplicando filtros (where) arbitrários.
    where usa os mesmos operadores do sql_filter_fn (_build_where).
    Ex.: {"LOJA__eq": "333"} ou {"TIPOMOVIMENTACAO__eq": "Saída", "date__date_between":["2025-01-01","2025-06-30"]}
    """
    if not _table_exists(table):
        return {"status": "error", "message": f"Tabela não encontrada: {table}"}
    where = where or {}
    try:
        where_sql, params = _build_where(table, where)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    q = f'SELECT COUNT(*) AS total FROM {_fq_table(table)}'
    if where_sql:
        q += f" WHERE {where_sql}"
    rows = _pg_exec(q, params)
    total = int(rows[0]["total"]) if rows else 0

    # envelope + html simples (caso alguém chame direto)
    return _envelope(
        True,
        answer_type="html",
        data=[{"total": total}],
        stats={"total": total, "plan": "count"},
        diagnostics={"table": table, "where": where},
        html_str=f"<p><b>Total:</b> {total}</p>",
    )

def sql_head_fn(table: str, n: int = 5):
    if not _table_exists(table):
        return {"status": "error", "message": f"Tabela não encontrada: {table}"}
    rows = _pg_exec(f'SELECT * FROM {_fq_table(table)} LIMIT %s', [max(1, int(n))])
    return {"status": "ok", "rows": rows}

def sql_aggregate_fn(
    table: str,
    by,
    value: str | None = None,
    op: str = "sum",
    top: int = 10,
    date_col: str | None = None,
    start: str | None = None,
    end: str | None = None,
):
    if not _table_exists(table):
        return {"status": "error", "message": f"Tabela não encontrada: {table}"}

    cols_all = set(_pg_cols(table))
    # normaliza BY → lista
    if isinstance(by, str):
        by_cols_raw = [c.strip() for c in re.split(r"[,\+\|]", by) if c.strip()]
    elif isinstance(by, (list, tuple)):
        by_cols_raw = [str(c).strip() for c in by if str(c).strip()]
    else:
        return {"status": "error", "message": "Parâmetro 'by' inválido"}

    if not by_cols_raw:
        return {"status": "error", "message": "Informe ao menos uma coluna em 'by'"}

    # resolve nomes (aceita chaves semânticas)
    resolved = []
    for b in by_cols_raw:
        col = _key_to_col(table, b) or _resolve_col(table, b) or b
        if col not in cols_all:
            return {"status": "error", "message": f"Coluna de agrupamento não existe: {b}"}
        resolved.append(col)

    # SELECT/GROUP: normaliza DATA para 'date(...)'
    if not CATALOG:
        rebuild_catalog()

    select_parts, group_parts, out_cols = [], [], []
    col_tags = (CATALOG.get(table, {}) or {}).get("col_tags", {})
    for col in resolved:
        tags = col_tags.get(col) or set()
        if "date" in tags:
            expr = f"date({_date_expr_auto(col)})"
            alias = col  # mantém nome original
        else:
            expr = f'"{col}"'
            alias = col
        select_parts.append(f"{expr} AS \"{alias}\"")
        group_parts.append(expr)
        out_cols.append(alias)

    # OP
    op = (op or "").lower()
    allowed = {"sum", "count", "avg", "min", "max"}
    if op not in allowed:
        return {"status": "error", "message": f"Operação não permitida: {op}"}

    if op == "count":
        agg_expr = "COUNT(*)"
    else:
        if value is None or value not in cols_all:
            return {"status": "error", "message": "Informe a coluna numérica em `value`"}
        agg_expr = f'{op}(CAST("{value}" AS REAL))'

    # WHERE de período
    where_sql, params = "", []
    if start and end:
        date_col_eff = date_col or _guess_cols(table).get("data")
        if date_col_eff:
            date_expr = _date_expr_auto(date_col_eff)
            where_sql = f" WHERE date({date_expr}) BETWEEN date(?) AND date(?)"
            params.extend([start, end])

    q = (
        f'SELECT {", ".join(select_parts)}, {agg_expr} AS valor '
        f'FROM {_fq_table(table)}'
    )
    if where_sql:
        q += where_sql
    q += f" GROUP BY {', '.join(group_parts)} ORDER BY valor DESC LIMIT %s"
    params.append(max(1, int(top)))

    rows = _pg_exec(q, params)
    return {"status": "ok", "rows": rows, "meta": {"table": table, "by": out_cols, "op": op, "value": value, "start": start, "end": end}}

def _guess_cols(table: str):
    cols = _pg_cols(table)

    def norm(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', s.lower())

    def find_col(candidates: list[str]) -> str | None:
        norm_map = {norm(c): c for c in cols}
        for cand in candidates:
            hit = norm_map.get(norm(cand))
            if hit:
                return hit

        for c in cols:
            lc = re.sub(r'[_]+', ' ', c.lower())
            for cand in candidates:
                if re.search(r'\b' + re.escape(cand.lower()) + r'\b', lc):
                    return c

        for c in cols:
            lc = c.lower()
            for cand in candidates:
                if cand.lower() in lc:
                    return c
        return None

    sku  = find_col(["SKU","COD_SKU","CODIGO_PRODUTO","PRODUTO","CODPROD","COD_PROD","ID_SKU","IDPRODUTO"])
    loja = find_col(["LOJA","ID_LOJA","LOJACOD","FILIAL","STORE","COD_LOJA","CODFILIAL"])
    data = find_col(["DATA","DATACANCELAMENTO","DATA_DEVOLUCAO","DATA_DEVOLUCOES","DATADEVOLUCAO",
                     "DATADEV","DATAVENDA","DT_MOV","DTMOV","DTMOVTO"])

    return {"sku": sku, "loja": loja, "data": data, "all": cols}

def _date_expr_auto(col: str) -> str:
    # remove horário se existir
    base = f'(CASE WHEN instr("{col}", " ")>0 THEN substr("{col}",1,instr("{col}"," ")-1) ELSE "{col}" END)'
    # posição do '-' em YYYY-MM-DD
    iso_check = f"(instr({base}, '-')=5)"
    # montar aaaa-mm-dd a partir de d/m/aaaa (1 ou 2 dígitos)
    y_from_slash = (
        f"substr({base}, "
        f"       instr({base},'/') + instr(substr({base}, instr({base},'/')+1), '/') + 1, 4)"
    )
    m_from_slash = (
        f"substr({base}, "
        f"       instr({base},'/')+1, "
        f"       instr(substr({base}, instr({base},'/')+1), '/')-1)"
    )
    d_from_slash = f"substr({base}, 1, instr({base},'/')-1)"
    dmy_flex = (
        f"printf('%04d-%02d-%02d', "
        f"       CAST({y_from_slash} AS INT), "
        f"       CAST({m_from_slash} AS INT), "
        f"       CAST({d_from_slash} AS INT))"
    )
    return (
        f"(CASE "
        f"   WHEN {iso_check} THEN substr({base},1,10) "
        f"   WHEN instr({base},'/')>0 THEN {dmy_flex} "
        f"   ELSE {base} "
        f" END)"
    )

def sku_intersection_fn(
    ajustes: str|None = None,
    devol:   str|None = None,
    loja:    int|str|None = None,
    start:   str|None = None,
    end:     str|None = None,
    limit:   int = 100,
    where:   dict | None = None,
    where_a: dict | None = None,
    where_d: dict | None = None,
):
    if not CATALOG:
        rebuild_catalog()

    if not ajustes:
        ajustes = next((t for t, info in CATALOG.items() if {"adjust","sku"} <= info["table_tags"]), None)
    if not devol:
        devol   = next((t for t, info in CATALOG.items() if {"return","sku"} <= info["table_tags"]), None)

    if not _table_exists(ajustes) or not _table_exists(devol):
        return {"status":"error","message":f"Tabelas não encontradas (ajustes={ajustes}, devol={devol})"}

    a_meta = _guess_cols(ajustes)
    d_meta = _guess_cols(devol)
    if not a_meta["sku"] or not d_meta["sku"]:
        return {"status":"error","message":"Não consegui identificar a coluna SKU nas tabelas informadas."}

    a_where_all = dict(where or {})
    d_where_all = dict(where or {})

    if loja is not None:
        a_where_all["loja__eq"] = str(loja)
        d_where_all["loja__eq"] = str(loja)

    if where_a: a_where_all.update(where_a)
    if where_d: d_where_all.update(where_d)

    a_where_sql, a_params = _build_where(ajustes, a_where_all)
    d_where_sql, d_params = _build_where(devol,   d_where_all)

    a_sku = f'"{_key_to_col(ajustes, "sku", a_meta) or a_meta["sku"]}"'
    d_sku = f'"{_key_to_col(devol,   "sku", d_meta) or d_meta["sku"]}"'

    q = f"""
    WITH a_f AS (
      SELECT {a_sku} AS SKU
      FROM {_fq_table(ajustes)}
      {"WHERE " + a_where_sql if a_where_sql else ""}
    ),
    d_f AS (
      SELECT {d_sku} AS SKU
      FROM {_fq_table(devol)}
      {"WHERE " + d_where_sql if d_where_sql else ""}
    ),
    inter AS (
      SELECT DISTINCT a_f.SKU
      FROM a_f JOIN d_f ON a_f.SKU = d_f.SKU
    )
    SELECT SKU FROM inter LIMIT %s
    """
    rows = _pg_exec(q, a_params + d_params + [max(1, int(limit))])

    diag_q = f"""
      SELECT
        (SELECT COUNT(*) FROM a_f) AS ajustes_filtrados,
        (SELECT COUNT(*) FROM d_f) AS devolucoes_filtradas,
        (SELECT COUNT(*) FROM inter) AS intersecao
      FROM (SELECT 1)
    """
    diag = _pg_exec(
        f"""
        WITH a_f AS (
          SELECT {a_sku} AS SKU FROM {_fq_table(ajustes)} {"WHERE " + a_where_sql if a_where_sql else ""}
        ),
        d_f AS (
          SELECT {d_sku} AS SKU FROM {_fq_table(devol)} {"WHERE " + d_where_sql if d_where_sql else ""}
        ),
        inter AS (
          SELECT DISTINCT a_f.SKU FROM a_f JOIN d_f ON a_f.SKU = d_f.SKU
        )
        {diag_q}
        """,
        a_params + d_params
    )
    meta = {
        "tables": {"ajustes": ajustes, "devol": devol},
        "diagnostic": (diag[0] if diag else {}),
        "resolved": {
            "ajustes": {"sku": a_meta.get("sku"),  "loja": a_meta.get("loja"), "data": a_meta.get("data")},
            "devol":   {"sku": d_meta.get("sku"),  "loja": d_meta.get("loja"), "data": d_meta.get("data")},
        },
    }

    return {"status":"ok","rows": rows, "meta": meta}

def _resolve_col(table: str, name: str) -> str | None:
    cols = _pg_cols(table)
    m = {c.lower(): c for c in cols}
    return m.get(name.lower())

def _is_digits_str(x) -> bool:
    s = str(x)
    return re.fullmatch(r"0*\d+", s) is not None

def _norm_numstr_sql(expr: str) -> str:
    # PostgreSQL usa ~ para regex (SQLite usava GLOB)
    return (
        f"(CASE "
        f"  WHEN {expr} ~ '^[0-9]' "
        f"  THEN (CASE WHEN ltrim({expr}, '0')='' THEN '0' ELSE ltrim({expr}, '0') END) "
        f"  ELSE {expr} "
        f"END)"
    )

def _extract_eq_pairs(text: str) -> dict:
    """
    Varre pares do tipo COL=VAL no texto (com ou sem aspas) e devolve em formato where {COL__eq: VAL}.
    Ex.: 'LOJA=017 e SKU="999"' -> {"LOJA__eq": "017", "SKU__eq": "999"}
    """
    out = {}
    if not text:
        return out
    for m in re.finditer(r'\b([A-Za-z0-9_]+)\s*=\s*("([^"]+)"|\'([^\']+)\'|([^\s,;]+))', text):
        col = m.group(1)
        val = m.group(3) or m.group(4) or m.group(5)
        if val is not None:
            out[f"{col}__eq"] = val
    return out

def _build_where(table: str, where: dict) -> tuple[str, list]:
    meta = _guess_cols(table)
    clauses, params = [], []

    for key, val in (where or {}).items():
        if "__" in key:
            key_raw, op = key.split("__", 1)
        else:
            key_raw, op = key, "eq"

        col = _key_to_col(table, key_raw, meta=meta) or _resolve_col(table, key_raw)
        if not col:
            raise ValueError(f"Coluna não existe (ou chave semântica não mapeada): {key_raw}")

        op = op.lower()
        norm_col = _norm_numstr_sql(f'"{col}"')

        if op == "eq":
            if _is_digits_str(val):
                clauses.append(f"{norm_col} = (CASE WHEN ltrim(%s, '0')='' THEN '0' ELSE ltrim(%s, '0') END)")
                params.extend([str(val), str(val)])
            else:
                clauses.append(f'"{col}" = %s'); params.append(val)

        elif op == "ne":
            if _is_digits_str(val):
                clauses.append(f"{norm_col} != (CASE WHEN ltrim(%s, '0')='' THEN '0' ELSE ltrim(%s, '0') END)")
                params.extend([str(val), str(val)])
            else:
                clauses.append(f'"{col}" != %s'); params.append(val)

        elif op in {"gt","gte","lt","lte"}:
            cast = f'CAST("{col}" AS REAL)'
            sym = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[op]
            if isinstance(val, str):
                try: val = float(val.replace(",", "."))
                except: pass
            clauses.append(f"{cast} {sym} %s"); params.append(val)

        elif op == "icontains":
            # normaliza o valor de busca para minúsculas
            val_norm = str(val).lower()
            # citação da coluna sem f-string aninhada (evita SyntaxError)
            col_quoted = f'"{col}"'
            col_norm = f"lower({col_quoted})"
            clauses.append(f"{col_norm} LIKE %s")
            params.append(f"%%{val_norm}%%")

        elif op == "between":
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(f"between espera [low, high] em {key}")
            lo, hi = val
            if isinstance(lo, str):
                try: lo = float(lo.replace(",", "."));
                except: pass
            if isinstance(hi, str):
                try: hi = float(hi.replace(",", "."));
                except: pass
            cast = f'CAST("{col}" AS REAL)'
            clauses.append(f"({cast} BETWEEN %s AND %s)"); params.extend([lo, hi])

        elif op == "date_between":
            expr = _date_expr_auto(col)
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(f"date_between espera [start, end] em {key}")
            start, end = val
            clauses.append(f"(date({expr}) BETWEEN date(%s) AND date(%s))"); params.extend([start, end])

        elif op == "contains":
            clauses.append(f'"{col}" LIKE %s'); params.append(f"%%{val}%%")

        elif op == "in":
            seq = list(val) if isinstance(val, (list, tuple, set)) else []
            if not seq:
                clauses.append("1=0")
            elif all(_is_digits_str(v) for v in seq):
                ph = ",".join(["(CASE WHEN ltrim(%s, '0')='' THEN '0' ELSE ltrim(%s, '0') END)"] * len(seq))
                clauses.append(f"{norm_col} IN ({ph})")
                for v in seq:
                    params.extend([str(v), str(v)])
            else:
                ph = ",".join(["%s"] * len(seq))
                clauses.append(f'"{col}" IN ({ph})'); params.extend(list(seq))
        else:
            raise ValueError(f"Operador não suportado: {op}")
        
    where_sql = " AND ".join(clauses) if clauses else ""
    return where_sql, params

def _envelope(ok: bool, *, answer_type="html", message=None, data=None, stats=None, diagnostics=None, html_str=None):
    return {
        "ok": bool(ok),
        "answer_type": answer_type,
        "message": message,
        "data": data if data is not None else [],
        "stats": stats or {},
        "diagnostics": diagnostics or {},
        "html": html_str or "",
        "status": "ok" if ok else "error",
    }

def _rows_to_html(rows: list[dict]) -> str:
    if not rows:
        return "<div class='muted'>Sem resultados para os filtros aplicados.</div>"
    cols = list(rows[0].keys())
    thead = "<thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead>"
    body_rows = []
    for r in rows:
        tds = "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols)
        body_rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"

def sql_filter_fn(table: str, where: dict | None = None, limit: int = 100,
                  order_by: str | None = None, order: str = "desc", offset: int = 0):
    if not _table_exists(table):
        return {"status":"error","message":f"Tabela não encontrada: {table}"}

    where = where or {}
    try:
        where_sql, params = _build_where(table, where)
    except ValueError as e:
        return {"status":"error","message":str(e)}

    ob_col = _resolve_col(table, order_by) if order_by else None
    ord_kw = "ASC" if str(order).lower() == "asc" else "DESC"

    q = f'SELECT * FROM {_fq_table(table)}'
    if where_sql:
        q += f" WHERE {where_sql}"
    if ob_col:
        q += f' ORDER BY "{ob_col}" {ord_kw}'
    q += " LIMIT %s OFFSET %s"

    rows = _pg_exec(q, params + [max(1, int(limit)), max(0, int(offset))])
    if not rows and where:
        # testa cada condição isoladamente para apontar a "culpada"
        diag = {}
        for k,v in where.items():
            try:
                ws, ps = _build_where(table, {k:v})
                cnt = _pg_exec(f'SELECT COUNT(*) AS c FROM {_fq_table(table)} WHERE {ws}', ps)[0]["c"]
                diag[k] = cnt
            except Exception as e:
                diag[k] = f"erro: {e}"
        return _envelope(
            True,
            answer_type="html",
            data=[],
            stats={"rows_returned": 0, "plan": "filter"},
            diagnostics={"table": table, "where": where, "isolated_counts": diag},
            html_str="<div class='muted'>Sem resultados para os filtros aplicados.</div>"
        )

    return _envelope(
        True,
        answer_type="html",
        data=rows,
        stats={"rows_returned": len(rows), "plan": "filter", "offset": int(offset), "limit": int(limit)},
        diagnostics={"table": table, "where": where, "order_by": ob_col, "order": ord_kw},
        html_str=_rows_to_html(rows),
    )

def _nl_aggregate_summary(rows: list[dict], *, op: str, by: str) -> str:
    try:
        if not rows:
            return ""
        # pega top 1 e top 3 somados
        top1 = rows[0]
        chave = top1.get("chave")
        valor = float(top1.get("valor", 0))
        total_top3 = sum(float(r.get("valor", 0)) for r in rows[:3])
        if op == "count":
            return f"• Destaque: {by} '{chave}' tem {int(valor)} registros (top 1). Top 3 somam {int(total_top3)}."
        else:
            return f"• Destaque: {by} '{chave}' lidera com {valor:,.2f}. Top 3 somam {total_top3:,.2f}."
    except Exception:
        return ""

@app.get("/tool/sql_filter")
def call_sql_filter(
    table: str = Query(...),
    where: str = Query("{}"),
    limit: int = Query(100),
    order_by: str | None = Query(None),
    order: str = Query("desc"),
    offset: int = Query(0),
):
    try:
        where_dict = json.loads(where) if where else {}
        if not isinstance(where_dict, dict):
            return {"status":"error","message":"`where` deve ser um JSON de objeto"}
    except Exception as e:
        return {"status":"error","message":f"JSON inválido em `where`: {e}"}
    return sql_filter_fn(table, where_dict, limit, order_by, order, offset)

@app.get("/tool/sku_intersection")
def call_sku_intersection(
    ajustes: str | None = Query(None, description="ex.: ajustes_estoque (opcional)"),
    devol:   str | None = Query(None, description="ex.: inventario_saida (opcional)"),
    loja:    str       = Query(...,  description="ex.: 17 ou 017"),
    start:   str       = Query(...,  description="AAAA-MM-DD"),
    end:     str       = Query(...,  description="AAAA-MM-DD"),
    limit:   int       = Query(100)
):
    return sku_intersection_fn(ajustes, devol, loja, start, end, limit)

@app.get("/tool/sql_head")
def call_sql_head(table: str = Query(...), n: int = Query(5)):
    return sql_head_fn(table, n)

@app.get("/tool/sql_aggregate")
def call_sql_aggregate(
    table: str = Query(...),
    by: str = Query(...),
    value: str | None = Query(None),
    op: str = Query("sum"),
    top: int = Query(10)
):
    return sql_aggregate_fn(table, by, value, op, top)

# =========================
# Ferramentas Avançadas de Cruzamento
# =========================

def sku_tre_table_fn(
    p_loja: str,
    p_data_ini: str,
    p_data_fim: str
):
    """
    Chama a FUNCTION do PostgreSQL 'sku_tre_table' que cruza SKUs entre
    as três tabelas: ajustes_estoque, inventario_saida e troca.

    Retorna SKUs que aparecem em TROCA (obrigatório) E em AJUSTE ou INVENTÁRIO.

    Parâmetros:
    - p_loja: código da loja (ex: '002', '111')
    - p_data_ini: data inicial YYYY-MM-DD
    - p_data_fim: data final YYYY-MM-DD

    Retorna:
    - Lista de SKUs com datas de cada origem (ajuste_contagem, inventario_saida, devolucao_troca)
    """
    try:
        # Valida parâmetros
        if not p_loja:
            return {"status": "error", "message": "Parâmetro 'p_loja' é obrigatório"}
        if not p_data_ini or not p_data_fim:
            return {"status": "error", "message": "Parâmetros 'p_data_ini' e 'p_data_fim' são obrigatórios"}

        # Normaliza loja (remove zeros à esquerda extra, mas mantém formato)
        p_loja = str(p_loja).strip()

        # Chama a FUNCTION do PostgreSQL (3 parâmetros: loja, data_ini, data_fim)
        q = f"SELECT * FROM {DB_SCHEMA}.sku_tre_table(%s, %s, %s)"
        rows = _pg_exec(q, [p_loja, p_data_ini, p_data_fim])

        return {
            "status": "ok",
            "rows": rows,
            "total": len(rows),
            "params": {
                "loja": p_loja,
                "data_ini": p_data_ini,
                "data_fim": p_data_fim
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao executar sku_tre_table: {e}"}





def pandas_analysis_fn(
    tables: list[str],
    analysis_type: str,
    filters: dict | None = None,
    group_by: str | None = None,
    date_range: list[str] | None = None,
    limit: int = 100
):
    """
    Análise avançada com pandas - cruza múltiplas tabelas, filtra, agrega.
    Similar ao GPT-5 com execução de código Python.

    Args:
        tables: Lista de nomes de tabelas ['ajustes_estoque', 'inventario_saida']
        analysis_type: 'intersection', 'cross_join', 'aggregate', 'filter'
        filters: {table_name: {column: value}} ex: {'ajustes_estoque': {'TIPO': 'saída'}}
        group_by: Coluna para agrupar
        date_range: ['2025-01-01', '2025-06-30']
        limit: Limite de resultados
    """
    try:
        # Conecta ao PostgreSQL Data Mart
        conn = _get_pg_connection()

        # Carrega tabelas em DataFrames
        dfs = {}
        for table in tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {_fq_table(table)}", conn)
                dfs[table] = df
            except Exception as e:
                conn.close()
                return {"status": "error", "message": f"Erro ao carregar {table}: {str(e)}", "rows": []}

        conn.close()

        # Aplica filtros por tabela
        if filters:
            for table_name, table_filters in filters.items():
                if table_name in dfs:
                    df = dfs[table_name]
                    for col, val in table_filters.items():
                        # Busca coluna case-insensitive
                        actual_col = None
                        for c in df.columns:
                            if c.upper() == col.upper():
                                actual_col = c
                                break

                        if actual_col and actual_col in df.columns:
                            if isinstance(val, str):
                                df = df[df[actual_col].astype(str).str.contains(val, case=False, na=False)]
                            else:
                                df = df[df[actual_col] == val]
                    dfs[table_name] = df

        # Aplica filtro de data
        if date_range and len(date_range) == 2:
            for table_name, df in dfs.items():
                date_col = _guess_cols(df.columns, ["data", "date", "dt"])[0]
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df[(df[date_col] >= date_range[0]) & (df[date_col] <= date_range[1])]
                    dfs[table_name] = df

        # Executa análise
        if analysis_type == "intersection" and len(tables) >= 2:
            # Interseção de SKUs (como GPT-5 faz)
            sku_col_map = {}
            for table_name, df in dfs.items():
                sku_col = _guess_cols(df.columns, ["sku", "produto", "cod"])[0]
                if not sku_col:
                    return {"status": "error", "message": f"Coluna SKU não encontrada em {table_name}", "rows": []}
                sku_col_map[table_name] = sku_col

            # Cria sets de SKUs
            skus_sets = {name: set(df[sku_col_map[name]].dropna().astype(str)) for name, df in dfs.items()}
            common_skus = set.intersection(*skus_sets.values())

            # Monta resultado detalhado
            result_rows = []
            for sku in sorted(list(common_skus))[:limit]:
                row = {"SKU": sku}
                for table_name, df in dfs.items():
                    sku_col = sku_col_map[table_name]
                    sku_data = df[df[sku_col].astype(str) == sku]
                    if not sku_data.empty:
                        date_col = _guess_cols(df.columns, ["data", "date", "dt"])[0]
                        if date_col and date_col in sku_data.columns:
                            dates = sku_data[date_col].dropna().astype(str).tolist()
                            row[f"{table_name}_DATAS"] = ", ".join(dates[:3])  # Primeiras 3 datas
                        row[f"{table_name}_QTD"] = len(sku_data)
                result_rows.append(row)

            summary = {
                "total_skus_comuns": len(common_skus),
                "tabelas_analisadas": len(tables),
                "registros_por_tabela": {name: len(df) for name, df in dfs.items()},
                "skus_por_tabela": {name: len(skus) for name, skus in skus_sets.items()}
            }

            return {
                "status": "ok",
                "rows": result_rows,
                "summary": summary
            }

        elif analysis_type == "aggregate" and group_by:
            # Agregação
            df = list(dfs.values())[0]
            actual_col = None
            for c in df.columns:
                if c.upper() == group_by.upper():
                    actual_col = c
                    break

            if actual_col and actual_col in df.columns:
                agg_df = df.groupby(actual_col).size().reset_index(name='CONTAGEM')
                agg_df = agg_df.sort_values('CONTAGEM', ascending=False).head(limit)
                result_rows = agg_df.to_dict('records')

                return {
                    "status": "ok",
                    "rows": result_rows,
                    "summary": {"total_grupos": len(agg_df), "total_registros": len(df)}
                }

        return {
            "status": "error",
            "message": f"Tipo '{analysis_type}' não suportado ou parâmetros insuficientes",
            "rows": []
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro em pandas_analysis: {str(e)}\n{traceback.format_exc()}",
            "rows": []
        }


# =========================
# Registro das tools e Agent
# =========================
sku_intersection_tool = tool(
    name="sku_intersection",
    description=(
        "Calcula a interseção de SKUs entre duas TABELAS distintas, aplicando os mesmos filtros de loja e período. "
        "Parâmetros: ajustes (opcional), devol (opcional), loja (ex.: 17), start (AAAA-MM-DD), end (AAAA-MM-DD), limit. "
        "Se os nomes das tabelas não forem informados, a ferramenta escolhe automaticamente com base no catálogo e nas tags."
    )
)(sku_intersection_fn)

# Ferramentas desabilitadas (CSV)
head_csv_tool = tool(
    name="head_csv",
    description="[DESABILITADA] Use sql_head para visualizar dados das tabelas do PostgreSQL."
)(head_csv_fn)

list_csvs_tool = tool(
    name="list_csvs",
    description="[DESABILITADA] Use list_tables para listar tabelas disponíveis no PostgreSQL."
)(list_csvs_fn)

# Nova ferramenta para listar tabelas
list_tables_tool = tool(
    name="list_tables",
    description="Lista todas as tabelas disponíveis no PostgreSQL Data Mart. Tabelas: ajustes_estoque, inventario_saida, troca, vendas_canceladas."
)(list_tables_fn)

# Ferramentas SQL (PostgreSQL Data Mart)
sql_head_tool = tool(
    name="sql_head",
    description="Mostra as primeiras N linhas de uma tabela do PostgreSQL. Tabelas disponíveis: ajustes_estoque, inventario_saida, troca, vendas_canceladas."
)(sql_head_fn)

sql_aggregate_tool = tool(
    name="sql_aggregate",
    description="Agrega valores por categoria na tabela do PostgreSQL. op: sum/count/avg/min/max. Tabelas: ajustes_estoque, inventario_saida, troca, vendas_canceladas."
)(sql_aggregate_fn)

sql_filter_tool = tool(
    name="sql_filter",
    description="Filtra linhas no PostgreSQL com operadores: eq, ne, gt, gte, lt, lte, in, contains, icontains. Tabelas: ajustes_estoque, inventario_saida, troca, vendas_canceladas."
)(sql_filter_fn)

sql_count_tool = tool(
    name="sql_count",
    description="Conta linhas em uma tabela do PostgreSQL aplicando filtros (where). Tabelas: ajustes_estoque, inventario_saida, troca, vendas_canceladas."
)(sql_count_fn)

sku_tre_table_tool = tool(
    name="sku_tre_table",
    description=(
        "**ANALISADOR CRUZADO DE SKUs (Ajuste / Inventário / Troca)**\n\n"
        "Use esta ferramenta quando o usuário pedir uma checagem dos SKUs que:\n"
        "- Aparecem em TROCA (obrigatório) e em AJUSTE ou INVENTÁRIO (ou ambos)\n"
        "- Quer evidências de datas por origem (todas as datas concatenadas)\n"
        "- Filtro por loja e período (datas)\n\n"
        "COMPORTAMENTO PRINCIPAL:\n"
        "Retorna UMA LINHA por SKU com as colunas: sku, ajuste_contagem, inventario_saida, devolucao_troca\n"
        "- A lógica aplicada: (Ajuste OU Inventário) E Troca\n"
        "- Troca é obrigatória (apenas SKUs que aparecem na tabela de troca no período)\n\n"
        "Parâmetros obrigatórios (tipos esperados):\n"
        "- p_loja (string): código da loja, ex: '002', '111', '449' etc...\n"
        "- p_data_ini (date string YYYY-MM-DD): data inicial do período\n"
        "- p_data_fim (date string YYYY-MM-DD): data final do período\n\n"
        "Retorno:\n"
        "- lista de objetos/linhas com: { sku, ajuste_contagem, inventario_saida, devolucao_troca }\n"
        "- datas vêm em formato 'DD/MM/YYYY; DD/MM/YYYY; ...' ou '—' quando ausente\n\n"
        "QUANDO USAR:\n"
        "✓ 'Me traga os SKUs da loja 002 entre janeiro e junho de 2025 que tem ajuste de contagem, inventario e troca.'\n"
        "✓ 'Quais os SKUs da loja 111 no mês de abril de 2025 que tem ajuste de contagem, inventario e troca'\n\n"
        "EXEMPLO DE CHAMADA (SQL-equivalente):\n"
        "SELECT * FROM sku_tre_table('002','2025-01-01','2025-06-30');\n\n"
        "IMPORTANTE: a IA deve extrair p_loja e o período da linguagem natural.\n"
    )
)(sku_tre_table_fn)

pandas_analysis_tool = tool(
    name="pandas_analysis",
    description=(
        "**ANÁLISE AVANÇADA COM PANDAS**\n\n"
        "Ferramenta para análises complexas com múltiplas tabelas usando pandas.\n\n"
        "QUANDO USAR:\n"
        "✓ Cruzamentos entre 2 tabelas específicas\n"
        "✓ Agregações e estatísticas avançadas\n"
        "✓ Para cruzamento de 3 tabelas (Ajuste/Inventário/Troca), use sku_tre_table\n\n"
        "Parâmetros:\n"
        "- tables: lista de tabelas ['ajustes_estoque', 'inventario_saida']\n"
        "- analysis_type: 'intersection' (interseção SKUs), 'aggregate' (agrupar), 'cross_join' (join)\n"
        "- filters: dict {table_name: {column: value}} para filtrar cada tabela\n"
        "- date_range: ['2025-01-01', '2025-06-30'] para filtrar por período\n"
        "- group_by: coluna para agrupar (usado com analysis_type='aggregate')\n"
        "- limit: número máximo de resultados\n\n"
        "Retorna: SKUs + datas de cada tabela + estatísticas detalhadas + resumo"
    )
)(pandas_analysis_fn)

def _execute_plan_to_html(plan: Plan) -> str:
    t = plan.tool
    p = dict(plan.params or {})

    def to_html_rows(res: dict, title: str) -> str:
        if isinstance(res, dict) and "ok" in res and ("html" in res or "data" in res):
            if res.get("html"):
                return res["html"]
            rows = res.get("data") or []
            return _rows_to_html(rows)

        if not isinstance(res, dict):
            return f"<div class='muted'>Resposta inesperada de {title}</div>"

        if res.get("status") != "ok":
            msg = res.get("message") or res
            return f"<div class='muted'>Erro em {title}: {html.escape(str(msg))}</div>"

        rows = res.get("rows") or res.get("preview") or res.get("results")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return _rows_to_html(rows)
        return f"<pre>{html.escape(json.dumps(res, ensure_ascii=False, indent=2))}</pre>"

    if t == "sql_head":
        table = p.get("table")
        if not table:
            return "<div class='muted'>Preciso do nome da tabela. Ex.: <code>mostre vendas_canceladas com 5 linhas</code></div>"
        n = int(p.get("n") or 5)
        return f"<h3>Amostra de <code>{html.escape(table)}</code></h3>" + to_html_rows(sql_head_fn(table, n), "sql_head")

    if t == "sql_filter":
        table = p.get("table")
        where = p.get("where") or {}
        limit = int(p.get("limit") or 100)
        order_by = p.get("order_by")
        order = p.get("order") or "desc"
        return f"<h3>Filtro em <code>{html.escape(table)}</code></h3>" + to_html_rows(sql_filter_fn(table, where, limit, order_by, order), "sql_filter")

    if t == "sql_aggregate":
        table = p.get("table")
        by = p.get("by")
        value = p.get("value")
        op = p.get("op") or "sum"
        top = int(p.get("top") or 10)
        date_col = p.get("date_col")
        start = p.get("start")
        end = p.get("end")
        res = sql_aggregate_fn(table, by, value, op, top, date_col, start, end)
        html_table = to_html_rows(res, "sql_aggregate")

        # pequeno resumo textual (mais natural) para melhorar leitura
        rows = res.get("rows") if isinstance(res, dict) else None
        summary = ""
        if isinstance(rows, list):
            summary = _nl_aggregate_summary(rows, op=op, by=by)

        intro = f"<h3>Agregação em <code>{html.escape(table)}</code></h3>"
        if summary:
            intro += f"<p class='muted'>{html.escape(summary)}</p>"
        return intro + html_table

    if t == "sql_count":
        table = p.get("table")
        where = p.get("where") or {}
        res = sql_count_fn(table, where)

        # tenta montar uma frase do tipo "COL = VAL aparece X vezes ..."
        total = (res.get("stats") or {}).get("total")
        if total is None:
            try:
                total = int((res.get("data") or [{}])[0].get("total", 0))
            except:
                total = 0

        eqs = []
        for k, v in where.items():
            if "__" in k:
                col, op = k.split("__", 1)
            else:
                col, op = k, "eq"
            if op == "eq" and k.lower() not in {"date__date_between"}:
                eqs.append((col, v))

        loja = where.get("loja__eq") or where.get("store__eq") or where.get("filial__eq")
        rng = where.get("date__date_between")

        if eqs:
            col, val = eqs[0]
            frase = f'{col} = {val} aparece {total} vezes em {table}'
            if loja is not None:
                frase += f' na loja {str(loja).lstrip("0") or "0"}'
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                frase += f' entre {rng[0]} e {rng[1]}'
            return f"<h3>Contagem</h3><p>{html.escape(frase)}</p>"

        return f"<h3>Contagem</h3><p>Total de linhas que atendem aos filtros: <b>{total}</b></p>"

    if t == "sku_intersection":
        res = sku_intersection_fn(
            ajustes=p.get("ajustes"),
            devol=p.get("devol"),
            loja=p.get("loja"),
            start=p.get("start"),
            end=p.get("end"),
            limit=int(p.get("limit") or 100),
            where=p.get("where"),
            where_a=p.get("where_a"),
            where_d=p.get("where_d"),
        )
        if isinstance(res, dict) and res.get("status") == "ok":
            rows = res.get("rows") or []
            meta = res.get("meta") or {}
            if not rows:
                d = meta.get("diagnostic") or {}
                aj = d.get("ajustes_filtrados", 0)
                dv = d.get("devolucoes_filtradas", 0)
                it = d.get("intersecao", 0)
                tabs = meta.get("tables") or {}

                sample_where = {}
                if p.get("loja") is not None:
                    sample_where["loja__eq"] = str(p["loja"]).lstrip("0") or "0"
                if p.get("start") and p.get("end"):
                    sample_where["date__date_between"] = [p["start"], p["end"]]

                a_sample = sql_filter_fn(tabs.get("ajustes"), sample_where, 20)
                d_sample = sql_filter_fn(tabs.get("devol"),   sample_where, 20)

                a_html = a_sample.get("html") or _rows_to_html(a_sample.get("data") or [])
                d_html = d_sample.get("html") or _rows_to_html(d_sample.get("data") or [])

                diag_html = (
                    "<div class='muted'>Sem resultados na interseção</div>"
                    f"<div class='muted'>Tabelas: <code>{html.escape(str(tabs.get('ajustes')))}</code> ∩ "
                    f"<code>{html.escape(str(tabs.get('devol')))}</code></div>"
                    f"<div class='muted'>Registros após filtros — ajustes: <b>{aj}</b>, devoluções: <b>{dv}</b>, interseção: <b>{it}</b></div>"
                )
                return (
                    "<h3>Interseção de SKUs</h3>" + diag_html +
                    "<div class='muted' style='margin-top:8px'>Amostra de SKUs em ajustes (até 20):</div>" + a_html +
                    "<div class='muted' style='margin-top:8px'>Amostra de SKUs em devolução (até 20):</div>" + d_html
                )

        return to_html_rows(res, "sku_intersection")

    if t == "sku_tre_table":
        p_loja = p.get("p_loja") or p.get("loja")
        p_data_ini = p.get("p_data_ini") or p.get("data_ini") or p.get("start")
        p_data_fim = p.get("p_data_fim") or p.get("data_fim") or p.get("end")

        if not p_loja or not p_data_ini or not p_data_fim:
            return "<div class='muted'>Parâmetros obrigatórios: p_loja, p_data_ini, p_data_fim</div>"

        res = sku_tre_table_fn(p_loja, p_data_ini, p_data_fim)

        if isinstance(res, dict) and res.get("status") == "ok":
            rows = res.get("rows") or []
            total = res.get("total", len(rows))
            params = res.get("params") or {}

            summary_html = (
                f"<div class='muted'>"
                f"Loja: <b>{html.escape(str(params.get('loja', p_loja)))}</b> | "
                f"Período: <b>{params.get('data_ini', p_data_ini)}</b> a <b>{params.get('data_fim', p_data_fim)}</b><br>"
                f"<b>{total}</b> SKUs encontrados com (Ajuste OU Inventário) E Troca"
                f"</div>"
            )

            if not rows:
                return f"<h3>🔍 Cruzamento SKU (Ajuste/Inventário/Troca)</h3>{summary_html}<div class='muted'>Nenhum SKU encontrado no cruzamento.</div>"

            table_html = to_html_rows(res, "sku_tre_table")
            return f"<h3>🔍 Cruzamento SKU (Ajuste/Inventário/Troca)</h3>{summary_html}{table_html}"

        return to_html_rows(res, "sku_tre_table")

    if t == "pandas_analysis":
        tables = p.get("tables") or []
        analysis_type = p.get("analysis_type", "intersection")
        filters = p.get("filters") or {}
        group_by = p.get("group_by")
        date_range = p.get("date_range")
        limit = int(p.get("limit") or 100)

        res = pandas_analysis_fn(tables, analysis_type, filters, group_by, date_range, limit)

        if isinstance(res, dict) and res.get("status") == "ok":
            rows = res.get("rows") or []
            summary = res.get("summary") or {}

            # Monta resumo bonito
            summary_parts = []
            if "total_skus_comuns" in summary:
                summary_parts.append(f"<b>{summary['total_skus_comuns']}</b> SKUs encontrados na interseção")
            if "registros_por_tabela" in summary:
                for tbl, count in summary["registros_por_tabela"].items():
                    summary_parts.append(f"<code>{html.escape(tbl)}</code>: {count} registros")

            summary_html = "<div class='muted'>" + " | ".join(summary_parts) + "</div>" if summary_parts else ""

            table_html = to_html_rows(res, "pandas_analysis")

            return f"<h3>📊 Análise com Pandas ({analysis_type})</h3>{summary_html}{table_html}"

        return to_html_rows(res, "pandas_analysis")

# =========================
# Fast-path (comandos diretos)
# =========================
_SMALLTALK_RE = re.compile(
    r"^\s*(oi|olá|ola|oie|hey|eai|e aí|fala|salve|bom dia|boa tarde|boa noite|"
    r"valeu|obg|obrigado|brigado|tudo bem|blz|beleza|teste|test|ping)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

def _is_smalltalk(msg: str) -> bool:
    s = (msg or "").strip()
    if not s:
        return True
    if re.fullmatch(r"[\s\W_]+", s):
        return True
    if re.fullmatch(r"[:;=xX][\-]?[)D\(Pp/\\]+", s):
        return True
    return bool(_SMALLTALK_RE.match(s))

def _help_reply_html() -> str:
    if not CATALOG:
        rebuild_catalog()
    tables = sorted(CATALOG.keys())
    exemplos = (
        "<ul>"
        "<li><code>mostre vendas_canceladas com 5 linhas</code></li>"
        "<li><code>filtre SKU=10220110944410 e LOJA=333 em ajustes_estoque limite 20</code></li>"
        "<li><code>agregue em vendas_canceladas por LOJA somando VALORBRUTO top 5</code></li>"
        "<li><code>SKUs com ajuste e devolução na loja 017 entre 2025-01-01 e 2025-06-30</code></li>"
        "</ul>"
    )
    if tables:
        lis = "".join(f"<li><code>{html.escape(t)}</code></li>" for t in tables[:12])
        cat = f"<p><strong>Tabelas disponíveis</strong>:</p><ul>{lis}</ul>"
    else:
        cat = "<p><em>Nenhuma tabela carregada ainda. Faça upload em <code>/docs → POST /upload_csv</code>.</em></p>"
    return (
        "<p>Olá! 👋 Sou o Sentinela. Diga o que quer ver no banco ou use um dos exemplos abaixo.</p>"
        + exemplos + cat
    )

def fastpath_markdown(msg: str) -> str | None:
    m_head = re.search(r"\bmostr(e|ar|a)\s+([A-Za-z0-9_\.]+)\s+com\s+(\d{1,3})\s+linhas?\b", msg, re.I)
    if m_head:
        table = m_head.group(2)
        n = int(m_head.group(3))
        if not _table_exists(table):
            chosen = resolve_table_from_text(msg)
            if chosen and _table_exists(chosen):
                table = chosen
            else:
                return "<div class='muted'>Não reconheci a tabela citada. Use um nome próximo do arquivo ou faça upload em /docs.</div>"
        res = sql_head_fn(table, n)
        rows = res.get("rows") or []
        html_table = _rows_to_html(rows)
        return f"<h3>Amostra de <code>{html.escape(table)}</code></h3>{html_table}"

    m = re.search(
        r"filtr(a|e)(?:\s+(.+?))?\s+em\s+([A-Za-z0-9_\.]+)(?:.*?(?:até|limite)\s+(\d+))?",
        msg, re.I
    )
    if m:
        conds_str = (m.group(2) or "").strip()
        table = m.group(3)
        limit_s = m.group(4)
        where = {}

        # 1) lista 'IN'
        for p in re.split(r"\s*(?:,| e )\s*", conds_str, flags=re.I):
            if not p: 
                continue
            if re.search(r"\bin\b", p, re.I):
                left, right = re.split(r"\bin\b", p, flags=re.I, maxsplit=1)
                col = left.strip()
                seq = [x.strip() for x in re.split(r"[,\s]+", right) if x.strip()]
                if seq:
                    where[f"{col}__in"] = seq
                continue
            # operadores clássicos (=, !=, >=, <=, >, <)
            for sym, tag in [(">=","gte"),("<=","lte"),("!=","ne"),(">","gt"),("<","lt"),("=","eq")]:
                if sym in p:
                    left, right = p.split(sym, 1)
                    col = left.strip()
                    val = right.strip().strip('"\'')
                    if tag in {"gt","gte","lt","lte"}:
                        try: val = float(val.replace(",", ".")) 
                        except: pass
                    key = f"{col}__{tag}" if tag!="eq" else col
                    where[key] = val
                    break

        # 2) intervalo de datas em linguagem natural
        if re.search(r"\bentre\s+.+\s+e\s+.+", msg, re.I):
            s, e = _extract_time_range_pt(msg, table)
            if s and e:
                where["date__date_between"] = [s, e]

        lim = int(limit_s) if limit_s else 100
        res = sql_filter_fn(table, where, lim)

        if isinstance(res, dict) and (res.get("status") == "ok" or res.get("ok") is True):
            if res.get("html"):
                return f"<h3>Filtro em <code>{html.escape(table)}</code> (PostgreSQL)</h3>{res['html']}"
            data = res.get("data") or res.get("rows") or []
            html_table = _rows_to_html(data)
            return f"<h3>Filtro em <code>{html.escape(table)}</code> (PostgreSQL)</h3>{html_table}"


def _rows_to_md(rows: list[dict]) -> str:
    if not rows:
        return "_sem resultados_"
    cols = list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    sep  = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])
# =========================
# Agente principal
# =========================

agent = Agent(
    name="sentinela",
    system_message=(
    "VOCÊ É UM ESPECIALISTA EM ANÁLISE DE FRAUDES E AUDITORIA DE ESTOQUE.\n"
    "Sua missão é identificar padrões suspeitos cruzando dados entre múltiplas tabelas do sistema de gestão de estoque.\n"
    "Você deve detectar inconsistências, movimentações anômalas e possíveis fraudes através de análise cruzada de dados.\n\n"

    "SCHEMA DO BANCO DE DADOS (PostgreSQL Data Mart)\n"

    "TABELA: ajustes_estoque\n"
    "Descrição: Registra ajustes manuais de estoque (correções, acertos)\n"
    "Colunas:\n"
    "   • id (serial) - Identificador único\n"
    "   • loja (varchar) - Código da loja (ex: '01', '02', '022')\n"
    "   • data (date) - Data do ajuste\n"
    "   • id_user (integer) - ID do usuário que fez o ajuste\n"
    "   • id_tipo_ajuste (integer) - Tipo de ajuste\n"
    "   • sku (varchar) - Código do produto\n"
    "   • qtd_antiga (varchar) - Quantidade antes do ajuste\n"
    "   • qtd_ajuste (varchar) - Quantidade ajustada\n"
    "   • tipo_ajuste (varchar) - Descrição do tipo de ajuste\n"
    "   • created_at (timestamp) - Data de criação do registro\n\n"

    "TABELA: vendas_canceladas\n"
    "   Descrição: Registra cancelamentos de vendas/orçamentos\n"
    "   Colunas:\n"
    "   • id (serial) - Identificador único\n"
    "   • loja (varchar) - Código da loja\n"
    "   • data_cancel (date) - Data do cancelamento\n"
    "   • id_user (integer) - ID do usuário que cancelou\n"
    "   • id_orcamento (bigint) - ID do orçamento cancelado\n"
    "   • valor_bruto (varchar) - Valor do cancelamento\n"
    "   • sku (varchar) - Código do produto\n"
    "   • confirmado_cancelado (varchar) - Status de confirmação\n"
    "   • ativo_cancelado (varchar) - Status ativo/cancelado\n"
    "   • created_at (timestamp) - Data de criação do registro\n\n"

    "TABELA: inventario_saida\n"
    "   Descrição: Registra saídas de inventário (baixas, perdas, transferências)\n"
    "   Colunas:\n"
    "   • id (serial) - Identificador único\n"
    "   • loja (varchar) - Código da loja\n"
    "   • data (date) - Data da saída\n"
    "   • complemento (varchar) - Informações adicionais\n"
    "   • sku (varchar) - Código do produto\n"
    "   • qtd_movimentada (varchar) - Quantidade movimentada\n"
    "   • valor (varchar) - Valor da movimentação\n"
    "   • created_at (timestamp) - Data de criação do registro\n\n"

    "TABELA: troca\n"
    "   Descrição: Registra trocas de produtos\n"
    "   Colunas:\n"
    "   • id (serial) - Identificador único\n"
    "   • loja (varchar) - Código da loja\n"
    "   • sku (varchar) - Código do produto\n"
    "   • data_troca (date) - Data da troca\n"
    "   • id_user (integer) - ID do usuário que registrou\n"
    "   • id_troca (bigint) - ID da troca\n"
    "   • id_orcamento_novo (bigint) - ID do novo orçamento\n"
    "   • id_cliente (varchar) - ID do cliente\n"
    "   • valor_produto (varchar) - Valor do produto\n"
    "   • diferenca_valor_troca (varchar) - Diferença de valor na troca\n"
    "   • tipo_movimentacao (varchar) - Tipo de movimentação\n"
    "   • created_at (timestamp) - Data de criação do registro\n\n"

    "FORMATO DE APRESENTAÇÃO OBRIGATÓRIO\n\n"

    "SEMPRE apresente os resultados em formato de TABELA MARKDOWN:\n\n"

    "Formato padrão para análise cruzada de SKUs:\n\n"
    "| SKU  | Ajuste-Estoque | Inventário-Saída | Troca | Cancelamento |\n"
    "|------|------|----------------|------------------|-------|-------------|\n"
    "| 12345| 2025-01-15; 2025-02-20 | 2025-01-16 | - | 2025-01-17 |\n"
    "| 67890| 2025-03-10 | 2025-03-11; 2025-03-12 | 2025-03-13 | - |\n\n"

    "REGRAS DE FORMATAÇÃO:\n"
    "• Cada célula pode conter MÚLTIPLAS DATAS separadas por ponto e vírgula (`;`)\n"
    "• Use hífen (`-`) quando não houver dados para aquela tabela\n"
    "• Sempre inclua a coluna SKU e Loja\n"
    "• Ordene por SKU ou por frequência de aparições (mais suspeitos primeiro)\n"
    "• Adicione uma linha de RESUMO ao final quando relevante\n\n"

    "FERRAMENTAS DISPONÍVEIS\n\n"

    "REGRA CRÍTICA - CRUZAMENTO DE TABELAS:\n"
    "Quando o usuário pedir SKUs que aparecem em DUAS OU MAIS tabelas:\n"
    "→ USE pandas_analysis com analysis_type='intersection'\n"
    "→ NUNCA use sql_filter múltiplas vezes\n\n"

    "EXEMPLO DE USO CORRETO:\n"
    "Pergunta: 'SKUs da loja 022 com ajuste de estoque que também aparecem em inventário saída'\n"
    "Resposta correta:\n"
    "pandas_analysis(\n"
    "  tables=['ajustes_estoque', 'inventario_saida'],\n"
    "  analysis_type='intersection',\n"
    "  filters={'ajustes_estoque': {'loja': '022'}, 'inventario_saida': {'loja': '022'}}\n"
    ")\n\n"

    "FERRAMENTAS PARA ANÁLISE ÚNICA (use apenas quando NÃO for cruzamento):\n"
    "• list_tables - Listar tabelas disponíveis no banco\n"
    "• sql_filter - Filtrar dados de UMA tabela específica\n"
    "• sql_aggregate - Agregar/agrupar dados de UMA tabela (COUNT, SUM, AVG)\n"
    "• sql_head - Visualizar primeiras linhas de UMA tabela\n"
    "• sql_count - Contar registros em UMA tabela\n\n"

    "FERRAMENTAS PARA CRUZAMENTO (use para detectar fraudes):\n"
    "• sku_tre_table - PRINCIPAL: cruza Ajuste + Inventário + Troca (3 tabelas)\n"
    "• pandas_analysis - Análise avançada com cruzamento de 2 tabelas\n"
    "• sku_intersection - Interseção de SKUs entre duas tabelas\n\n"

    "PADRÕES DE FRAUDE A DETECTAR\n\n"

    "1. SKU com ajuste de estoque seguido de cancelamento (possível fraude)\n"
    "2. SKU com múltiplos ajustes em curto período (suspeito)\n"
    "3. SKU que aparece em inventário saída E troca no mesmo dia (inconsistência)\n"
    "4. Usuário com alta frequência de ajustes/cancelamentos (padrão anômalo)\n"
    "5. Loja com volume atípico de movimentações\n\n"

    "Sempre que identificar padrões suspeitos, destaque-os na resposta com emojis:\n"
    "🚨 = Fraude altamente provável\n"
    "⚠️ = Padrão suspeito que requer investigação\n"
    "ℹ️ = Informação relevante\n"
    ),
    tools=[
        sku_tre_table_tool,
        pandas_analysis_tool,
        sku_intersection_tool,
        sql_aggregate_tool,
        sql_head_tool,
        sql_count_tool,
        list_tables_tool,  # Ferramenta para listar tabelas do PostgreSQL
    ],
    model=_make_model(PRIMARY_MODEL, "primary")
)

# =========================
# Chat JSON + Chat HTML + UI
# =========================
def render_md(text: str) -> str:
    return markdown.markdown(
        text or "",
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"]
    )

class ChatIn(BaseModel):
    message: str

def _detect_cross_table_query(msg: str) -> bool:
    """Detecta se a pergunta requer cruzamento de tabelas"""
    msg_lower = msg.lower()

    # Padrões que indicam cruzamento
    cross_patterns = [
        r'aparecem?\s+(em|na|no)',
        r'que\s+est[aã]o\s+(em|na|no)',
        r'present(es?)?\s+(em|na|no)',
        r'tamb[ée]m\s+(em|na|no)',
        r'e\s+(em|na|no)\s+\w+',
        r'com\s+\w+\s+e\s+\w+',
    ]

    # Verifica se menciona múltiplas tabelas
    tables = ['ajustes', 'devolucao', 'devolu[cç][aã]o', 'cancelamento', 'inventario', 'invent[aá]rio']
    table_count = sum(1 for t in tables if re.search(t, msg_lower))

    # Verifica padrões de cruzamento
    has_cross_pattern = any(re.search(p, msg_lower) for p in cross_patterns)

    return table_count >= 2 or has_cross_pattern

@app.post("/chat")
def chat(in_: ChatIn, response: Response):
    try:
        raw = (in_.message or "").strip()
        forced, msg = _extract_forced_provider(raw)

        # Detecta se é query de cruzamento e adiciona hint
        if _detect_cross_table_query(msg):
            msg = f"[CRUZAMENTO DE TABELAS DETECTADO] {msg}"

        # 1) smalltalk/help
        if _is_smalltalk(msg):
            _set_resp_headers(response, provider="none", model_id="smalltalk", path="fastpath")
            return {"answer": _help_reply_html()}

        # 2) fast-path determinístico
        md = fastpath_markdown(msg)
        if md is not None:
            _set_resp_headers(response, provider="none", model_id="fastpath", path="fastpath")
            return {"answer": md}

        # 3) planner (opcional)
        mode, payload = ("delegate_outside", None)
        if 'USE_PLANNER' in globals() and USE_PLANNER:
            res = planner_route(msg)
            if isinstance(res, (tuple, list)) and len(res) == 2:
                mode, payload = res

        if mode == "tool" and payload:
            _set_resp_headers(response, provider="free-or-primary", model_id="planner", path="planner->tool")
            return f"<div class='reply'>{payload}</div>"
# --- delegate_outside + autoexec (JSON) ---
        if mode == "delegate_outside" and SEMANTIC_FALLBACK:
            out = _answer_outside_tools(msg)
            _set_resp_headers(response, provider="primary", model_id=PRIMARY_MODEL, path="planner->outside")

            cmds = _extract_run_commands(out)

            explanation_md = re.sub(r"\[\[RUN\]\].+?\[\[/RUN\]\]", "", out, flags=re.S)
            html_expl = render_md(explanation_md)

            if OUTSIDE_AUTOEXEC and cmds:
                exec_blocks = []
                for cmd in cmds:
                    html_exec = fastpath_markdown(cmd)
                    if not html_exec:
                        mode2, payload2 = planner_route(cmd)
                        if mode2 == "tool" and payload2:
                            html_exec = payload2
                    if not html_exec:
                        html_exec = "<div class='muted'>Comando sugerido não reconhecido para execução automática.</div>"
                    exec_blocks.append(html_exec)

                joined = "<hr/>".join(exec_blocks)
                return {
                    "answer": (
                        f"<div class='reply'>{html_expl}"
                        f"<div class='muted' style='margin-top:8px'>Execução automática (outside):</div>"
                        f"{joined}</div>"
                    )
                }

            return {"answer": f"<div class='reply'>{html_expl}</div>"}


        # 4) agente principal com roteamento + fallback
        provider, model_id = _route_model(msg)
        if forced == "free" and FREE_MODEL_ID:
            provider, model_id = "free", FREE_MODEL_ID
        elif forced == "primary":
            provider, model_id = "primary", PRIMARY_MODEL

        agent.model = _make_model(model_id, provider)
        ok, out, err, tools_used = _run_with_retry(agent, msg)

        if (not ok or _needs_fallback(out)) and provider != "primary":
            agent.model = _make_model(PRIMARY_MODEL, "primary")
            ok2, out2, err2, tools_used2 = _run_with_retry(agent, msg)
            _set_resp_headers(response, provider="primary", model_id=PRIMARY_MODEL, path="agent->fallback", fallback=True, tools_used=tools_used2)
            return {"answer": (out2 or out or f"Erro: {err2 or err}")}

        _set_resp_headers(response, provider=provider, model_id=model_id, path="agent", tools_used=tools_used)
        return {"answer": out}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_html", response_class=HTMLResponse)
@app.post("/chat_html", response_class=HTMLResponse)
def chat_html(message: str | None = None, in_: ChatIn | None = None, response:Response = None):
    raw = (message or (in_.message if in_ else "") or "").strip()
    forced, msg = _extract_forced_provider(raw)

    # 1) smalltalk/help
    if _is_smalltalk(msg):
        _set_resp_headers(response, provider="none", model_id="smalltalk", path="fastpath")
        return f"<div class='reply'>{_help_reply_html()}</div>"

    # 2) fast-path determinístico
    fast = fastpath_markdown(msg)
    if fast is not None:
        _set_resp_headers(response, provider="none", model_id="fastpath", path="fastpath")
        if fast.lstrip().startswith("<"):
            return f"<div class='reply'>{fast}</div>"
        return f"<div class='reply'>{render_md(fast)}</div>"

    # 3) planner (opcional)
    mode, payload = ("delegate_outside", None)
    if 'USE_PLANNER' in globals() and USE_PLANNER:
        res = planner_route(msg)
        if isinstance(res, (tuple, list)) and len(res) == 2:
            mode, payload = res

        if mode == "tool" and payload:
            _set_resp_headers(response, provider="free-or-primary", model_id="planner", path="planner->tool")
            return f"<div class='reply'>{payload}</div>"

# --- delegate_outside + autoexec (HTML) ---
        if mode == "delegate_outside" and SEMANTIC_FALLBACK:
            out = _answer_outside_tools(msg)
            _set_resp_headers(response, provider="primary", model_id=PRIMARY_MODEL, path="planner->outside")

            cmds = _extract_run_commands(out)

            explanation_md = re.sub(r"\[\[RUN\]\].+?\[\[/RUN\]\]", "", out, flags=re.S)
            html_expl = render_md(explanation_md)

            if OUTSIDE_AUTOEXEC and cmds:
                exec_blocks = []
                for cmd in cmds:
                    html_exec = fastpath_markdown(cmd)
                    if not html_exec:
                        mode2, payload2 = planner_route(cmd)
                        if mode2 == "tool" and payload2:
                            html_exec = payload2
                    if not html_exec:
                        html_exec = "<div class='muted'>Comando sugerido não reconhecido para execução automática.</div>"
                    exec_blocks.append(html_exec)

                joined = "<hr/>".join(exec_blocks)
                return (
                    f"<div class='reply'>{html_expl}"
                    f"<div class='muted' style='margin-top:8px'>Execução automática (outside):</div>"
                    f"{joined}</div>"
                )

            return f"<div class='reply'>{html_expl}</div>"

    # 4) agente principal com roteamento + fallback
    provider, model_id = _route_model(msg)
    if forced == "free" and FREE_MODEL_ID:
        provider, model_id = "free", FREE_MODEL_ID
    elif forced == "primary":
        provider, model_id = "primary", PRIMARY_MODEL

    agent.model = _make_model(model_id, provider)
    ok, out, err, tools_used = _run_with_retry(agent, msg)

    if (not ok or _needs_fallback(out)) and provider != "primary":
        agent.model = _make_model(PRIMARY_MODEL, "primary")
        ok2, out2, err2, tools_used2 = _run_with_retry(agent, msg)
        final = out2 if (ok2 and out2) else (out or f"Erro: {err2 or err}")
        _set_resp_headers(response, provider="primary", model_id=PRIMARY_MODEL, path="agent->fallback", fallback=True, tools_used=tools_used2)
    else:
        final = out or (f"Erro: {err}" if err else "")
        _set_resp_headers(response, provider=provider, model_id=model_id, path="agent", tools_used=tools_used)

    return f"<div class='reply'>{render_md(final)}</div>"

# =========================
# UI Web simples
# =========================
@app.get("/", response_class=HTMLResponse)
def chat_ui():
    return HTMLResponse(
        """
<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinela • Chat</title>
<style>
  :root { --bg:#0b1220; --card:#111a2b; --in:#0d1b31; --txt:#e8eefc; --muted:#9bb0d3; --accent:#3b82f6; }
  *{box-sizing:border-box} body{margin:0;background:var(--bg);font:16px/1.45 system-ui,Segoe UI,Roboto,Arial;color:var(--txt);}
  .wrap{max-width:860px;margin:0 auto;padding:24px}
  .title{font-weight:700;font-size:20px;margin:4px 0 16px;letter-spacing:.2px}
  .chat{display:flex;flex-direction:column;gap:12px}
  .msg{padding:14px 16px;border-radius:14px;max-width:85%;white-space:pre-wrap}
  .user{align-self:flex-end;background:#173159}
  .bot{align-self:flex-start;background:var(--card)}
  .muted{color:var(--muted);font-size:13px}
  .row{display:flex;gap:8px;margin-top:12px}
  input[type=text]{flex:1;padding:12px 14px;border-radius:12px;border:1px solid #26344d;background:var(--in);color:var(--txt);outline:none}
  button{padding:12px 16px;border-radius:12px;border:0;background:var(--accent);color:white;font-weight:600;cursor:pointer}
  button[disabled]{opacity:.6;cursor:not-allowed}

  .reply table{
    border-collapse: collapse;
    width:100%;
    display:block;
    overflow-x:auto;
    white-space:nowrap;
    border:1px solid #21314f;
    border-radius:8px;
  }
  .reply table th, .reply table td{
    border-bottom:1px solid #21314f;
    padding:8px 10px;
    text-align:left;
  }
  .reply table thead th{
    position:sticky;
    top:0;
    background:#0e223e;
    z-index:1;
  }
  .reply table tbody tr:nth-child(even){
    background:#0c1a30;
  }
  .reply code{background:#0e223e;padding:2px 6px;border-radius:6px}
  .typing{align-self:flex-start;color:var(--muted);font-style:italic}
  .metrics{
    position: fixed; top: 12px; right: 12px;
    background: var(--card); border:1px solid #26344d;
    padding: 6px 10px; border-radius: 10px; font-size:12px;
    color: var(--muted); display:flex; align-items:center; gap:8px;
    z-index: 50;
  }
  .metrics .dot{width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block}
  .metrics .dot.warn{background:#f59e0b}
  .metrics .dot.err{background:#ef4444}
  .metrics b{color:#e8eefc}
  .msg .badge{
    margin-top:6px;
    font-size:12px;
    color: var(--muted);
    opacity:.9;
  }
  .msg.bot .badge{ text-align:right; }
</style>
</head>
<body>
 <div id="metrics" class="metrics" title="Métricas em tempo real">carregando…</div>
  <div class="wrap">
    <div class="title">🛡️ Sentinela — Chat</div>
    <div id="chat" class="chat"></div>

    <div class="row">
      <input id="msg" type="text" placeholder="Ex.: mostre vendas_canceladas com 5 linhas" autofocus />
      <button id="send">Enviar</button>
    </div>
    <div class="muted" style="margin-top:8px">Dica: faça upload em <code>/docs → POST /upload_csv</code> e pergunte pela tabela criada.</div>
  </div>

<script>
  const chatEl = document.getElementById('chat');
  const msgEl  = document.getElementById('msg');
  const btnEl  = document.getElementById('send');

  // 🔹 SID GLOBAL (antes de qualquer fetch)
  const SID_KEY = 'sentinela_sid';
  let SID = localStorage.getItem(SID_KEY);
  if(!SID){
    SID = (crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random()).slice(2);
    localStorage.setItem(SID_KEY, SID);
  }

  function addBubble(text, who='bot', isHTML=false, meta=null){
    const div = document.createElement('div');
    div.className = 'msg ' + (who==='user' ? 'user' : 'bot');
    if(isHTML){ div.innerHTML = text; } else { div.textContent = text; }

    if(meta){
      const m = document.createElement('div');
      m.className = 'badge';
      m.textContent = meta;
      div.appendChild(m);
    }
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function addTyping(){
    const d = document.createElement('div');
    d.className = 'msg typing';
    d.textContent = 'Sentinela está pensando...';
    chatEl.appendChild(d);
    chatEl.scrollTop = chatEl.scrollHeight;
    return d;
  }

  async function send(){
    const text = msgEl.value.trim();
    if(!text) return;
    addBubble(text,'user',false);
    msgEl.value = '';
    btnEl.disabled = true;

    const typing = addTyping();
    try{
      const res = await fetch('/chat_html', {
        method:'POST',
        headers:{
          'Content-Type':'application/json',
          'X-Client-Id': SID
        },
        body: JSON.stringify({message: text})
      });
      const provider = res.headers.get('X-Model-Provider') || '';
      const modelId  = res.headers.get('X-Model-Id') || '';
      const path     = res.headers.get('X-Responder-Path') || '';
      const toolsUsed = res.headers.get('X-Tools-Used') || '';
      const fb       = res.headers.get('X-Model-Fallback') ? ' (fallback)' : '';
      let meta       = [path, provider, modelId].filter(Boolean).join(' • ') + fb;
      if (toolsUsed) {
        meta += '\\nTools used → ' + toolsUsed;
      }

      const html = await res.text();
      typing.remove();
      addBubble(html, 'bot', true, meta);

    }catch(e){
      typing.remove();
      addBubble('Erro ao falar com o servidor: ' + e, 'bot', false);
    }finally{
      btnEl.disabled = false;
      msgEl.focus();
    }
  }

  btnEl.addEventListener('click', send);
  msgEl.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ e.preventDefault(); send(); }});

  // Mensagem de boas-vindas
  addBubble('Tente perguntar: "loja 022 quais os sku de ajustes_estoque que aparecem em inventario_saida?".', 'bot', false);

  // 🔹 Badge de métricas
  const metricsEl = document.getElementById('metrics');

  function fmt(n){ try{ return Number(n).toLocaleString('pt-BR'); }catch(_){ return n; } }

  async function pollStats(){
    try{
      const r = await fetch('/stats', { headers: {'X-Client-Id': SID} });
      const j = await r.json();
      const now = j.active_requests_now || 0;
      const act = j.active_sessions_last_minutes || 0;
      const winMin = Math.round((j.window_seconds || 300) / 60);
      const cls = now > 5 ? 'dot err' : (now > 1 ? 'dot warn' : 'dot');
      metricsEl.innerHTML = `
        <span class="${cls}"></span>
        agora: <b>${fmt(now)}</b>
        &nbsp;•&nbsp;
        ativos(${winMin}min): <b>${fmt(act)}</b>
      `;
    }catch(e){
      metricsEl.innerHTML = `<span class="dot err"></span> métricas indisponíveis`;
    }
  }

  setInterval(pollStats, 5000);
  pollStats();
</script>
</body>
</html>
        """
    )

# inicializa catálogo na carga do módulo
# rebuild_catalog()  # Comentado - será executado no lifespan do app
