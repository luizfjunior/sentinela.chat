# agent_app.py
# Sentinela – Chat + Tools (SQLite/CSV) com planner, roteamento e UI
# Requisitos principais: fastapi, uvicorn, agno, pydantic, pandas, numpy, markdown, starlette

import os, traceback
from typing import Any, Dict, Optional, Literal
import numpy as np
import pandas as pd
import re, unicodedata
import json
import requests
import calendar
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Response
from fastapi.responses import HTMLResponse
import time
from starlette.middleware.base import BaseHTTPMiddleware

from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, ValidationError
import html

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
import sqlite3, tempfile, shutil, pathlib
import markdown

# =========================
# App & rotas básicas
# =========================
app = FastAPI(title="Sentinela")

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
# Constantes
# =========================
DATA_DIR = os.environ.get("DATA_DIR", "data")
DB_PATH  = os.path.join(DATA_DIR, "sentinela.db")
USE_PLANNER = True
OUTSIDE_AUTOEXEC = os.environ.get("OUTSIDE_AUTOEXEC", "on").lower() in {"on","1","true","yes"}
RUN_OPEN = "[[RUN]]"
RUN_CLOSE = "[[/RUN]]"
# ===== Fallback fora de tools =====
SEMANTIC_FALLBACK = os.environ.get("SEMANTIC_FALLBACK", "on").lower() in {"on","1","true","yes"}
OUT_OF_TOOL_MAXTOKENS = int(os.environ.get("OUT_OF_TOOL_MAXTOKENS", "5000"))

# ===== Model routing (planner barato + fallback) =====
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "gpt-4o-mini").strip()
FREE_MODEL_ID = os.environ.get("FREE_MODEL_ID", "").strip()  # ex.: "llama-3.1-8b-instant"
FREE_API_BASE = os.environ.get("FREE_API_BASE", "").strip()  # ex.: "https://api.groq.com/openai/v1"
FREE_API_KEY  = os.environ.get("FREE_API_KEY", "").strip()   # ex.: sua GROQ_API_KEY

MODEL_KW = dict(temperature=0.0)  # zero criatividade; queremos determinismo

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
            return True, (resp.content or ""), None
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if any(x in msg for x in ["rate limit", "429", "temporarily unavailable", "timeout", "service unavailable", "overloaded"]):
                time.sleep(0.4)
                continue
            break
    return False, "", last_err

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
def _set_resp_headers(resp: Response | None, *, provider: str, model_id: str, path: str, fallback: bool=False):
    if resp is None:
        return
    resp.headers["X-Model-Provider"] = provider or ""
    resp.headers["X-Model-Id"] = model_id or ""
    resp.headers["X-Responder-Path"] = path or ""
    if fallback:
        resp.headers["X-Model-Fallback"] = "primary"

def _set_router_debug_headers(resp: Response, *, msg: str, provider: str):
    try:
        intent = _intent_hint(msg)
    except Exception:
        intent = "error"
    resp.headers["X-Router-Intent"] = intent
    resp.headers["X-Free-Enabled"] = "1" if bool(FREE_MODEL_ID) else "0"
    resp.headers["X-Forced-Provider"] = provider or ""

def head_csv_fn(filename: str, n: int = 5):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {"status": "error", "message": f"Arquivo não encontrado: {filename}"}

    def try_read():
        try:
            return pd.read_csv(
                path, sep=None, engine="python", encoding="latin-1",
                nrows=max(5, n), on_bad_lines="skip"
            )
        except Exception:
            for sep in [";", ",", "\t", "|"]:
                try:
                    return pd.read_csv(
                        path, sep=sep, engine="python", encoding="latin-1",
                        nrows=max(5, n), on_bad_lines="skip"
                    )
                except Exception:
                    continue
            raise

    try:
        df = try_read()

        def to_py(v):
            if pd.isna(v): return None
            if isinstance(v, np.integer):  return int(v)
            if isinstance(v, np.floating): return float(v)
            if isinstance(v, np.bool_):    return bool(v)
            return v

        preview = df.iloc[:n].applymap(to_py).to_dict(orient="records")
        cols = [str(c) for c in df.columns.tolist()]
        return {"status": "ok", "columns": cols, "preview": preview}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}

ALLOWED_TOOLS = {"sql_head", "sql_filter", "sql_aggregate", "sku_intersection", "head_csv", "list_csvs","none", "sql_count"}
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

    "sql_list_tables": "list_csvs",
    "list_tables": "list_csvs",
    "tables": "list_csvs",

    "sku_intersect": "sku_intersection",
    "sku_overlap": "sku_intersection",
    "count": "sql_count",
    "contar": "sql_count",
    "conte": "sql_count",
    "quantas_vezes": "sql_count",
    "ocorrencias": "sql_count",
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
# SQLite helpers + upload
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

def _sqlite_exec(q: str, params: list | tuple = ()):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(q, params)
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    finally:
        con.close()

def _sqlite_cols(table: str) -> list[str]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
        return cols
    finally:
        con.close()

def _table_exists(table: str) -> bool:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        ok = con.execute(
            'SELECT 1 FROM sqlite_master WHERE type="table" AND name=? LIMIT 1', (table,)
        ).fetchone() is not None
        return ok
    finally:
        con.close()

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

def _import_csv_to_sqlite(temp_csv_path: str, table: str) -> dict:
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
    """Recebe um CSV (multipart/form-data) e ingere em SQLite (data/sentinela.db)."""
    if not file.filename.lower().endswith(".csv"):
        return {"status": "error", "message": "Envie um arquivo .csv"}

    safe_table = _safe_table_name(table or pathlib.Path(file.filename).stem)

    os.makedirs(DATA_DIR, exist_ok=True)

    # salva upload em arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        info = _import_csv_to_sqlite(tmp_path, safe_table)
        rebuild_catalog()
        return {"status": "ok", "db": DB_PATH, "table": safe_table, **info}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _safe_table_name(stem: str) -> str:
    name = re.sub(r'[^A-Za-z0-9_]+', '_', stem).strip('_').lower()
    return name or "tabela"

def ingest_all_fn(recursive: bool = False):
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
            info = _import_csv_to_sqlite(path, table)
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
    global CATALOG
    CATALOG = {}
    tabs = _sqlite_exec("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    for r in tabs:
        tname = r["name"]
        cols = [c["name"] for c in _sqlite_exec(f'PRAGMA table_info("{tname}")')]
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

    cols = set(_sqlite_cols(table))
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
    cols = _sqlite_cols(table)
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
        "  \"tool\": \"sql_head|sql_filter|sql_aggregate|sku_intersection|sql_count|list_csvs|null\","

        "  \"params\": { ... },"
        "  \"rewrite\": \"mensagem reescrita para o agente responder sem consultar DB\","
        "  \"confidence\": 0.0"
        "}\n\n"
        "Regras:\n"
        "- Use mode \"tool\" quando o usuário quer números, contagens, filtros, amostras, agregações ou cruzar tabelas.\n"
        "- Use mode \"answer\" para perguntas conceituais/explicativas. NÃO invente números; apenas reescreva a pergunta (rewrite) já normalizada (datas AAAA-MM-DD, loja sem zeros à esquerda, termos claros).\n"
        "- Use mode \"delegate\" se estiver incerto.\n"
        "- Se a pergunta tiver meses PT-BR (janeiro..dezembro), converta para start/end AAAA-MM-DD.\n"
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

        ok, out, _err = _run_with_retry(planner, prompt)
        if (not ok) or (not out or not out.strip()):
            if provider != "primary":
                planner = Agent(
                    name="planner",
                    system_message="Retorne APENAS JSON válido, sem texto extra.",
                    model=_make_model(PRIMARY_MODEL, "primary"),
                )
                ok, out, _err = _run_with_retry(planner, prompt)
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
        "Você é um ANALISTA & SOLUCIONADOR do Sentinela.\n"
        "Regras:\n"
        "1) Se a pergunta for conceitual, responda de forma clara e objetiva.\n"
        "2) Se exigir dados (contar/somar/filtrar/agrupar/período), NÃO invente números.\n"
        "   Em vez disso, proponha EXATAMENTE UM comando válido do Sentinela, "
        "   dentro das tags [[RUN]] ... [[/RUN]]. Sem explicações DENTRO das tags.\n"
        "   Exemplos válidos: \n"
        "     [[RUN]]mostre cancelamento_2025 com 5 linhas[[/RUN]]\n"
        "     [[RUN]]filtre LOJA=017 e SKU=999999992513081 em inventario_saida_042025 limite 20[[/RUN]]\n"
        "     [[RUN]]agregue em devolucao por SKU somando VALORBRUTO entre 2025-01-01 e 2025-06-30 top 20[[/RUN]]\n"
        "3) Depois das tags, explique brevemente o que o comando faz e como interpretar o resultado.\n"
        "4) Considere problemas de codificação/acentos (ex.: 'Sa?a' vs 'Saída'); quando filtrar texto, prefira icontains.\n"
        "5) Use TABELAS e COLUNAS somente do catálogo a seguir (não invente nomes).\n"
        f"Catálogo enxuto: {json.dumps(schema, ensure_ascii=False)}\n"
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
    ok, out, err = _run_with_retry(tmp_agent, prompt)
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

    q = f'SELECT COUNT(*) AS total FROM "{table}"'
    if where_sql:
        q += f" WHERE {where_sql}"
    rows = _sqlite_exec(q, params)
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
    rows = _sqlite_exec(f'SELECT * FROM "{table}" LIMIT ?', [max(1, int(n))])
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

    cols_all = set(_sqlite_cols(table))
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
        f'FROM "{table}"'
    )
    if where_sql:
        q += where_sql
    q += f" GROUP BY {', '.join(group_parts)} ORDER BY valor DESC LIMIT ?"
    params.append(max(1, int(top)))

    rows = _sqlite_exec(q, params)
    return {"status": "ok", "rows": rows, "meta": {"table": table, "by": out_cols, "op": op, "value": value, "start": start, "end": end}}

def _guess_cols(table: str):
    cols = _sqlite_cols(table)

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
      FROM "{ajustes}"
      {"WHERE " + a_where_sql if a_where_sql else ""}
    ),
    d_f AS (
      SELECT {d_sku} AS SKU
      FROM "{devol}"
      {"WHERE " + d_where_sql if d_where_sql else ""}
    ),
    inter AS (
      SELECT DISTINCT a_f.SKU
      FROM a_f JOIN d_f ON a_f.SKU = d_f.SKU
    )
    SELECT SKU FROM inter LIMIT ?
    """
    rows = _sqlite_exec(q, a_params + d_params + [max(1, int(limit))])

    diag_q = f"""
      SELECT 
        (SELECT COUNT(*) FROM a_f) AS ajustes_filtrados,
        (SELECT COUNT(*) FROM d_f) AS devolucoes_filtradas,
        (SELECT COUNT(*) FROM inter) AS intersecao
      FROM (SELECT 1)
    """
    diag = _sqlite_exec(
        f"""
        WITH a_f AS (
          SELECT {a_sku} AS SKU FROM "{ajustes}" {"WHERE " + a_where_sql if a_where_sql else ""}
        ),
        d_f AS (
          SELECT {d_sku} AS SKU FROM "{devol}" {"WHERE " + d_where_sql if d_where_sql else ""}
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
    cols = _sqlite_cols(table)
    m = {c.lower(): c for c in cols}
    return m.get(name.lower())

def _is_digits_str(x) -> bool:
    s = str(x)
    return re.fullmatch(r"0*\d+", s) is not None

def _norm_numstr_sql(expr: str) -> str:
    return (
        f"(CASE "
        f"  WHEN {expr} GLOB '[0-9]*' "
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
                clauses.append(f"{norm_col} = (CASE WHEN ltrim(?, '0')='' THEN '0' ELSE ltrim(?, '0') END)")
                params.extend([str(val), str(val)])
            else:
                clauses.append(f'"{col}" = ?'); params.append(val)

        elif op == "ne":
            if _is_digits_str(val):
                clauses.append(f"{norm_col} != (CASE WHEN ltrim(?, '0')='' THEN '0' ELSE ltrim(?, '0') END)")
                params.extend([str(val), str(val)])
            else:
                clauses.append(f'"{col}" != ?'); params.append(val)

        elif op in {"gt","gte","lt","lte"}:
            cast = f'CAST("{col}" AS REAL)'
            sym = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[op]
            if isinstance(val, str):
                try: val = float(val.replace(",", "."))
                except: pass
            clauses.append(f"{cast} {sym} ?"); params.append(val)

        elif op == "icontains":
            # normaliza o valor de busca para minúsculas
            val_norm = str(val).lower()
            # citação da coluna sem f-string aninhada (evita SyntaxError)
            col_quoted = f'"{col}"'
            col_norm = f"lower({col_quoted})"
            clauses.append(f"{col_norm} LIKE ?")
            params.append(f"%{val_norm}%")

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
            clauses.append(f"({cast} BETWEEN ? AND ?)"); params.extend([lo, hi])

        elif op == "date_between":
            expr = _date_expr_auto(col)
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(f"date_between espera [start, end] em {key}")
            start, end = val
            clauses.append(f"(date({expr}) BETWEEN date(?) AND date(?))"); params.extend([start, end])

        elif op == "contains":
            clauses.append(f'"{col}" LIKE ?'); params.append(f"%{val}%")

        elif op == "in":
            seq = list(val) if isinstance(val, (list, tuple, set)) else []
            if not seq:
                clauses.append("1=0")
            elif all(_is_digits_str(v) for v in seq):
                ph = ",".join(["(CASE WHEN ltrim(?, '0')='' THEN '0' ELSE ltrim(?, '0') END)"] * len(seq))
                clauses.append(f"{norm_col} IN ({ph})")
                for v in seq:
                    params.extend([str(v), str(v)])
            else:
                ph = ",".join(["?"] * len(seq))
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

    q = f'SELECT * FROM "{table}"'
    if where_sql:
        q += f" WHERE {where_sql}"
    if ob_col:
        q += f' ORDER BY "{ob_col}" {ord_kw}'
    q += " LIMIT ? OFFSET ?"

    rows = _sqlite_exec(q, params + [max(1, int(limit)), max(0, int(offset))])
    if not rows and where:
        # testa cada condição isoladamente para apontar a "culpada"
        diag = {}
        for k,v in where.items():
            try:
                ws, ps = _build_where(table, {k:v})
                cnt = _sqlite_exec(f'SELECT COUNT(*) AS c FROM "{table}" WHERE {ws}', ps)[0]["c"]
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
    ajustes: str | None = Query(None, description="ex.: ajustes_estoque_2025 (opcional)"),
    devol:   str | None = Query(None, description="ex.: devolucao (opcional)"),
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

head_csv_tool = tool(
    name="head_csv",
    description="Mostra colunas e as primeiras linhas de um CSV da pasta data/."
)(head_csv_fn)

list_csvs_tool = tool(
    name="list_csvs",
    description="Lista os arquivos CSV disponíveis em data/."
)(list_csvs_fn)

sql_head_tool = tool(
    name="sql_head",
    description="Mostra as primeiras N linhas de uma tabela do SQLite (data/sentinela.db)."
)(sql_head_fn)

sql_aggregate_tool = tool(
    name="sql_aggregate",
    description="Agrega valores por categoria na tabela do SQLite. op: sum/count/avg/min/max."
)(sql_aggregate_fn)

sql_filter_tool = tool(
    name="sql_filter",
    description="Filtra linhas no SQLite com operadores: eq, ne, gt, gte, lt, lte, in, contains, icontains."
)(sql_filter_fn)

sql_count_tool = tool(
    name="sql_count",
    description="Conta linhas em uma tabela do SQLite aplicando filtros (where)."
)(sql_count_fn)

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
            return "<div class='muted'>Preciso do nome da tabela. Ex.: <code>mostre vendas_2025 com 5 linhas</code></div>"
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
        "<li><code>mostre cancelamento_2025 com 5 linhas</code></li>"
        "<li><code>filtre SKU=10220110944410 e LOJA=333 em ajustes_estoque_2025 limite 20</code></li>"
        "<li><code>agregue em cancelamento_2025 por LOJA somando VALORBRUTO top 5</code></li>"
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
                return f"<h3>Filtro em <code>{html.escape(table)}</code> (SQLite)</h3>{res['html']}"
            data = res.get("data") or res.get("rows") or []
            html_table = _rows_to_html(data)
            return f"<h3>Filtro em <code>{html.escape(table)}</code> (SQLite)</h3>{html_table}"


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
        "Você é o Sentinela. Sempre use ferramentas para obter dados (SQLite/CSV) e nunca invente. "
        "Priorize SQLite sempre que existir tabela equivalente. "
        "Escolha a ferramenta pelo tipo de intenção:\n"
        "• Visualizar primeiras linhas de uma TABELA → sql_head.\n"
        "• Filtrar por colunas/condições → sql_filter.\n"
        "• Agregar (sum, count, avg, min, max) → sql_aggregate.\n"
        "• Descobrir SKUs presentes em DUAS TABELAS (mesmos filtros) → sku_intersection.\n"
        "• Contagens simples (ex.: “LOJA=333 aparece quantas vezes”) → sql_count.\n"
        "Converta períodos de tempo para datas AAAA-MM-DD quando necessário, normalize códigos de loja (remova zeros à esquerda) "
    ),
    tools=[head_csv_tool, list_csvs_tool, sql_head_tool, sql_aggregate_tool, sql_filter_tool, sku_intersection_tool, sql_count_tool,  sql_count_tool],
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

@app.post("/chat")
def chat(in_: ChatIn, response: Response):
    try:
        raw = (in_.message or "").strip()
        forced, msg = _extract_forced_provider(raw)

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
        ok, out, err = _run_with_retry(agent, msg)

        if (not ok or _needs_fallback(out)) and provider != "primary":
            agent.model = _make_model(PRIMARY_MODEL, "primary")
            ok2, out2, err2 = _run_with_retry(agent, msg)
            _set_resp_headers(response, provider="primary", model_id=PRIMARY_MODEL, path="agent->fallback", fallback=True)
            return {"answer": (out2 or out or f"Erro: {err2 or err}")}

        _set_resp_headers(response, provider=provider, model_id=model_id, path="agent")
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
    ok, out, err = _run_with_retry(agent, msg)

    if (not ok or _needs_fallback(out)) and provider != "primary":
        agent.model = _make_model(PRIMARY_MODEL, "primary")
        ok2, out2, err2 = _run_with_retry(agent, msg)
        final = out2 if (ok2 and out2) else (out or f"Erro: {err2 or err}")
        _set_resp_headers(response, provider="primary", model_id=PRIMARY_MODEL, path="agent->fallback", fallback=True)
    else:
        final = out or (f"Erro: {err}" if err else "")
        _set_resp_headers(response, provider=provider, model_id=model_id, path="agent")

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
      <input id="msg" type="text" placeholder="Ex.: mostre cancelamento_2025 com 5 linhas" autofocus />
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
      const fb       = res.headers.get('X-Model-Fallback') ? ' (fallback)' : '';
      const meta     = [path, provider, modelId].filter(Boolean).join(' • ') + fb;

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
  addBubble('Oi! Eu sou o Sentinela. Depois de fazer upload em /docs → POST /upload_csv, pergunte: "mostre as 5 primeiras linhas de cancelamento_2025".', 'bot', false);

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
rebuild_catalog()
