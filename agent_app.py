import os, traceback
from typing import Optional
import numpy as np
import pandas as pd
import re, unicodedata
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse
import time
from starlette.middleware.base import BaseHTTPMiddleware

from contextlib import asynccontextmanager

from pydantic import BaseModel
import html


from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

import sqlite3, tempfile, shutil, pathlib
import markdown


# =========================
# App & rotas básicas
# =========================
app = FastAPI(title="Sentinela (mínimo)")

@app.get("/status")
def status():
    return {"status": "up"}

@app.get("/ping")
def ping():
    return {"ok": True}

def _missing_key_banner() -> str:
    return (
        "\n"
        "============================================================\n"
        "  ❌ OPENAI_API_KEY não definida\n"
        "  Defina a variável no terminal OU use o launcher recomendado:\n\n"
        "    Windows PowerShell:\n"
        '      $env:OPENAI_API_KEY = "SUA_CHAVE_AQUI"\n'
        "      python serve.py --host 0.0.0.0 --port 8000 --reload\n\n"
        "    Linux/macOS:\n"
        '      export OPENAI_API_KEY=\"SUA_CHAVE_AQUI\"\n'
        "      python serve.py --host 0.0.0.0 --port 8000 --reload\n\n"
        "  Links após iniciar:\n"
        "    • UI (Chat):   http://<host>:<port>/\n"
        "    • Swagger:     http://<host>:<port>/docs\n"
        "============================================================\n"
    )

@asynccontextmanager
async def app_lifespan(app):
    # STARTUP
    if not os.environ.get("OPENAI_API_KEY"):
        print(_missing_key_banner(), flush=True)
        # aborta o start pra ficar claro
        raise RuntimeError("OPENAI_API_KEY ausente")
    yield
app.router.lifespan_context = app_lifespan
ACTIVE_REQUESTS = 0                   # requisições sendo atendidas agora
SESSIONS_LAST_SEEN = {}               # sid -> epoch seconds
METRICS_STARTED_AT = time.time()
ACTIVE_WINDOW_SECONDS = int(os.environ.get("ACTIVE_WINDOW_SECONDS", "300"))  # 5 min

def _extract_client_id(request):
    # tenta header setado pelo front (recomendado), senão X-Forwarded-For (ngrok/proxy), senão IP
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

# =========================
# Constantes
# =========================
DATA_DIR = os.environ.get("DATA_DIR", "data")
DB_PATH  = os.path.join(DATA_DIR, "sentinela.db")


# =========================
# CSV (legacy) - funções + endpoints
# =========================
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
    # Lê primeiras linhas e escolhe o separador com maior ocorrência no header
    try:
        with open(file_path, "rb") as f:
            head = f.read(4096).decode("latin-1", "ignore")
        first = head.splitlines()[0] if head else ""
    except Exception:
        first = ""
    candidates = [";", ",", "\t", "|"]
    counts = {s: first.count(s) for s in candidates}
    # escolhe o que mais aparece; fallback para ';' e depois ','
    sep = max(counts, key=counts.get) if any(counts.values()) else ";"
    return sep

def _import_csv_to_sqlite(temp_csv_path: str, table: str) -> dict:
    """Importa CSV para SQLite com heurística de separador e em chunks."""
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
                dtype=str,            # evita quebrar SKU/zeros à esquerda
                chunksize=50_000,
            )

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

    safe_table = (table or pathlib.Path(file.filename).stem).lower()
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
    # nome seguro: minúsculas, só [a-z0-9_]
    name = re.sub(r'[^A-Za-z0-9_]+', '_', stem).strip('_').lower()
    return name or "tabela"

def ingest_all_fn(recursive: bool = False):
    """
    Converte todos os .csv de DATA_DIR para tabelas no SQLite (data/sentinela.db).
    A tabela recebe o nome do arquivo (sem extensão), normalizado.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    imported = []

    # escolhe iterador (raiz ou recursivo)
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

# endpoint manual (Swagger)
@app.post("/tool/ingest_all")
def call_ingest_all(recursive: bool = Query(False, description="Se true, percorre subpastas de data/")):
    res = ingest_all_fn(recursive=recursive)  # importa primeiro
    rebuild_catalog()                          # depois reconstrói o catálogo
    return res


# --- normalização / tokenização leves ---

_PT_STOP = {"de","da","do","das","dos","e","em","no","na","nos","nas","com","por","para","a","o","os","as"}
_SYNONYMS = {
    "estoque": {"estoque","stock"},
    "ajuste": {"ajuste","ajustes","ajust"},
    "devolucao": {"devolucao","devolucoes","devoluções","devol","troca","retorno","return"},
    "cancelamento": {"cancelamento","cancel","canc"},
    "inventario": {"inventario","inventário","invent"},
    "saida": {"saida","saída","out"},
    # conceitos úteis
    "sku": {"sku","produto","prod","codigo_produto","codigo"},
    "data": {"data","dt","datacancelamento","data_devolucao","data_devolucoes","datadev"},
    "loja": {"loja","filial","store"},
    "valor": {"valor","valorbruto","preco","preço","amount"},
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
    if re.search(r'\b(loja|filial|store)\b', c): tags.add("store")
    if re.search(r'\b(sku|produto|prod|codigo|codigo_produto)\b', c): tags.add("sku")
    if re.search(r'\b(data|dt)\b', c): tags.add("date")
    if re.search(r'\b(valor|preco|amount|bruto)\b', c): tags.add("amount")
    if re.search(r'\b(qtd|qtde|quant|movimentada|ajuste)\b', c): tags.add("quantity")
    if re.search(r'\b(ajuste|estoque|stock)\b', c): tags.add("adjust")
    if re.search(r'\b(devol)\b', c): tags.add("return")
    if re.search(r'\b(cancel)\b', c): tags.add("cancel")
    return tags

# 🔹 NOVO: tags a partir do nome da tabela
def _tags_from_tokens(tokens: set[str]) -> set[str]:
    tags = set()
    if tokens & _SYNONYMS.get("ajuste", set()):        tags.add("adjust")
    if tokens & _SYNONYMS.get("estoque", set()):       tags.add("adjust")
    if tokens & _SYNONYMS.get("devolucao", set()):     tags.add("return")
    if tokens & _SYNONYMS.get("cancelamento", set()):  tags.add("cancel")
    if tokens & _SYNONYMS.get("sku", set()):           tags.add("sku")
    if tokens & _SYNONYMS.get("data", set()):          tags.add("date")
    if tokens & _SYNONYMS.get("loja", set()):          tags.add("store")
    if tokens & _SYNONYMS.get("valor", set()):         tags.add("amount")
    return tags

CATALOG = {}  # table -> {"tokens","columns","col_tags","table_tags"}

def rebuild_catalog():
    global CATALOG
    CATALOG = {}
    tabs = _sqlite_exec("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    for r in tabs:
        tname = r["name"]
        cols = [c["name"] for c in _sqlite_exec(f'PRAGMA table_info("{tname}")')]
        t_tokens = _tokens(tname)
        col_tags = {c: _tag_of_column(c) for c in cols}
        name_tags = _tags_from_tokens(t_tokens)  # <- usa nome da tabela
        t_tags = (set().union(*col_tags.values()) if col_tags else set()) | name_tags
        CATALOG[tname] = {"tokens": t_tokens, "columns": cols, "col_tags": col_tags, "table_tags": t_tags}

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


# =========================
# SQL tools (consultas)
# =========================
def sql_head_fn(table: str, n: int = 5):
    if not _table_exists(table):
        return {"status": "error", "message": f"Tabela não encontrada: {table}"}
    rows = _sqlite_exec(f'SELECT * FROM "{table}" LIMIT ?', [max(1, int(n))])
    return {"status": "ok", "rows": rows}

def sql_aggregate_fn(table: str, by: str, value: str | None = None, op: str = "sum", top: int = 10):
    if not _table_exists(table):
        return {"status": "error", "message": f"Tabela não encontrada: {table}"}
    cols = set(_sqlite_cols(table))
    if by not in cols:
        return {"status": "error", "message": f"Coluna de agrupamento não existe: {by}"}

    op = op.lower()
    allowed = {"sum", "count", "avg", "min", "max"}
    if op not in allowed:
        return {"status": "error", "message": f"Operação não permitida: {op}"}

    if op == "count":
        agg_expr = "COUNT(*)"
    else:
        if value is None or value not in cols:
            return {"status": "error", "message": "Informe a coluna numérica em `value`"}
        agg_expr = f'{op}(CAST("{value}" AS REAL))'

    q = f'SELECT "{by}" AS chave, {agg_expr} AS valor FROM "{table}" GROUP BY "{by}" ORDER BY valor DESC LIMIT ?'
    rows = _sqlite_exec(q, [max(1, int(top))])
    return {"status": "ok", "rows": rows}

def _guess_cols(table: str):
    """Escolhe SKU / LOJA / DATA priorizando:
    (1) match exato normalizado, (2) match por palavra (com borda),
    (3) substring como último recurso. Evita confundir 'PRODUTO' com 'VALORVENDAPRODUTO' etc.
    """
    cols = _sqlite_cols(table)

    def norm(s: str) -> str:
        # normaliza para comparar ignorando _ e acentos/caixa
        return re.sub(r'[^a-z0-9]+', '', s.lower())

    def find_col(candidates: list[str]) -> str | None:
        # 1) match exato normalizado
        norm_map = {norm(c): c for c in cols}
        for cand in candidates:
            hit = norm_map.get(norm(cand))
            if hit:
                return hit

        # 2) match por palavra com bordas (evita pegar '...PRODUTO...' dentro de 'VALORVENDAPRODUTO')
        for c in cols:
            lc = re.sub(r'[_]+', ' ', c.lower())
            for cand in candidates:
                if re.search(r'\b' + re.escape(cand.lower()) + r'\b', lc):
                    return c

        # 3) fallback por substring
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

# 🔹 DATA AUTO: aceita AAAA-MM-DD (ISO) ou DD/MM/AAAA
def _date_expr_auto(col: str) -> str:
    # Se tiver '-', assume ISO e pega os 10 primeiros; senão, converte de DD/MM/AAAA
    return (
        f"CASE WHEN instr(\"{col}\", '-')=5 "  # posição do primeiro '-' em 'YYYY-'
        f"THEN substr(\"{col}\",1,10) "
        f"ELSE substr(\"{col}\",7,4)||'-'||substr(\"{col}\",4,2)||'-'||substr(\"{col}\",1,2) END"
    )

def sku_intersection_fn(
    ajustes: str|None = None,
    devol: str|None = None,
    loja: int|str|None = None,
    start: str|None = None,
    end: str|None = None,
    limit: int = 100,
):
  
    if not CATALOG:
        rebuild_catalog()

    if not ajustes:
        ajustes = next((t for t, info in CATALOG.items() if {"adjust", "sku"} <= info["table_tags"]), None)
    if not devol:
        devol   = next((t for t, info in CATALOG.items() if {"return", "sku"} <= info["table_tags"]), None)

    if not _table_exists(ajustes) or not _table_exists(devol):
        return {"status":"error","message":f"Tabelas não encontradas (ajustes={ajustes}, devol={devol})"}

    a = _guess_cols(ajustes)
    d = _guess_cols(devol)
    for tname, meta in (("ajustes", a), ("devol", d)):
        if not meta["sku"] or not meta["loja"] or not meta["data"]:
            return {"status":"error","message":f"Não consegui identificar SKU/LOJA/DATA na tabela {tname} ({' ,'.join(meta['all'])})."}

    # normaliza loja para inteiro (017 -> 17)
    try:
        loja_int = int(str(loja).lstrip("0") or "0")
    except:
        return {"status":"error","message":"Parâmetro `loja` inválido (use número, ex.: 17)."}

    a_date = _date_expr_auto(a["data"])
    d_date = _date_expr_auto(d["data"])

    q = f'''
    SELECT DISTINCT a."{a['sku']}" AS SKU
    FROM "{ajustes}" a
    JOIN "{devol}"   d
      ON a."{a['sku']}" = d."{d['sku']}"
    WHERE CAST(a."{a['loja']}" AS INT) = ?
      AND CAST(d."{d['loja']}" AS INT) = ?
      AND date({a_date}) BETWEEN date(?) AND date(?)
      AND date({d_date}) BETWEEN date(?) AND date(?)
    LIMIT ?
    '''
    rows = _sqlite_exec(q, [loja_int, loja_int, start, end, start, end, max(1,int(limit))])
    return {"status":"ok","rows":rows, "tables":{"ajustes":ajustes, "devol":devol}}

def _resolve_col(table: str, name: str) -> str | None:
    """Mapeia coluna case-insensitive para o nome real da tabela."""
    cols = _sqlite_cols(table)
    m = {c.lower(): c for c in cols}
    return m.get(name.lower())

def _build_where(table: str, where: dict) -> tuple[str, list]:
    clauses, params = [], []
    for key, val in (where or {}).items():
        # chave no formato: COL, COL__op (eq/ne/gt/gte/lt/lte/in/contains/icontains)
        if "__" in key:
            col_key, op = key.split("__", 1)
        else:
            col_key, op = key, "eq"
        col = _resolve_col(table, col_key)
        if not col:
            raise ValueError(f"Coluna não existe: {col_key}")

        op = op.lower()
        if op == "eq":
            clauses.append(f'"{col}" = ?'); params.append(val)
        elif op == "ne":
            clauses.append(f'"{col}" != ?'); params.append(val)
        elif op in {"gt","gte","lt","lte"}:
            # compara como numérico
            cast = f'CAST("{col}" AS REAL)'
            sym = { "gt": ">", "gte": ">=", "lt": "<", "lte": "<=" }[op]
            # aceita número com vírgula (pt-BR)
            if isinstance(val, str):
                try: val = float(val.replace(",", "."))
                except: pass
            clauses.append(f"{cast} {sym} ?"); params.append(val)
        elif op == "contains":
            clauses.append(f'"{col}" LIKE ?'); params.append(f"%{val}%")
        elif op == "icontains":
            clauses.append(f'LOWER("{col}") LIKE ?'); params.append(f"%{str(val).lower()}%")
        elif op == "in":
            if not isinstance(val, (list, tuple, set)) or len(val) == 0:
                clauses.append("1=0")  # vazio => sem resultados
            else:
                ph = ",".join(["?"] * len(val))
                clauses.append(f'"{col}" IN ({ph})'); params.extend(list(val))
        else:
            raise ValueError(f"Operador não suportado: {op}")
    where_sql = " AND ".join(clauses) if clauses else ""
    return where_sql, params

def sql_filter_fn(table: str, where: dict | None = None, limit: int = 100,
                  order_by: str | None = None, order: str = "desc"):
    if not _table_exists(table):
        return {"status":"error","message":f"Tabela não encontrada: {table}"}

    where = where or {}
    try:
        where_sql, params = _build_where(table, where)
    except ValueError as e:
        return {"status":"error","message":str(e)}

    # order_by seguro (checa coluna)
    ob_col = _resolve_col(table, order_by) if order_by else None
    ord_kw = "ASC" if str(order).lower() == "asc" else "DESC"

    q = f'SELECT * FROM "{table}"'
    if where_sql:
        q += f" WHERE {where_sql}"
    if ob_col:
        q += f' ORDER BY "{ob_col}" {ord_kw}'
    q += " LIMIT ?"
    rows = _sqlite_exec(q, params + [max(1, int(limit))])
    return {"status":"ok","rows":rows}

# endpoint manual (útil para depurar sem LLM)
@app.get("/tool/sql_filter")
def call_sql_filter(
    table: str = Query(...),
    where: str = Query("{}"),
    limit: int = Query(100),
    order_by: str | None = Query(None),
    order: str = Query("desc"),
):
    try:
        where_dict = json.loads(where) if where else {}
        if not isinstance(where_dict, dict):
            return {"status":"error","message":"`where` deve ser um JSON de objeto"}
    except Exception as e:
        return {"status":"error","message":f"JSON inválido em `where`: {e}"}
    return sql_filter_fn(table, where_dict, limit, order_by, order)

# endpoints manuais (opcionais p/ depurar sem LLM)
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

# após definir DB_PATH e helpers:
rebuild_catalog()

@app.get("/catalog")
def catalog():
    return {t: {
        "columns": info["columns"],
        "table_tags": sorted(info["table_tags"])
    } for t, info in CATALOG.items()}


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

def _rows_to_html(rows: list[dict]) -> str:
    if not rows:
        return "<em>sem resultados</em>"
    cols = list(rows[0].keys())
    thead = "<thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr></thead>"
    body_rows = []
    for r in rows:
        tds = "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols)
        body_rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"

def fastpath_markdown(msg: str) -> str | None:
    m = re.search(r"filtr(a|e)\s+(.+?)\s+em\s+([A-Za-z0-9_\.]+)(?:.*?(?:até|limite)\s+(\d+))?", msg, re.I)
    if m:
        conds_str, table, limit_s = m.group(2), m.group(3), m.group(4)
        if not table.lower().endswith(".csv"):
            chosen = table if _table_exists(table) else resolve_table_from_text(msg, required_tags={"date","store","sku"} if "sku" in msg.lower() else None)
            if chosen and _table_exists(chosen):
                table = chosen
            else:
                return "<div class='muted'>Não reconheci a tabela citada. Use um nome próximo do arquivo ou faça upload em /docs.</div>"
            # parse condições no formato: COL=VAL, COL>VAL, COL<VAL, COL>=VAL, COL<=VAL, COL!=VAL
            parts = re.split(r"\s*(?:,| e )\s*", conds_str, flags=re.I)
            where = {}
            for p in parts:
                op = None
                for sym, tag in [(">=","gte"),("<=","lte"),("!=","ne"),(">","gt"),("<","lt"),("=","eq")]:
                    if sym in p:
                        left, right = p.split(sym, 1)
                        op = tag
                        col = left.strip()
                        val = right.strip().strip('"\'')
                        # tenta número pt-BR
                        if op in {"gt","gte","lt","lte"}:
                            try:
                                val = float(val.replace(",", "."))
                            except:
                                pass
                        key = f"{col}__{op}" if op != "eq" else col
                        where[key] = val
                        break
            lim = int(limit_s) if limit_s else 100
            res = sql_filter_fn(table, where, lim)
            if res.get("status") == "ok":
                html_table = _rows_to_html(res["rows"])
                return f"<h3>Filtro em <code>{table}</code> (SQLite)</h3>{html_table}"

def _rows_to_md(rows: list[dict]) -> str:
    if not rows:
        return "_sem resultados_"
    cols = list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    sep  = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return "\n".join([head, sep, body])

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
        "Converta períodos de tempo para datas AAAA-MM-DD quando necessário, normalize códigos de loja (remova zeros à esquerda) "
    ),
    tools=[head_csv_tool, list_csvs_tool, sql_head_tool, sql_aggregate_tool, sql_filter_tool, sku_intersection_tool],
    model=OpenAIChat(id="gpt-4o-mini")
)


# =========================
# Chat JSON + Chat HTML + UI
# =========================
class ChatIn(BaseModel):
    message: str

@app.post("/chat")
def chat(in_: ChatIn):
    try:
        msg = (in_.message or "").strip()
        md = fastpath_markdown(msg)
        if md is not None:
            return {"answer": md}   # evita chamada ao LLM

        response = agent.run(msg)
        return {"answer": response.content}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_html", response_class=HTMLResponse)
@app.post("/chat_html", response_class=HTMLResponse)
def chat_html(message: str = None, in_: ChatIn = None):
    msg = message or (in_.message if in_ else "")
    fast = fastpath_markdown(msg)
    if fast is not None:
        # se o fast-path já devolveu HTML, não passe pelo markdown.markdown
        if fast.lstrip().startswith("<"):
            return f"<div class='reply'>{fast}</div>"
        html_md = markdown.markdown(fast)
        return f"<div class='reply'>{html_md}</div>"

    response = agent.run(msg)
    html_md = markdown.markdown(response.content)
    return f"<div class='reply'>{html_md}</div>"

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
  .reply table{border-collapse: collapse; width:100%; overflow:auto}
  .reply table td,.reply table th{border:1px solid #21314f;padding:8px}
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

  function addBubble(text, who='bot', isHTML=false){
    const div = document.createElement('div');
    div.className = 'msg ' + (who==='user' ? 'user' : 'bot');
    if(isHTML){ div.innerHTML = text; } else { div.textContent = text; }
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
      // ✅ UMA chamada só, com X-Client-Id
      const res = await fetch('/chat_html', {
        method:'POST',
        headers:{
          'Content-Type':'application/json',
          'X-Client-Id': SID
        },
        body: JSON.stringify({message: text})
      });
      const html = await res.text();
      typing.remove();
      addBubble(html, 'bot', true);
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
        """
    )
