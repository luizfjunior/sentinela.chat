import os, traceback
from typing import Optional
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
    cur = con.execute(q, params)
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def _sqlite_cols(table: str) -> list[str]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cols = [r["name"] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
    con.close()
    return cols

def _table_exists(table: str) -> bool:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    ok = con.execute(
        'SELECT 1 FROM sqlite_master WHERE type="table" AND name=? LIMIT 1', (table,)
    ).fetchone() is not None
    con.close()
    return ok

def _import_csv_to_sqlite(temp_csv_path: str, table: str) -> dict:
    """Importa CSV para SQLite com heurística de separador e em chunks."""
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.row_factory = sqlite3.Row

    first = True
    total = 0
    try:
        seps = [None, ";", ",", "\t", "|"]  # autodetect + fallbacks
        last_err = None
        for sep in seps:
            try:
                for chunk in pd.read_csv(
                    temp_csv_path,
                    sep=sep,
                    engine="python",
                    encoding="latin-1",
                    on_bad_lines="skip",
                    chunksize=50_000,
                ):
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
        return {"status": "ok", "db": DB_PATH, "table": safe_table, **info}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


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

# endpoints manuais (opcionais p/ depurar sem LLM)
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
# Registro das tools e Agent (uma única vez)
# =========================
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

agent = Agent(
    name="sentinela",
    system_message=(
        "Você é o Sentinela. Sempre use ferramentas para obter dados. "
        "Prefira as ferramentas SQL (`sql_head`, `sql_aggregate`) quando o usuário citar uma TABELA. "
        "Se o usuário citar um nome de arquivo CSV, use a tabela com o mesmo nome do arquivo (sem extensão, em minúsculas), "
        "por exemplo CANCELAMENTO_2025.csv -> cancelamento_2025. "
        "Responda em no máximo 2 frases e mostre no máximo 10 linhas/tópicos. Nunca invente dados."
    ),
    tools=[head_csv_tool, list_csvs_tool, sql_head_tool, sql_aggregate_tool],
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
        response = agent.run(msg)
        return {"answer": response.content}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_html", response_class=HTMLResponse)
@app.post("/chat_html", response_class=HTMLResponse)
def chat_html(message: str = None, in_: ChatIn = None):
    msg = message or (in_.message if in_ else "")
    response = agent.run(msg)
    html = markdown.markdown(response.content)
    return f"<div class='reply'>{html}</div>"

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
</style>
</head>
<body>
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
      const res = await fetch('/chat_html', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
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

  addBubble('Oi! Eu sou o Sentinela. Depois de fazer upload em /docs → POST /upload_csv, pergunte: \"mostre as 5 primeiras linhas de cancelamento_2025\".', 'bot', false);
</script>
</body>
</html>
        """
    )
