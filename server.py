import os, sys, getpass, socket, argparse
import uvicorn

# --------------------------
# utilitários
# --------------------------
def get_local_ip() -> str:
    ip = "127.0.0.1"
    try:
        import socket as _s
        s = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        pass
    finally:
        try: s.close()
        except Exception: pass
    return ip

def ensure_api_key():
    """
    Garante OPENAI_API_KEY presente (para o modelo primário).
    Mantém a lógica original de pedir via getpass se não houver no ambiente.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return
    print("OPENAI_API_KEY nao encontrada.")
    print("Cole sua chave da OpenAI (entrada oculta):")
    key = getpass.getpass("> ").strip()
    if not key:
        print("Erro: chave vazia. Abandonando.")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = key
    print("Chave configurada neste processo.")

def _input_default(prompt: str, default: str) -> str:
    """
    input() com valor padrão: se usuário só apertar Enter, retorna default.
    """
    txt = input(f"{prompt} [{default}]: ").strip()
    return txt or default

def prompt_model_config():
    """
    Pergunta de forma interativa qual será o modelo PRIMÁRIO e, opcionalmente, o FREE.
    As variáveis são salvas no ambiente do processo (não persistem fora).
    A entrada da FREE_API_KEY fica COMENTADA por padrão, como você pediu.
    """
    print("\nConfiguracao de modelos (ENTER mantem o padrao atual)\n")

    # -------- PRIMARY --------
    curr_primary = os.environ.get("PRIMARY_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    primary = _input_default("Modelo primário", curr_primary)
    os.environ["PRIMARY_MODEL"] = primary

    # -------- FREE (opcional) --------
    use_free = _input_default("Deseja configurar um modelo 'free/barato' para planner? (s/n)", "n").lower()
    if use_free.startswith("s"):
        curr_free_id   = os.environ.get("FREE_MODEL_ID", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
        curr_free_base = os.environ.get("FREE_API_BASE", "https://api.groq.com/openai/v1").strip() or "https://api.groq.com/openai/v1"
        free_id   = _input_default("FREE_MODEL_ID (ex.: llama-3.1-8b-instant)", curr_free_id)
        free_base = _input_default("FREE_API_BASE", curr_free_base)

        os.environ["FREE_MODEL_ID"] = free_id
        os.environ["FREE_API_BASE"] = free_base

        print("Cole a FREE_API_KEY do provider 'free' (entrada oculta):")
        free_key = getpass.getpass("> ").strip()
        if free_key:
           os.environ["FREE_API_KEY"] = free_key

        # apenas avisa se já havia key no ambiente
        if os.environ.get("FREE_API_KEY"):
            print("FREE_API_KEY ja esta definida no ambiente.")
        else:
            print("FREE_API_KEY nao definida aqui (ok para testes se ja estiver no ambiente do Windows).")
    else:
        # Se usuário optar por não usar free, limpamos as variáveis no processo
        os.environ.pop("FREE_MODEL_ID", None)
        os.environ.pop("FREE_API_BASE", None)
        # não mexe em FREE_API_KEY — pode estar setada no sistema e tudo bem

    # resumo
    print("\nResumo da sessão:")
    print(f"  PRIMARY_MODEL = {os.environ.get('PRIMARY_MODEL')}")
    if os.environ.get("FREE_MODEL_ID"):
        print(f"  FREE_MODEL_ID = {os.environ.get('FREE_MODEL_ID')}")
        print(f"  FREE_API_BASE = {os.environ.get('FREE_API_BASE')}")
        print(f"  FREE_API_KEY  = {'<definida>' if os.environ.get('FREE_API_KEY') else '<não definida>'}")
    else:
        print("  Free planner  = desativado")
    print("")

# --------------------------
# launcher
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Launcher do Sentinela")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Reload automático em dev")
    parser.add_argument("--ask-models", action="store_true", help="Pergunta modelos/keys ao iniciar")
    args = parser.parse_args()

    # 1) Garante chave da OpenAI (primária) — sua app exige no startup
    ensure_api_key()

    # 2) Pergunta modelos (se solicitado)
    if args.ask_models:
        prompt_model_config()

    # 3) imprime links úteis
    local_ip = get_local_ip()
    urls = []
    if args.host in ("0.0.0.0", "127.0.0.1", "localhost"):
        urls.append(f"http://127.0.0.1:{args.port}")
        urls.append(f"http://{local_ip}:{args.port}")
    else:
        urls.append(f"http://{args.host}:{args.port}")

    print("\nSentinela iniciando...")
    for u in urls:
        print(f"• UI (Chat):   {u}/")
        print(f"• Swagger:     {u}/docs")
    print("")

    # 4) inicia o servidor (fora do loop)
    uvicorn.run(
        "agent_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        proxy_headers=True
    )

if __name__ == "__main__":
    main()
