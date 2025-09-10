import os, sys, getpass, socket, argparse
import uvicorn

def get_local_ip() -> str:
    ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        pass
    finally:
        try: s.close()
        except Exception: pass
    return ip

def ensure_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return
    print("🔑 OPENAI_API_KEY não encontrada.")
    print("Cole sua chave da OpenAI (entrada oculta):")
    key = getpass.getpass("> ").strip()
    if not key:
        print("Erro: chave vazia. Abandonando.")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = key
    print("✅ Chave configurada neste processo.")

def main():
    parser = argparse.ArgumentParser(description="Launcher do Sentinela")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Reload automático em dev")
    args = parser.parse_args()

    # pede a chave se não houver
    ensure_api_key()

    # imprime links úteis
    local_ip = get_local_ip()
    urls = []
    if args.host in ("0.0.0.0", "127.0.0.1", "localhost"):
        urls.append(f"http://127.0.0.1:{args.port}")
        urls.append(f"http://{local_ip}:{args.port}")
    else:
        urls.append(f"http://{args.host}:{args.port}")

    print("\n🚀 Sentinela iniciando…")
    for u in urls:
        print(f"• UI (Chat):   {u}/")
        print(f"• Swagger:     {u}/docs")
        print("")

        uvicorn.run("agent_app:app", host=args.host, port=args.port, reload=args.reload, proxy_headers=True)


if __name__ == "__main__":
    main()
