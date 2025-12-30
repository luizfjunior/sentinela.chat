@echo off
title Sentinela - Assistente de Dados

echo ========================================
echo   SENTINELA - Assistente de Dados
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Verificando ambiente virtual...
if not exist .venv (
    echo ERRO: Ambiente virtual nao encontrado!
    echo Execute: python -m venv .venv
    pause
    exit /b 1
)
echo [OK] Ambiente virtual encontrado

echo.
echo [2/4] Ativando ambiente virtual...
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERRO: Falha ao ativar ambiente virtual
    pause
    exit /b 1
)
echo [OK] Ambiente ativado

echo.
echo [3/4] Verificando dependencias...
python -c "import fastapi, uvicorn, agno, pandas" 2>nul
if errorlevel 1 (
    echo AVISO: Algumas dependencias podem estar faltando
    echo Instalando dependencias...
    pip install -r requirements.txt
)
echo [OK] Dependencias OK

echo.
echo [4/4] Iniciando servidor...
echo.
echo ========================================
echo   URLs disponiveis:
echo   - UI Chat:  http://127.0.0.1:8000/
echo   - Swagger:  http://127.0.0.1:8000/docs
echo   - Status:   http://127.0.0.1:8000/status
echo ========================================
echo.
echo Pressione Ctrl+C para parar o servidor
echo.

python server.py --host 127.0.0.1 --port 8000

pause
