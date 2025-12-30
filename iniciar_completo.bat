@echo off
echo ============================================
echo    SENTINELA - Iniciando Backend + Frontend
echo ============================================
echo.

:: Verificar se Python está disponível
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale o Python 3.10+
    pause
    exit /b 1
)

:: Verificar se Node está disponível
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Node.js nao encontrado. Instale o Node.js 18+
    pause
    exit /b 1
)

echo [1/2] Iniciando Backend (FastAPI) na porta 8000...
start "Sentinela Backend" cmd /k "cd /d %~dp0 && python server.py --host 0.0.0.0 --port 8000"

echo [2/2] Iniciando Frontend (React/Vite) na porta 5173...
timeout /t 3 /nobreak >nul
start "Sentinela Frontend" cmd /k "cd /d %~dp0frontend && npm install && npm run dev"

echo.
echo ============================================
echo    Servidores iniciados!
echo    Backend:  http://localhost:8000
echo    Frontend: http://localhost:5173
echo ============================================
echo.
echo Pressione qualquer tecla para fechar esta janela...
pause >nul
