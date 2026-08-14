@echo off
rem ==========================================================================
rem  Cortexus - launcher de um clique (Windows)
rem  Sobe a API e o painel visual (Streamlit) em janelas proprias e abre o
rem  navegador quando tudo estiver pronto.
rem  Pre-requisito: Ollama rodando + venv criado (ver README.md).
rem ==========================================================================
cd /d "%~dp0"
title Cortexus

if not exist ".venv\Scripts\python.exe" (
    echo [Cortexus] Ambiente virtual .venv nao encontrado.
    echo            Rode o setup do README.md antes ^(py -3.12 -m venv .venv^).
    pause
    exit /b 1
)

echo [Cortexus] Iniciando a API...
start "Cortexus API" ".venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000

echo [Cortexus] Aguardando a API responder...
:esperar_api
timeout /t 1 /nobreak >nul
curl -s -o nul http://127.0.0.1:8000/docs
if errorlevel 1 goto esperar_api

echo [Cortexus] Iniciando o painel visual...
rem --server.headless true evita o prompt de e-mail que travaria a janela.
start "Cortexus Painel" ".venv\Scripts\python.exe" -m streamlit run ui.py --server.headless true

echo [Cortexus] Aguardando o painel subir...
:esperar_painel
timeout /t 1 /nobreak >nul
curl -s -o nul http://127.0.0.1:8501/_stcore/health
if errorlevel 1 goto esperar_painel

echo [Cortexus] Tudo no ar! Abrindo o navegador...
start "" http://localhost:8501

echo.
echo [Cortexus] Estudio de Podcast: http://localhost:8000/estudio
echo            ^(ja esta no ar - a API acima o serve; ou use estudio.bat^)
echo.
echo [Cortexus] Abriram duas janelas: "Cortexus API" e "Cortexus Painel".
echo            Feche as duas para desligar o servidor. Pode fechar ESTA janela.
pause
