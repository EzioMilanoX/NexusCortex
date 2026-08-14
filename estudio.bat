@echo off
rem ==========================================================================
rem  Cortexus - Estudio de Podcast - launcher de um clique (Windows)
rem  Sobe SO a API e abre a pagina do estudio no navegador.
rem
rem  Por que existe separado do cortexus.bat: o estudio nao usa o Streamlit,
rem  entao subir o painel junto seria um processo a mais, uma janela a mais e
rem  uns segundos a mais de espera por nada. Quem quer o painel da agencia usa
rem  o cortexus.bat; quem quer gravar um episodio usa este.
rem
rem  Pre-requisito: venv criado. O Ollama so e necessario para a IA ESCREVER o
rem  roteiro - colando um roteiro pronto, o audio sai sem ele.
rem ==========================================================================
cd /d "%~dp0"
title Cortexus - Estudio de Podcast

if not exist ".venv\Scripts\python.exe" (
    echo [Estudio] Ambiente virtual .venv nao encontrado.
    echo           Rode o setup do README.md antes ^(py -3.12 -m venv .venv^).
    pause
    exit /b 1
)

rem Aviso amigavel: sem voz baixada o estudio abre, mas so o motor "tom"
rem funciona. Nao e um erro - por isso segue em frente depois de avisar.
if not exist "vozes\pt_BR-faber-medium.onnx" (
    echo [Estudio] Nenhuma voz do Piper encontrada em vozes\.
    echo           Para baixar as duas vozes pt-BR ^(~126 MB, uma vez^):
    echo.
    echo             .venv\Scripts\python.exe -m piper.download_voices pt_BR-faber-medium pt_BR-cadu-medium --download-dir vozes
    echo.
    echo           Sem elas, use o motor "Tom" na pagina para testar a montagem.
    echo.
)

echo [Estudio] Iniciando a API...
start "Cortexus API" ".venv\Scripts\python.exe" -m uvicorn api:app --host 127.0.0.1 --port 8000

echo [Estudio] Aguardando a API responder...
:esperar_api
timeout /t 1 /nobreak >nul
curl -s -o nul http://127.0.0.1:8000/docs
if errorlevel 1 goto esperar_api

echo [Estudio] No ar! Abrindo o estudio...
start "" http://localhost:8000/estudio

echo.
echo [Estudio] Abriu a janela "Cortexus API". Feche-a para desligar o servidor.
echo           Pode fechar ESTA janela.
pause
