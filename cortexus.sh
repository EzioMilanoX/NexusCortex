#!/usr/bin/env bash
# ==========================================================================
#  Cortexus - launcher de um clique (Linux Mint)
#  Sobe a API e o painel Streamlit, abre o navegador e, ao encerrar (Ctrl+C),
#  desliga os dois. Pre-requisito: Ollama rodando + venv criado (ver README).
#  Uso:  chmod +x cortexus.sh   depois   ./cortexus.sh
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "[Cortexus] Ambiente virtual .venv nao encontrado. Rode o setup do README."
    exit 1
fi

# Ao sair (Ctrl+C), mata os dois processos filhos.
trap 'kill $(jobs -p) 2>/dev/null' EXIT

echo "[Cortexus] Iniciando a API..."
"$PY" -m uvicorn api:app --host 127.0.0.1 --port 8000 &
until curl -s -o /dev/null http://127.0.0.1:8000/docs; do sleep 1; done

echo "[Cortexus] Iniciando o painel..."
"$PY" -m streamlit run ui.py --server.headless true &
until curl -s -o /dev/null http://127.0.0.1:8501/_stcore/health; do sleep 1; done

echo "[Cortexus] Tudo no ar! Abrindo o navegador..."
xdg-open http://localhost:8501 >/dev/null 2>&1 || true

echo "[Cortexus] Estudio de Podcast: http://localhost:8000/estudio"
echo "           (ja esta no ar - a API acima o serve; ou use ./estudio.sh)"
echo "[Cortexus] Rodando. Pressione Ctrl+C para desligar tudo."
wait
