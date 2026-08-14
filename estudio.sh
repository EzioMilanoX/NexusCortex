#!/usr/bin/env bash
# ==========================================================================
#  Cortexus - Estudio de Podcast - launcher de um clique (Linux Mint)
#  Sobe SO a API e abre a pagina do estudio no navegador.
#
#  Por que existe separado do cortexus.sh: o estudio nao usa o Streamlit,
#  entao subir o painel junto seria um processo a mais por nada.
#
#  Pre-requisito: venv criado. O Ollama so e necessario para a IA ESCREVER o
#  roteiro - colando um roteiro pronto, o audio sai sem ele.
#
#  Uso:  chmod +x estudio.sh   depois   ./estudio.sh
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "[Estudio] Ambiente virtual .venv nao encontrado. Rode o setup do README."
    exit 1
fi

# Aviso amigavel: sem voz baixada o estudio abre, mas so o motor "tom" funciona.
if [ ! -f "vozes/pt_BR-faber-medium.onnx" ]; then
    echo "[Estudio] Nenhuma voz do Piper encontrada em vozes/."
    echo "          Para baixar as duas vozes pt-BR (~126 MB, uma vez):"
    echo
    echo "            $PY -m piper.download_voices \\"
    echo "                pt_BR-faber-medium pt_BR-cadu-medium --download-dir vozes"
    echo
    echo '          Sem elas, use o motor "Tom" na pagina para testar a montagem.'
    echo
fi

# Ao sair (Ctrl+C), mata a API.
trap 'kill $(jobs -p) 2>/dev/null' EXIT

echo "[Estudio] Iniciando a API..."
"$PY" -m uvicorn api:app --host 127.0.0.1 --port 8000 &
until curl -s -o /dev/null http://127.0.0.1:8000/docs; do sleep 1; done

echo "[Estudio] No ar! Abrindo o estudio..."
xdg-open http://localhost:8000/estudio >/dev/null 2>&1 || true

echo "[Estudio] Rodando. Pressione Ctrl+C para desligar."
wait
