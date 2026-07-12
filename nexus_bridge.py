#!/usr/bin/env python3
"""Ponte cliente para o Nexus (NexusCortex/Cortexus rodando em localhost:8000).

Ferramenta de linha de comando para OUTRO orquestrador (ex.: Roo Code no
VS Code) pedir uma pesquisa ou documentação ao Nexus de forma pontual, SEM
apontar o modelo dele para a porta 8000 (o que causaria delegação infinita
de agente-para-agente). É só uma chamada HTTP: dispara, espera, imprime.

Uso:
  python nexus_bridge.py --pesquisar "Quais as novidades do FastAPI 0.115?"
  python nexus_bridge.py --documentar "o módulo de crossfade do meu player"
  python nexus_bridge.py --pesquisar "..." --timeout 900   # espera até 15 min

O relatório vai para o STDOUT (para ser lido/parseado); mensagens de status
e erros vão para o STDERR. Código de saída: 0 sucesso; !=0 falha.
"""

from __future__ import annotations

import argparse
import sys

import requests

NEXUS_URL = "http://localhost:8000/v1/nexus-assist"

# Portugues com acento no terminal do Windows sem quebrar (cp1252).
for _fluxo in (sys.stdout, sys.stderr):
    if hasattr(_fluxo, "reconfigure"):
        _fluxo.reconfigure(encoding="utf-8")


def _erro(msg: str) -> None:
    print(f"[Nexus] {msg}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consulta o Nexus (CrewAI) via HTTP e imprime o relatório."
    )
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--pesquisar", metavar="DÚVIDA",
        help="Pede uma PESQUISA (dossiê com fontes) sobre a dúvida informada.",
    )
    grupo.add_argument(
        "--documentar", metavar="ASSUNTO",
        help="Pede a DOCUMENTAÇÃO (relatório técnico em prosa) do assunto.",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Segundos de espera pela resposta do Nexus (padrão: 600).",
    )
    parser.add_argument(
        "--url", default=NEXUS_URL,
        help=f"URL do endpoint do Nexus (padrão: {NEXUS_URL}).",
    )
    args = parser.parse_args()

    if args.pesquisar:
        tarefa, tipo = args.pesquisar, "pesquisa"
    else:
        tarefa, tipo = args.documentar, "documentacao"

    _erro(f"Consultando o Nexus ({tipo})… isso pode levar alguns minutos.")
    try:
        resposta = requests.post(
            args.url, json={"tarefa": tarefa, "tipo": tipo}, timeout=args.timeout
        )
    except requests.exceptions.Timeout:
        _erro(f"Tempo esgotado ({args.timeout}s). Aumente com --timeout.")
        return 4
    except requests.exceptions.ConnectionError:
        _erro("Não consegui falar com o Nexus. A API está no ar? (python api.py)")
        return 2
    except requests.RequestException as erro:
        _erro(f"Falha na requisição: {erro}")
        return 2

    if resposta.status_code == 503:
        _erro(f"Ollama indisponível: {_detalhe(resposta)}")
        return 3
    if resposta.status_code != 200:
        _erro(f"HTTP {resposta.status_code}: {_detalhe(resposta)}")
        return 1

    dados = resposta.json()
    # O relatório vai limpo para o stdout (o agente cliente lê isto).
    print(dados.get("relatorio", "(o Nexus não devolveu conteúdo)"))

    arquivo = dados.get("arquivo_salvo")
    if arquivo:
        _erro(f"Relatório também salvo em: {arquivo}")
    return 0


def _detalhe(resposta: requests.Response) -> str:
    """Extrai a mensagem de erro do corpo JSON, com fallback para o texto cru."""
    try:
        return str(resposta.json().get("detail", resposta.text[:300]))
    except ValueError:
        return resposta.text[:300]


if __name__ == "__main__":
    raise SystemExit(main())
