"""Cortexus — agência local de múltiplos agentes de IA (CrewAI + Ollama).

Equipe de três agentes (Pesquisador -> Escritor -> Revisor) que produz
um relatório em .txt no diretório output/. O Pesquisador tem busca web
real e gratuita (pacote ddgs — metabusca sem chave de API).

Este arquivo pode ser escrito e versionado HOJE sem nenhum modelo baixado:
nada aqui faz chamada de rede ao ser importado — a conexão com o Ollama
só acontece quando crew.kickoff() é executado.

Requisitos em tempo de execução (para o futuro, quando o SSD chegar):
  1. Ollama instalado e rodando (http://localhost:11434)
  2. Modelo baixado:  ollama pull llama3.1

Uso:
  python main.py                          # usa o tema padrão
  python main.py Agentes de IA locais     # tema customizado
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Caminhos — regra de ouro cross-platform.
#
# pathlib resolve sozinho o separador correto ("\" no Windows, "/" no Linux)
# e o operador "/" abaixo funciona identicamente nos dois sistemas.
# Ancorar em Path(__file__) garante que output/ fica sempre ao lado deste
# arquivo, não importa de qual diretório o script for chamado.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# No Windows, stdout redirecionado (pipe, arquivo de log) usa cp1252 e
# quebra os acentos do português; força UTF-8 nos dois SOs.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# LLM local via Ollama.
#
# O prefixo "ollama/" no nome do modelo indica ao roteador interno do
# CrewAI (LiteLLM) que o provedor é o Ollama local — sem ele, o CrewAI
# tentaria interpretar "llama3.1" como um modelo da OpenAI.
#
# Sobre a GPU: o offload para a RTX 4060 NÃO é configurado aqui. Quem
# decide isso é o servidor Ollama, que detecta o driver NVIDIA/CUDA
# automaticamente. O llama3.1 8B quantizado (~4,9 GB) cabe inteiro nos
# 8 GB de VRAM, então o Ollama carrega 100% das camadas na GPU e a CPU
# fica livre. (Verificação: seção "Guia futuro" do README.md.)
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
MODELO_BASE = "llama3.1"        # tag baixada com `ollama pull`
# Variante local com num_ctx=8192 assado (ver Modelfile): o padrão do
# Ollama (4096) trunca o começo do prompt — as instruções de ferramenta
# do CrewAI — e passar num_ctx via LLM(...) quebra no crewai 1.15.1.
MODELO_OLLAMA = "llama3.1-8k"
MODEL = f"ollama/{MODELO_OLLAMA}"   # prefixo "ollama/" = provedor p/ CrewAI

TEMA_PADRAO = "O impacto de agentes de IA locais na privacidade de dados"


@tool("Busca na Web")
def busca_web(query: str) -> str:
    """Busca na web e retorna os 5 melhores resultados, numerados, com
    título, URL e resumo. A entrada deve ser uma consulta curta de
    palavras-chave (não uma frase longa)."""
    # Regras de robustez para tools com modelos locais 8B: nunca levantar
    # exceção e sempre devolver texto não vazio — o laço ReAct realimenta
    # a resposta ao modelo, que então reformula a busca e tenta de novo.
    try:
        resultados = DDGS().text(query, max_results=5)
    except Exception as erro:
        return (
            f"A busca falhou ({erro}). Tente de novo com palavras-chave "
            "mais simples ou diferentes."
        )
    if not resultados:
        return f"Nenhum resultado para '{query}'. Tente outras palavras-chave."
    return "\n\n".join(
        f"{i}. {r.get('title', '(sem título)')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(resultados, 1)
    )


def diagnostico_ollama(url: str = OLLAMA_URL, timeout: float = 5.0) -> str | None:
    """Pre-flight: retorna None se o Ollama está de pé E o modelo existe;
    caso contrário, uma mensagem dizendo exatamente o que falta.

    Evita que o usuário espere o traceback do LiteLLM para descobrir que
    esqueceu de subir o Ollama — ou que o modelo ainda não foi baixado
    (o endpoint /api/tags lista os modelos locais; a raiz do servidor
    responde 200 mesmo sem nenhum modelo instalado).
    """
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as resposta:
            dados = json.load(resposta)
    except (OSError, http.client.HTTPException, json.JSONDecodeError):
        # OSError cobre URLError (conexão recusada, timeout);
        # HTTPException cobre um serviço não-HTTP ocupando a porta.
        return (
            f"Servidor Ollama não encontrado em {url}.\n"
            "  Instale/inicie o Ollama (ele sobe o servidor sozinho)."
        )

    bases = {m.get("name", "").split(":")[0] for m in dados.get("models", [])}
    if MODELO_OLLAMA in bases:
        return None
    if MODELO_BASE not in bases:
        return (
            f"O Ollama está rodando, mas o modelo '{MODELO_BASE}' ainda "
            "não foi baixado. Na pasta do projeto, rode:\n"
            f"  1. ollama pull {MODELO_BASE}\n"
            f"  2. ollama create {MODELO_OLLAMA} -f Modelfile"
        )
    return (
        f"O modelo base '{MODELO_BASE}' existe, mas falta criar a variante "
        f"'{MODELO_OLLAMA}' (contexto de 8192 tokens).\n"
        f"  Na pasta do projeto, rode:  ollama create {MODELO_OLLAMA} -f Modelfile"
    )


def montar_equipe(tema: str, llm: LLM) -> Crew:
    """Monta a agência Cortexus: Pesquisador -> Escritor -> Revisor."""
    pesquisador = Agent(
        role="Pesquisador Sênior",
        goal=f"Levantar fatos atuais e verificáveis sobre: {tema}",
        backstory=(
            "Você é um analista de pesquisa meticuloso: usa a busca na web "
            "para levantar fatos atuais, anota a URL da fonte de cada "
            "achado e separa fato de especulação. Assim que tiver material "
            "suficiente, entregue a resposta final imediatamente — não "
            "continue buscando."
        ),
        llm=llm,
        tools=[busca_web],
        # Mitigações para modelos locais 8B usando ferramentas (o CrewAI
        # usa prompting ReAct com o Ollama, não function calling nativo):
        max_iter=5,              # limita o laço se o modelo errar o formato
        allow_delegation=False,  # 8B não lida bem com delegação
        verbose=True,
    )

    escritor = Agent(
        role="Escritor Técnico",
        goal=(
            "Transformar a pesquisa em um relatório claro, bem estruturado "
            "e em português do Brasil"
        ),
        backstory=(
            "Você é um redator técnico experiente: escreve com precisão, "
            "didática e fluidez, sem jargão desnecessário."
        ),
        llm=llm,
        verbose=True,
    )

    pesquisa = Task(
        description=(
            f"Use a ferramenta 'Busca na Web' para pesquisar o tema "
            f"'{tema}'. Faça de 1 a 3 buscas com palavras-chave "
            "diferentes. Produza os 5 a 7 pontos mais importantes, cada "
            "um com uma breve explicação e a URL da fonte."
        ),
        expected_output=(
            "Lista em tópicos com os principais achados e a URL da fonte "
            "de cada um, em português."
        ),
        agent=pesquisador,
    )

    revisor = Agent(
        role="Revisor de Qualidade",
        goal=(
            "Garantir que o relatório final esteja fiel à pesquisa, bem "
            "estruturado e em português impecável"
        ),
        backstory=(
            "Você é um editor-chefe rigoroso: confere a consistência do "
            "texto com os achados da pesquisa, corrige imprecisões e "
            "melhora a fluidez sem mudar o sentido."
        ),
        llm=llm,
        verbose=True,
    )

    redacao = Task(
        description=(
            "Usando exclusivamente os achados do Pesquisador, escreva um "
            "relatório de aproximadamente 400 palavras com título, "
            "introdução, desenvolvimento em seções e conclusão."
        ),
        expected_output=(
            "Relatório em texto corrido, pronto para revisão."
        ),
        agent=escritor,
        context=[pesquisa],  # o Escritor recebe a saída do Pesquisador
    )

    revisao = Task(
        description=(
            "Revise o relatório do Escritor comparando-o com os achados do "
            "Pesquisador: corrija imprecisões, elimine repetições e melhore "
            "a fluidez. Não comente as mudanças nem adicione observações — "
            "devolva SOMENTE o texto final revisado."
        ),
        expected_output=(
            "A versão final e revisada do relatório, em texto corrido."
        ),
        agent=revisor,
        context=[pesquisa, redacao],  # recebe a pesquisa E o rascunho
    )

    return Crew(
        agents=[pesquisador, escritor, revisor],
        tasks=[pesquisa, redacao, revisao],
        process=Process.sequential,
        verbose=True,
    )


def limpar_saida(texto: str) -> str:
    """Remove artefatos do laço ReAct que modelos 8B deixam escapar no
    início do texto (ex.: um cabeçalho "Resposta Final:")."""
    # Os asteriscos só são absorvidos NA LINHA do cabeçalho: um "**" na
    # linha seguinte é o início de um título legítimo e deve sobreviver.
    return re.sub(
        r"^[\s*#]*(resposta final|final answer)[ \t]*:?[ \t]*\**[ \t]*(?:\r?\n)*",
        "",
        texto.strip(),
        flags=re.IGNORECASE,
    )


def salvar_relatorio(conteudo: str, tema: str) -> Path:
    """Grava o relatório em output/ com timestamp no nome do arquivo."""
    conteudo = limpar_saida(conteudo)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arquivo = OUTPUT_DIR / f"relatorio_{carimbo}.txt"

    # encoding="utf-8" explícito é obrigatório aqui: sem ele, o Windows
    # gravaria em cp1252 e os acentos do português quebrariam ao abrir
    # o mesmo arquivo no Linux Mint.
    arquivo.write_text(f"Tema: {tema}\n\n{conteudo}\n", encoding="utf-8")
    return arquivo


def main() -> int:
    tema = " ".join(sys.argv[1:]).strip() or TEMA_PADRAO

    problema = diagnostico_ollama()
    if problema:
        print(
            f"[Cortexus] {problema}\n"
            "  Detalhes no README.md, seção 'Guia futuro'."
        )
        return 1

    llm = LLM(
        model=MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,  # 8B erra o formato ReAct com temperatura mais alta
    )

    print(f"[Cortexus] Iniciando a equipe para o tema: {tema}\n")
    equipe = montar_equipe(tema, llm)
    try:
        resultado = equipe.kickoff()
    except Exception as erro:  # ex.: Ollama caiu no meio da execução
        print(f"\n[Cortexus] Falha ao executar a equipe: {erro}")
        return 1

    arquivo = salvar_relatorio(str(resultado), tema)
    print(f"\n[Cortexus] Relatório salvo em: {arquivo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
