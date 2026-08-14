"""
Cortexus — Agência local de múltiplos agentes de IA (CrewAI + Ollama).

Esta versão (Híbrida) é capaz de atuar tanto como Redatora de Textos
quanto como Engenheira de Software. A agência adapta seu comportamento
com base no seu comando, removendo "tagarelice" indesejada ao gerar código
e salvando o arquivo com a extensão correta automaticamente.

Nível 1 de evolução: o Pesquisador também lê pastas e arquivos locais
(DirectoryReadTool/FileReadTool), além de buscar na web. Por segurança,
o acesso a disco fica confinado à pasta do projeto (BASE_DIR) — ver a
seção "FERRAMENTAS DE DISCO" abaixo para como liberar outras pastas.

Requisitos em tempo de execução:
  1. Ollama instalado e rodando (http://localhost:11434)
  2. Modelo baixado e variante criada:
       ollama pull qwen2.5:7b
       ollama create qwen2.5-8k -f Modelfile.qwen

Uso (Terminal do Linux Mint ou PowerShell do Windows):
  python main.py Escreva um artigo sobre IA
  python main.py Crie um script que mostre a hora e salve como relogio.py
  python main.py Liste os arquivos do projeto e resuma o README.md
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import shlex
import socket
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from crewai_tools import (
    DirectoryReadTool,
    DirectorySearchTool,
    FileReadTool,
    ScrapeWebsiteTool,
)
from crewai_tools.security.safe_path import (
    format_error_for_display,
    format_path_for_display,
    validate_file_path,
)
from ddgs import DDGS

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DE CAMINHOS E SISTEMA (CROSS-PLATFORM)
# ---------------------------------------------------------------------------
# O pathlib resolve automaticamente se o sistema usa "\" (Windows) ou "/" (Linux).
# Isso garante que a pasta output/ seja criada e acessada perfeitamente
# independentemente de qual SO do seu dual-boot você esteja usando.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# Força a saída do terminal para UTF-8. 
# Evita que o Windows quebre a formatação de acentos em português.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DO LLM (OLLAMA + GPU RTX 4060)
# ---------------------------------------------------------------------------
# O prefixo "ollama/" avisa ao CrewAI para não tentar buscar chaves da OpenAI.
# A alocação do modelo na sua RTX 4060 é feita silenciosamente pelo próprio 
# servidor do Ollama via CUDA, liberando os 28 núcleos do seu processador.
OLLAMA_URL = "http://localhost:11434"
# Nível 4: qwen2.5 7B no lugar do llama3.1 8B — mesma pegada de VRAM
# (~6 GB a 8k, 100% GPU) e é o modelo local ~7B que a comunidade CrewAI
# reporta como o tool-caller mais confiável. (O 32B pedido não cabe na
# RTX 4060: ~20 GB de pesos → ~3-6 tok/s transbordando p/ CPU = horas
# por job. O 14B forçaria split 60/40 GPU/CPU. Ver README.)
MODELO_BASE = "qwen2.5"
MODELO_OLLAMA = "qwen2.5-8k"  # Variante de alto contexto (criada via Modelfile.qwen)
# O prefixo do modelo não decide o caminho de tool-calling — quem decide
# é LLMLocal.supports_function_calling() abaixo. Mantemos "ollama/".
MODEL = f"ollama/{MODELO_OLLAMA}"

TEMA_PADRAO = "O impacto de agentes de IA locais na privacidade de dados"

# Embedder LOCAL (nomic-embed-text via Ollama) — usado tanto pela memória de
# longo prazo quanto pela Busca Semântica (RAG). Formato validado empiricamente
# (mesmo ProviderSpec do build_embedder do CrewAI). Nunca toca na OpenAI.
CONFIG_EMBEDDER = {
    "provider": "ollama",
    "config": {"model_name": "nomic-embed-text", "url": OLLAMA_URL},
}

# ---------------------------------------------------------------------------
# EXTENSÕES DE SAÍDA (fonte única — usada para nomear o arquivo E para o
# painel escolher o syntax highlighting). Antes só py/txt/md/csv/json eram
# reconhecidas, então "salve como setup_kvm.sh" caía no fallback .txt e o
# painel renderizava o script como texto puro (sem cores). Aqui ampliamos.
# ---------------------------------------------------------------------------
LINGUAGEM_POR_EXT = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".sh": "bash", ".ps1": "powershell", ".bat": "batch",
    ".html": "html", ".css": "css", ".sql": "sql",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".xml": "xml", ".java": "java", ".c": "c", ".cpp": "cpp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".dockerfile": "dockerfile",
}
# Extensões de prosa/dados: aceitas para nomear o arquivo, mas NÃO viram
# aba "Código" (são renderizadas como texto/markdown na aba "Análise").
_EXT_PROSA = (".md", ".txt", ".csv")
# Todas as extensões que nome_arquivo_pedido aceita como nome de saída.
EXT_SAIDA = tuple(LINGUAGEM_POR_EXT) + _EXT_PROSA


def classificar_saida(arquivo: str) -> tuple[str, str]:
    """A partir do nome do arquivo salvo, devolve (tipo, linguagem):
    tipo = 'codigo' (com linguagem p/ highlighting) ou 'analise' (prosa).
    A extensão é a autoridade — quem a decidiu foi nome_arquivo_pedido()."""
    ext = Path(arquivo).suffix.lower()
    if ext in LINGUAGEM_POR_EXT:
        return "codigo", LINGUAGEM_POR_EXT[ext]
    return "analise", ""


def criar_llm() -> LLM:
    """Fábrica única do LLM local — usada pelo main() e pela api.py, para
    o parâmetro do modelo não divergir entre os dois pontos de entrada.

    Aqui mora o fix mais importante do Nível 4. Contexto do bug:
    - Com memory=True, o CrewAI usa o executor EXPERIMENTAL de tool-calling
      NATIVO sempre que o LLM se declara capaz — e o litellm marca qwen2.5
      (e llama3.1) como capazes.
    - Nesse caminho nativo, o qwen2.5 devolve resposta VAZIA quando o
      prompt fica grande (memória + esquemas de 3 ferramentas + backstory):
      "Invalid response from LLM call - None or empty", derrubando o job.
      Reproduzido 2/2 COM memória, 0/2 SEM memória.

    A correção é forçar o caminho ReAct (baseado em prompt, texto puro — o
    mesmo que o llama3.1 usava e que o limpar_saida já higieniza), fazendo
    supports_function_calling() devolver False. NÃO dá para subclassar o
    LLM: o LLM.__new__ do CrewAI é uma fábrica que despacha para uma classe
    nativa do provider (OpenAICompatibleCompletion) e ignora a subclasse.
    Por isso sobrescrevemos o método NA INSTÂNCIA retornada.
    """
    llm = LLM(
        model=MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,  # temperatura baixa = código mais estável e focado
    )
    # Atribuído na instância (não bound method): _supports_native_tool_calling
    # do CrewAI chama self.llm.supports_function_calling() — um callable que
    # devolve False basta para mantê-lo no laço ReAct.
    llm.supports_function_calling = lambda: False
    return llm

# ---------------------------------------------------------------------------
# FERRAMENTAS (TOOLS)
# ---------------------------------------------------------------------------
# Domínios que nunca têm prosa para ler (vídeo, rede social, áudio).
_DOMINIOS_SEM_TEXTO = (
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "twitch.tv",
    "instagram.com", "tiktok.com", "facebook.com", "twitter.com", "x.com",
    "spotify.com", "soundcloud.com", "pinterest.com",
)
# Agregadores de slides/respostas: têm prosa, mas prosa de qualidade
# imprevisível e sem revisão. Nunca são lidos — só poluiriam o contexto.
_DOMINIOS_DESCARTADOS = (
    "prezi.com", "slideshare.net", "scribd.com", "coursehero.com",
    "passeidireto.com", "brainly.com", "quora.com", "answers.com",
    "pinterest.com", "medium.com/@",
)
# FONTE DE REFERÊNCIA: enciclopédia, domínio institucional, documentação
# oficial. É de onde devem sair datas, nomes próprios e definições.
_DOMINIOS_REFERENCIA = (
    "wikipedia.org", "britannica.com", "plato.stanford.edu", "arxiv.org",
    ".gov", ".gov.br", ".edu", ".edu.br", ".ac.uk", "docs.python.org", "docs.",
    "developer.mozilla.org", "scielo.",
)
# Ordem de preferência para escolher quem ler primeiro.
_DOMINIOS_PREFERIDOS = _DOMINIOS_REFERENCIA + (".org",)

# Quantas páginas a busca lê sozinha, e o teto de texto que elas somam. Dois
# mil e quinhentos caracteres são ~600 tokens: cabem no num_ctx 8192 junto do
# backstory e dos esquemas das cinco ferramentas.
MAX_FONTES_LIDAS = 2
ORCAMENTO_FONTES = 2500


def _vale_ler(url: str) -> bool:
    baixo = (url or "").lower()
    if not baixo.startswith(("http://", "https://")):
        return False
    if baixo.endswith((".pdf", ".zip", ".mp4", ".mp3", ".jpg", ".png")):
        return False
    if any(d in baixo for d in _DOMINIOS_DESCARTADOS):
        return False
    return not any(d in baixo for d in _DOMINIOS_SEM_TEXTO)


def _eh_referencia(url: str) -> bool:
    """True para enciclopédia/instituição/documentação — a fonte que pode ser
    citada para uma DATA ou uma DEFINIÇÃO."""
    return any(d in (url or "").lower() for d in _DOMINIOS_REFERENCIA)


def _nota_de_credibilidade(url: str) -> int:
    """Menor é melhor — usado só para ordenar quem será lido primeiro."""
    baixo = (url or "").lower()
    for posicao, marca in enumerate(_DOMINIOS_PREFERIDOS):
        if marca in baixo:
            return posicao
    return len(_DOMINIOS_PREFERIDOS)


def _ler_fontes(resultados: list[dict]) -> str:
    """Abre sozinha as fontes mais credíveis da busca e devolve o texto delas.

    POR QUE A BUSCA LÊ SOZINHA, em vez de esperar o agente pedir. A ferramenta
    'Ler Página Web' existe e funciona, mas foi MEDIDO que o qwen 7B não a
    chama: num teste completo o log registrou uma única execução de ferramenta
    ('busca_na_web') e nenhuma leitura, mesmo com a instrução "OBRIGATÓRIO" no
    prompt do agente E na tarefa. É a mesma limitação já documentada do
    terminal, que o Escritor também ignora.
    Enquanto a leitura era uma DECISÃO do modelo, ela simplesmente não
    acontecia. Fundindo busca e leitura numa ferramenta só, a decisão deixa de
    existir: o agente pede uma busca e recebe prosa junto, queira ou não. É a
    mesma escolha de projeto do `escolher_saida` e do `parse_roteiro` — quando
    o passo é essencial, ele mora no código, não na obediência do modelo.
    """
    candidatos = sorted(
        (r for r in resultados if _vale_ler(r.get("href", ""))),
        key=lambda r: _nota_de_credibilidade(r.get("href", "")),
    )

    trechos: list[str] = []
    gasto = 0
    for r in candidatos:
        if len(trechos) >= MAX_FONTES_LIDAS or gasto >= ORCAMENTO_FONTES:
            break
        url = r.get("href", "")
        # Chama a função interna, não a @tool: `ler_pagina_web` já é um objeto
        # de ferramenta do CrewAI aqui, e invocá-lo por dentro custaria o
        # empacotamento de argumentos sem ganho nenhum.
        problema = _url_segura(url)
        if problema:
            continue
        try:
            texto = _enxugar_pagina(
                _raspador.run(website_url=url),
                limite=min(ORCAMENTO_FONTES - gasto, ORCAMENTO_FONTES),
            )
        except Exception:
            continue  # fonte ruim não derruba a busca; tenta a próxima
        if not texto:
            continue  # página sem prosa (vídeo, JS) — não conta no orçamento
        # O rótulo do nível importa: num teste real o único erro factual que
        # sobreviveu ("equilíbrio de Nash formulado no início do século XX";
        # foi 1950-51) veio de um blog de consultoria lido junto da Wikipédia.
        # O modelo não tinha como saber qual das duas fontes pesava mais.
        nivel = "FONTE DE REFERÊNCIA" if _eh_referencia(url) else "FONTE SECUNDÁRIA"
        trechos.append(f"[{nivel}] TEXTO DE {url}:\n{texto}")
        gasto += len(texto)

    if not trechos:
        return (
            "\n\nNENHUMA das páginas devolveu texto legível. NÃO invente o "
            "conteúdo: refaça a busca com outras palavras-chave, ou diga "
            "explicitamente que não encontrou fonte."
        )

    regra = (
        "DATAS, NOMES PRÓPRIOS e DEFINIÇÕES devem sair da FONTE DE REFERÊNCIA. "
        "A FONTE SECUNDÁRIA serve para exemplos e aplicações, e não para "
        "corrigir ou contradizer a de referência."
        if any(t.startswith("[FONTE DE REFERÊNCIA]") for t in trechos)
        else "Nenhuma fonte de referência foi encontrada: trate as datas e os "
        "números abaixo como incertos e diga isso no dossiê."
    )
    return (
        "\n\n=== TEXTO DAS FONTES (leia e cite daqui) ===\n"
        f"{regra}\n\n" + "\n\n".join(trechos)
    )


@tool("Busca na Web")
def busca_web(query: str) -> str:
    """Busca na web e retorna os 5 melhores resultados E o TEXTO das páginas
    mais confiáveis entre eles, já aberto para você. Use as definições, nomes e
    números que aparecem nesse texto — é a sua fonte. Usado pelo Pesquisador."""
    try:
        resultados = DDGS().text(query, max_results=5)
    except Exception as erro:
        # Tratamento de erro vital para modelos locais de 8B não entrarem em colapso
        return f"A busca falhou ({erro}). Tente de novo com palavras-chave diferentes."

    if not resultados:
        return f"Nenhum resultado para '{query}'. Tente outras palavras-chave."

    lista = "\n\n".join(
        f"{i}. {r.get('title', '(sem título)')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(resultados, 1)
    )
    return lista + _ler_fontes(resultados)

# ---------------------------------------------------------------------------
# LEITURA DE PÁGINA WEB
# ---------------------------------------------------------------------------
# POR QUE ESTA FERRAMENTA EXISTE. A `busca_web` devolve apenas título + link +
# snippet do DDGS — o agente nunca via o TEXTO de nada. Diagnosticado num teste
# real: pedimos um podcast sobre "teoria dos jogos", a busca retornou três
# vídeos do YouTube e uma página de curso, e o modelo, sem nada em que se
# apoiar, escreveu de memória — definiu equilíbrio de Nash errado e inventou
# exemplos ("jogo do lembretista"). Não era limite do 7B: era falta de acesso
# ao conteúdo. Com esta ferramenta o fluxo fecha: buscar -> escolher a fonte
# credível -> LER -> escrever a partir do texto.
#
# A ScrapeWebsiteTool do crewai-tools é gratuita e sem chave de API (é
# scraper, não busca — a observação do README sobre "todas exigem chave" vale
# para as ferramentas de BUSCA). Mas ela é crua: devolve o menu de navegação
# junto e não tem teto de tamanho. Daí o invólucro abaixo, na mesma linha do
# FileReadToolUTF8: usa o motor de terceiro, corrige o comportamento.
LIMITE_PAGINA_WEB = 4000  # ~1000 tokens: caber no num_ctx 8192 com folga

# Prefixo que a ScrapeWebsiteTool cola no começo do retorno.
_PREFIXO_SCRAPE = "The following text is scraped website content:"


def _url_segura(url: str) -> str | None:
    """Valida a URL antes de buscá-la. Devolve o erro, ou None se estiver ok.

    Isto é sandbox, não paranoia decorativa. O Pesquisador processa texto NÃO
    CONFIÁVEL da web aberta, então uma injeção de prompt pode fazê-lo pedir
    qualquer URL. Duas portas ficam fechadas:

    1. ESQUEMA: só http/https. Sem isto, um `file:///C:/Users/...` usaria esta
       ferramenta para contornar o confinamento a BASE_DIR que o crewai-tools
       impõe às ferramentas de disco.
    2. DESTINO INTERNO (SSRF): resolve o host e recusa loopback, rede privada
       e link-local. Senão `http://localhost:11434` ou um IP 192.168.x.x da
       rede doméstica seriam alcançáveis pelo agente — inclusive a própria
       API do Cortexus, que roda sem autenticação.
    """
    try:
        partes = urllib.parse.urlparse(url.strip())
    except ValueError:
        return "URL malformada."

    if partes.scheme not in ("http", "https"):
        return (
            f"Esquema '{partes.scheme or 'ausente'}' bloqueado. Só http:// e "
            "https:// são permitidos (para ler arquivos locais use as "
            "ferramentas de disco)."
        )
    if not partes.hostname:
        return "URL sem domínio."

    try:
        enderecos = socket.getaddrinfo(partes.hostname, None)
    except (OSError, UnicodeError):
        return f"Não consegui resolver o domínio '{partes.hostname}'."

    for info in enderecos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return (
                f"Endereço interno bloqueado ({ip}). Esta ferramenta só "
                "alcança a internet pública."
            )
    return None


# Fim de frase, tolerando o fechamento de citação/parêntese depois do ponto.
_RE_FIM_DE_FRASE = re.compile(r"[.!?][\"'”’)\]]?$")
# Marcador de referência da Wikipédia ("[1]", "[nota 2]") — não se pronuncia.
_RE_REFERENCIA = re.compile(r"\[\s*(?:\d+|nota\s*\d+|editar[^\]]*)\s*\]", re.IGNORECASE)


def _enxugar_pagina(bruto: str, limite: int = LIMITE_PAGINA_WEB) -> str:
    """Tira a navegação e trunca — o que sobra é texto de leitura.

    O CRITÉRIO é "a linha termina em pontuação de frase?", e não o tamanho
    dela. Foi a segunda tentativa: filtrar por linhas curtas deixava passar
    justamente o pior lixo, porque o scraper junta o menu inteiro numa linha
    longuíssima ("Página principal Conteúdo destacado Eventos atuais...") e a
    lista de idiomas da Wikipédia noutra. Nenhuma das duas termina em ponto;
    todo parágrafo de prosa termina.

    Efeito colateral desejado: uma página de vídeo do YouTube não tem prosa
    nenhuma, então sobra vazio e a ferramenta avisa que não há o que ler — em
    vez de entregar "About Press Copyright Contact us" como se fosse fonte.
    Era exatamente o caso que produziu o roteiro alucinado.
    """
    texto = bruto.replace(_PREFIXO_SCRAPE, "", 1)

    uteis: list[str] = []
    for linha in texto.splitlines():
        linha = " ".join(linha.split())  # colapsa espaços e tabs
        if not linha:
            continue
        linha = _RE_REFERENCIA.sub("", linha).strip()
        # 5 palavras é o piso de uma frase de conteúdo; abaixo disso é rótulo.
        if len(linha.split()) < 5 or not _RE_FIM_DE_FRASE.search(linha):
            continue
        uteis.append(linha)

    # Linhas idênticas repetidas são navegação/rodapé (a Wikipédia repete o
    # menu no topo e na barra lateral). Parágrafo real não se repete literal.
    resultado = "\n".join(dict.fromkeys(uteis)).strip()

    if len(resultado) > limite:
        # Corta na última palavra inteira, para não entregar sílaba partida.
        resultado = resultado[:limite].rsplit(" ", 1)[0] + " […texto truncado]"
    return resultado


# Instanciada sem URL fixa: com `website_url` no construtor, o crewai-tools
# trava a ferramenta naquela página só (mesmo motivo das ferramentas de disco).
_raspador = ScrapeWebsiteTool()


@tool("Ler Página Web")
def ler_pagina_web(url: str) -> str:
    """Abre uma URL http/https e devolve o TEXTO da página, sem o menu de
    navegação e limitado a ~4000 caracteres. Use DEPOIS da 'Busca na Web',
    para ler de fato a fonte mais confiável que ela retornou (documentação
    oficial, Wikipédia, artigo) em vez de confiar só no resumo do resultado.
    Exemplo: 'https://pt.wikipedia.org/wiki/Equilíbrio_de_Nash'."""
    problema = _url_segura(url)
    if problema:
        return f"BLOQUEADO: {problema}"

    try:
        bruto = _raspador.run(website_url=url.strip())
    except Exception as erro:
        # Nunca levanta: o laço ReAct precisa do erro como texto para tentar
        # outra fonte, igual ao que busca_web e executar_comando já fazem.
        return f"Não consegui ler a página ({erro}). Tente outra fonte."

    limpo = _enxugar_pagina(bruto or "")
    if not limpo:
        return (
            "A página não devolveu texto legível (pode ser vídeo, PDF ou "
            "conteúdo carregado por JavaScript). Tente outra fonte."
        )
    return f"Conteúdo de {url.strip()}:\n\n{limpo}"

# ---------------------------------------------------------------------------
# FERRAMENTAS DE DISCO (NÍVEL 1: LER PASTAS E ARQUIVOS LOCAIS)
# ---------------------------------------------------------------------------
# Instanciadas SEM caminho fixo de propósito: se um "directory"/"file_path"
# for passado aqui no construtor, o crewai-tools trava a ferramenta só
# naquele caminho e o agente perde a capacidade de pedir outro. Deixando
# em branco, o "directory"/"file_path" vira um argumento que o próprio
# agente escolhe a cada chamada — é isso que permite tanto o padrão
# (BASE_DIR, reforçado no prompt do Pesquisador) quanto navegar para
# outro caminho dentro do projeto.
#
# Segurança: o crewai-tools valida todo caminho internamente e REJEITA
# qualquer um que caia fora do diretório de trabalho atual (BASE_DIR,
# quando você roda "python main.py" na pasta do projeto) — path
# traversal (“..”) e caminhos absolutos de outras pastas são bloqueados
# por padrão. Isso é proposital: o Pesquisador também lê conteúdo da
# web aberta (busca_web), e destravar leitura irrestrita de disco num
# agente que processa texto não confiável é um vetor clássico de
# exfiltração via prompt injection. Para liberar outras pastas por sua
# conta e risco, defina a variável de ambiente antes de rodar:
#   CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true
class FileReadToolUTF8(FileReadTool):
    """Igual ao FileReadTool original, mas lendo em UTF-8.

    Bug confirmado na versão instalada: o crewai-tools abre o arquivo com
    open(file_path, "r") sem encoding — no Windows isso usa cp1252 e
    quebra qualquer acento (testado lendo o próprio README.md do
    projeto). errors="replace" evita crash ao ler um arquivo binário.
    """

    def _run(
        self,
        file_path: str | None = None,
        start_line: int | None = 1,
        line_count: int | None = None,
    ) -> str:
        file_path = file_path or self.file_path
        start_line = start_line or 1
        line_count = line_count or None

        if file_path is None:
            return "Error: No file path provided. Please provide a file path either in the constructor or as an argument."

        try:
            file_path = validate_file_path(file_path)
        except ValueError as e:
            return f"Error: Invalid file path: {e!s}"

        display_path = format_path_for_display(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                if start_line == 1 and line_count is None:
                    return file.read()

                start_idx = max(start_line - 1, 0)
                selected_lines = [
                    line
                    for i, line in enumerate(file)
                    if i >= start_idx
                    and (line_count is None or i < start_idx + line_count)
                ]
                if not selected_lines and start_idx > 0:
                    return f"Error: Start line {start_line} exceeds the number of lines in the file."
                return "".join(selected_lines)
        except FileNotFoundError:
            return f"Error: File not found at path: {display_path}"
        except PermissionError:
            return f"Error: Permission denied when trying to read file: {display_path}"
        except Exception as e:
            return f"Error: Failed to read file {display_path}. {format_error_for_display(e)}"


leitor_pastas = DirectoryReadTool()
leitor_arquivos = FileReadToolUTF8()

# ---------------------------------------------------------------------------
# FERRAMENTA DE TERMINAL (NÍVEL 4: AGENTE EXECUTOR — SANDBOX RÍGIDO)
# ---------------------------------------------------------------------------
# Princípios do sandbox (um modelo local pequeno VAI errar comandos, então
# a segurança não pode depender do bom senso dele):
#   1. shell=False sempre — sem interpretação de ; | && > etc.
#   2. Allowlist de executáveis: python, git (subcomandos limitados), pytest.
#   3. git: subcomandos só de leitura/commit local (nada de push, checkout,
#      reset, clean) e bloqueio de flags que executam código ou redirecionam
#      o repo (-c, -C, --git-dir, --exec, --output...).
#   4. python/pytest: só scripts que resolvem DENTRO do projeto (Path.resolve
#      fecha o buraco de "..\" e symlink); sem -c/-m arbitrário.
#   5. cwd travado em BASE_DIR, timeout de 2 minutos, saída truncada.
#   6. A ferramenta NUNCA levanta exceção — devolve o erro como texto para o
#      laço ReAct do agente se corrigir.

GIT_SUBCOMANDOS_PERMITIDOS = {"status", "diff", "log", "add", "commit", "show", "branch"}
# Flags do git que executam código externo, mudam o repo-alvo ou gravam em
# caminho arbitrário. Comparadas por prefixo para pegar as formas "--flag=x".
GIT_FLAGS_PROIBIDAS = (
    "-c", "-C", "--config-env", "--git-dir", "--work-tree", "--exec",
    "--upload-pack", "--receive-pack", "--output", "--ext-diff",
)


def _dentro_do_projeto(caminho: str) -> bool:
    """True se o caminho (relativo ou absoluto) resolve dentro de BASE_DIR."""
    try:
        alvo = (BASE_DIR / caminho).resolve() if not Path(caminho).is_absolute() else Path(caminho).resolve()
        return alvo == BASE_DIR or BASE_DIR in alvo.parents
    except (OSError, ValueError):
        return False


def _validar_argv(argv: list[str]) -> str | None:
    """Valida o comando contra o sandbox. Retorna a mensagem de erro, ou None se ok."""
    if not argv:
        return "Comando vazio."

    exe = Path(argv[0]).name.lower().removesuffix(".exe")

    if exe == "git":
        if len(argv) < 2 or argv[1] not in GIT_SUBCOMANDOS_PERMITIDOS:
            return (
                "Subcomando git não permitido. Use apenas: "
                + ", ".join(sorted(GIT_SUBCOMANDOS_PERMITIDOS)) + "."
            )
        for arg in argv[1:]:
            if arg.startswith("-") and any(
                arg == f or arg.startswith(f + "=") for f in GIT_FLAGS_PROIBIDAS
            ):
                return f"Flag git bloqueada pelo sandbox: {arg}"
        # commit sem -m abriria um editor interativo e travaria até o timeout
        if argv[1] == "commit" and not any(
            a in ("-m", "--message") or a.startswith("--message=") for a in argv[2:]
        ):
            return 'git commit exige a mensagem inline: git commit -m "mensagem"'
        return None

    if exe in ("python", "python3", "py"):
        alvos = [a for a in argv[1:] if not a.startswith("-")]
        flags = [a for a in argv[1:] if a.startswith("-")]
        if flags:
            return (
                f"Opções do python bloqueadas pelo sandbox: {' '.join(flags)}. "
                "Use apenas: python <script.py> [argumentos simples]."
            )
        if not alvos or not alvos[0].endswith(".py"):
            return "Informe o caminho de um script .py dentro do projeto."
        if not _dentro_do_projeto(alvos[0]):
            return f"Script fora da pasta do projeto — bloqueado: {alvos[0]}"
        return None

    if exe == "pytest":
        for a in argv[1:]:
            if a.startswith("-"):
                return f"Opções do pytest bloqueadas pelo sandbox: {a}"
            if not _dentro_do_projeto(a):
                return f"Caminho fora da pasta do projeto — bloqueado: {a}"
        return None

    return (
        f"Executável '{exe}' não permitido. O sandbox aceita apenas: "
        "python <script.py>, git (status/diff/log/add/commit/show/branch) e pytest."
    )


@tool("Executar Comando no Terminal")
def executar_comando(comando: str) -> str:
    """Executa um comando de terminal DENTRO da pasta do projeto e retorna a
    saída (stdout/stderr) com o código de saída. Comandos permitidos:
    'python <script.py>' (script dentro do projeto), 'git status|diff|log|
    add|commit|show|branch ...' e 'pytest [caminho]'. Exemplos:
    'python output/relogio.py' | 'git status' | 'git commit -m "mensagem"'."""
    # Normaliza \ para / antes do shlex (o modo posix comeria as barras
    # invertidas de caminhos Windows; git e python aceitam / nos dois SOs).
    try:
        argv = shlex.split(comando.replace("\\", "/"), posix=True)
    except ValueError as erro:
        return f"Não consegui interpretar o comando ({erro}). Simplifique as aspas."

    problema = _validar_argv(argv)
    if problema:
        return f"BLOQUEADO: {problema}"

    exe = Path(argv[0]).name.lower().removesuffix(".exe")
    if exe in ("python", "python3", "py"):
        argv = [sys.executable, *argv[1:]]       # python do venv atual
    elif exe == "pytest":
        argv = [sys.executable, "-m", "pytest", *argv[1:]]

    try:
        resultado = subprocess.run(
            argv,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            shell=False,
            # Sem stdin: qualquer comando que tente ser interativo falha na
            # hora, em vez de ficar pendurado esperando input até o timeout.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return "O comando estourou o tempo limite de 120s e foi interrompido."
    except (OSError, ValueError) as erro:
        return f"Falha ao executar ({erro})."

    saida = (
        f"exit code: {resultado.returncode}\n"
        f"--- stdout ---\n{resultado.stdout or '(vazio)'}\n"
        f"--- stderr ---\n{resultado.stderr or '(vazio)'}"
    )
    if len(saida) > 3000:
        saida = saida[:2200] + "\n[... saída truncada ...]\n" + saida[-600:]
    return saida

# ---------------------------------------------------------------------------
# PRE-FLIGHT CHECK (DIAGNÓSTICO)
# ---------------------------------------------------------------------------
def diagnostico_ollama(url: str = OLLAMA_URL, timeout: float = 5.0) -> str | None:
    """Verifica se o servidor e os modelos estão ativos antes de rodar a IA.
    Isso evita tracebacks longos e confusos no terminal caso o Ollama esteja fechado."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as resposta:
            dados = json.load(resposta)
    except (OSError, http.client.HTTPException, json.JSONDecodeError):
        return (
            f"Servidor Ollama não encontrado em {url}.\n"
            "  Certifique-se de que o aplicativo do Ollama está rodando."
        )

    bases = {m.get("name", "").split(":")[0] for m in dados.get("models", [])}
    if MODELO_OLLAMA in bases:
        return None
    if MODELO_BASE not in bases:
        return f"O Ollama está rodando, mas o modelo '{MODELO_BASE}' não foi baixado."
    return f"O modelo '{MODELO_BASE}' existe, mas falta criar a variante de 8k."

# ---------------------------------------------------------------------------
# A AGÊNCIA E SUAS TAREFAS (CREW)
# ---------------------------------------------------------------------------
def montar_equipe(
    tema: str,
    llm: LLM,
    usar_memoria: bool = False,
    contexto_extra: str = "",
    usar_gerente: bool = False,
) -> Crew:
    """Cria os agentes e define como eles interpretam o pedido do usuário.

    usar_memoria: liga a memória de longo prazo (ChromaDB + embeddings
    locais). Default FALSE de propósito — com o qwen 7B, a memória às vezes
    faz o modelo recuperar e repetir a saída de uma tarefa anterior parecida
    (contaminação), então ela é opt-in via o toggle do painel.

    contexto_extra: texto de arquivo(s) anexado(s) no painel. É injetado no
    TOPO da tarefa do Pesquisador como fonte primária, para a agência já ter
    o conteúdo "na ponta da língua" sem precisar ler do disco.

    usar_gerente: se True, roda em PROCESSO HIERÁRQUICO — um Gerente (LLM)
    lê o pedido e convoca dinamicamente os especialistas (incl. Arquiteto e
    Especialista em Integrações). Default FALSE porque o hierárquico é bem
    mais lento no 7B (várias rodadas de delegação) E abre mão da rede de
    segurança determinística do código (escolher_saida): a saída final passa
    a ser a síntese do Gerente. O modo sequencial (padrão) continua sendo o
    pipeline confiável Pesquisador -> Escritor -> Revisor.
    """
    # Bloco de contexto (vazio se nada foi anexado) — prefixado à Tarefa 1.
    bloco_contexto = ""
    if contexto_extra.strip():
        bloco_contexto = (
            "CONTEXTO FORNECIDO PELO USUÁRIO (fonte PRIMÁRIA — use este "
            "conteúdo para responder e NÃO tente relê-lo do disco):\n"
            "<<<CONTEXTO>>>\n"
            f"{contexto_extra.strip()}\n"
            "<<<FIM DO CONTEXTO>>>\n\n"
        )

    # BUSCA SEMÂNTICA (RAG) — criada aqui (lazy) porque o construtor da
    # DirectorySearchTool já cria um cliente/coleção ChromaDB; deixá-la no
    # nível do módulo pesaria todo import. Usa o embedder LOCAL nomic-embed-text.
    # A ferramenta indexa a pasta RECURSIVAMENTE, então o prompt orienta o
    # agente a apontar para uma SUBPASTA específica, nunca a raiz (que tem .venv).
    busca_semantica = DirectorySearchTool(
        name="Busca Semântica na Pasta",
        config={"embedding_model": CONFIG_EMBEDDER},
    )

    # AGENTE 0: O Analista de Requisitos (Engenheiro de Prompt) — atua PRIMEIRO.
    analista = Agent(
        role="Analista de Requisitos",
        goal="Transformar o pedido do usuário num briefing técnico preciso.",
        backstory=(
            "Você é um engenheiro de prompt. Pega um pedido simples ou vago e "
            "o expande num BRIEFING objetivo: o objetivo real, o público-alvo, "
            "as dúvidas técnicas a responder, as restrições e os requisitos do "
            "sistema. Você NÃO resolve o pedido nem pesquisa — apenas deixa "
            "explícito, em tópicos, o que precisa ser investigado, para que o "
            "Pesquisador não deixe nada escapar. Seja conciso e técnico."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    # AGENTE 1: O Pesquisador-Curador (rigoroso, não um mero coletor)
    pesquisador = Agent(
        role="Pesquisador e Coletor de Dados",
        goal=f"Produzir um dossiê rigoroso e verificável sobre: {tema}",
        backstory=(
            "Você é um investigador rigoroso — um CURADOR, não um mero coletor. "
            "Trabalha com QUATRO fontes e escolhe a certa antes de agir:\n"
            "1) DISCO LOCAL (leitura direta) — 'List files in directory' e "
            "'Read a file's content' para listar uma pasta ou ler um arquivo "
            "PEQUENO. Ao citar código, extraia o BLOCO EXATO, sem parafrasear.\n"
            "2) BUSCA SEMÂNTICA ('Busca Semântica na Pasta') — para ENCONTRAR "
            "trechos em arquivos GRANDES ou numa base de código; informe uma "
            "SUBPASTA específica, NUNCA a raiz do projeto.\n"
            "3) BUSCA NA WEB ('Busca na Web') — devolve os resultados E JÁ ABRE "
            "as páginas mais confiáveis, entregando o TEXTO delas num bloco "
            "'TEXTO DAS FONTES MAIS CONFIÁVEIS'. É desse bloco que saem as "
            "definições, os nomes e os números que você vai afirmar. Se ele "
            "vier vazio, refaça a busca ou diga que não encontrou fonte — "
            "nunca preencha o vazio de memória.\n"
            "4) LER PÁGINA WEB ('Ler Página Web') — para abrir uma página "
            "ESPECÍFICA que a busca não leu e você quer conferir.\n\n"
            f"Só é possível ler/buscar em DISCO dentro de '{BASE_DIR}'.\n\n"
            "RIGOR (obrigatório): avalie a CREDIBILIDADE de cada fonte; "
            "descarte especulação; e entregue um DOSSIÊ ESTRUTURADO — tópicos, "
            "cada um com o achado, a FONTE (URL ou caminho do arquivo) e, "
            "quando for código, o trecho exato. Nunca jogue texto corrido cru "
            "para o próximo agente."
        ),
        llm=llm,
        tools=[
            busca_web,
            ler_pagina_web,
            leitor_pastas,
            leitor_arquivos,
            busca_semantica,
        ],
        # 10 e não 8: a web agora exige DOIS passos (buscar, depois ler), e um
        # link que não devolve texto obriga a tentar o próximo.
        max_iter=10,
        allow_delegation=False,
        verbose=True,
    )

    # AGENTE 2: O Criador Híbrido
    escritor = Agent(
        role="Especialista em Resolução (Programador/Redator)",
        goal=(
            "Classificar o pedido pela Regra de Intencionalidade ANTES de "
            "escrever, e responder no formato certo para essa categoria."
        ),
        backstory=(
            "Antes de escrever qualquer coisa, você classifica o pedido em "
            "UMA das duas categorias abaixo — nunca as duas ao mesmo tempo:\n\n"
            "AÇÃO DE CRIAÇÃO — o pedido usa um verbo como 'criar/crie', "
            "'escrever/escreva', 'desenvolver/desenvolva' ou 'gerar/gere' "
            "pedindo um script, programa, função ou aplicação NOVOS. "
            "Saída: CÓDIGO CRU — só o código, sem introdução, explicação "
            "ou conclusão ao redor.\n\n"
            "AÇÃO DE ANÁLISE — o pedido usa um verbo como 'explicar/"
            "explique', 'resumir/resuma', 'analisar/analise', 'auditar/"
            "audite' ou 'comentar/comente' sobre um arquivo, código ou "
            "projeto EXISTENTE. Saída: RELATÓRIO TÉCNICO EM PROSA, bem "
            "estruturado (títulos e parágrafos) — NUNCA um bloco de "
            "código, mesmo que o assunto seja um arquivo de código.\n\n"
            "REGRA CRÍTICA CONTRA ALUCINAÇÃO: numa Ação de Análise, é "
            "PROIBIDO escrever um script que leia, resuma ou analise o "
            "arquivo em nome do usuário — isso seria fugir da tarefa. "
            "VOCÊ é quem lê os achados que o Pesquisador já levantou e "
            "escreve a análise diretamente, em português, agora.\n\n"
            "FERRAMENTA DE TERMINAL: você pode usar 'Executar Comando no "
            "Terminal' para: (a) rodar um script .py JÁ EXISTENTE no "
            "projeto e verificar se funciona; (b) rodar pytest; (c) rodar "
            "git status/diff/log; (d) git add/commit SOMENTE se o usuário "
            "pediu commit explicitamente no pedido. Use no máximo 2 "
            "execuções por tarefa. Se o comando for bloqueado ou falhar, "
            "siga em frente e termine a tarefa sem ele — a ferramenta é "
            "opcional, o texto/código final é obrigatório."
        ),
        llm=llm,
        tools=[executar_comando],
        max_iter=6,              # mesmas mitigações de tool-use do Pesquisador
        allow_delegation=False,
        verbose=True,
    )

    # AGENTE 3: O Validador de Saída
    revisor = Agent(
        role="Engenheiro de Qualidade",
        goal=(
            "Confirmar que a saída do Escritor respeitou a Regra de "
            "Intencionalidade e remover qualquer 'lixo' conversacional."
        ),
        backstory=(
            "Você recebe o pedido original e o rascunho do Escritor, e "
            "aplica a mesma classificação: AÇÃO DE CRIAÇÃO (código) ou "
            "AÇÃO DE ANÁLISE (prosa).\n"
            "- Se for CRIAÇÃO: É OBRIGATÓRIO que a sua resposta final seja "
            "APENAS o bloco de código cru e funcional. Remova TUDO que for "
            "texto explicativo, mas NUNCA, sob nenhuma hipótese, remova, "
            "narre ou resuma a sintaxe do código.\n"
            "- Se for ANÁLISE: mantenha o texto em prosa. Corte apenas "
            "tagarelice conversacional ('Aqui está o resumo:', 'Espero "
            "ter ajudado'). Se o rascunho veio como código em vez de "
            "prosa (erro do Escritor), REESCREVA como relatório técnico "
            "em prosa usando os achados do Pesquisador — nunca devolva "
            "código numa Ação de Análise."
        ),
        llm=llm,
        verbose=True,
    )

    # AGENTES 4 e 5: só participam no MODO GERENTE (hierárquico). O Gerente
    # os convoca quando o pedido justifica; no modo sequencial não entram.
    arquiteto = Agent(
        role="Arquiteto de Software",
        goal=(
            "Rascunhar a arquitetura de sistemas complexos antes de programar: "
            "bibliotecas, módulos e estrutura de diretórios."
        ),
        backstory=(
            "Você é um arquiteto de software. Para pedidos de sistemas ou "
            "infraestrutura (ex.: KVM/QEMU, um player de música, uma API), "
            "você define quais bibliotecas usar e como organizar os arquivos "
            "e módulos — um plano enxuto e prático para o Escritor programar "
            "em cima. Não escreve o código final; entrega a planta."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    especialista_integracoes = Agent(
        role="Especialista em Integrações",
        goal=(
            "Garantir que webhooks e requisições de API do código estejam "
            "bem estruturados para automação (n8n, ClickUp, etc.)."
        ),
        backstory=(
            "Você é especialista em integrações e automação. Lê o código "
            "gerado e valida os pontos de integração: payloads JSON, headers, "
            "métodos HTTP, tratamento de erro dos webhooks/requests — deixando "
            "tudo pronto para ser consumido por plataformas como o n8n. "
            "Ajusta o código quando necessário, mas devolve CÓDIGO, não prosa."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    # TAREFA 0: Análise de Requisitos (expande o pedido num briefing)
    analise_requisitos = Task(
        description=(
            f"{bloco_contexto}"
            f"O pedido do usuário é: '{tema}'.\n"
            "Transforme-o num BRIEFING técnico curto e objetivo, em tópicos: "
            "(a) objetivo real; (b) público-alvo; (c) as dúvidas técnicas "
            "concretas a responder; (d) restrições/requisitos relevantes. "
            "NÃO resolva o pedido nem pesquise — só explicite o que investigar."
        ),
        expected_output=(
            "Um briefing em tópicos com objetivo, público, dúvidas técnicas e "
            "requisitos — as instruções precisas para o Pesquisador."
        ),
        agent=analista,
    )

    # TAREFA 1: Pesquisa Dinâmica (guiada pelo briefing; entrega um dossiê)
    pesquisa = Task(
        description=(
            f"{bloco_contexto}"
            f"Pedido original: '{tema}'. Siga o BRIEFING do Analista de "
            "Requisitos (no contexto) para não deixar nada escapar.\n"
            "- Se houver CONTEXTO FORNECIDO acima, trate-o como fonte "
            "primária e NÃO use ferramentas de disco para reobtê-lo.\n"
            "- Arquivo/pasta/projeto pequeno: use 'List files in directory' / "
            f"'Read a file's content' (padrão: '{BASE_DIR}').\n"
            "- Arquivo grande / base de código: use a 'Busca Semântica na "
            "Pasta' (subpasta específica) e extraia o trecho exato.\n"
            "- Informação externa/atual: 'Busca na Web' para achar os links e "
            "em seguida 'Ler Página Web' no melhor deles, para ler o texto. "
            "Afirmação sem página lida é especulação — não entra no dossiê.\n"
            "- Script simples sem dependência de arquivos: pode pular as "
            "ferramentas e esboçar só a lógica.\n"
            "Avalie a credibilidade das fontes e entregue um DOSSIÊ "
            "estruturado (achado + fonte/URL + código exato quando houver)."
        ),
        expected_output=(
            "Um dossiê estruturado em tópicos: cada achado com sua fonte (URL "
            "ou caminho) e, quando for código, o trecho exato — pronto para o "
            "Escritor usar."
        ),
        agent=pesquisador,
        context=[analise_requisitos],
    )

    # TAREFA 2: Regra de Intencionalidade (Aqui a IA classifica antes de agir)
    redacao = Task(
        description=(
            f"O pedido original do usuário é: '{tema}'.\n\n"
            "PASSO 1 — CLASSIFIQUE o pedido em uma das duas categorias:\n"
            "- AÇÃO DE CRIAÇÃO: contém um verbo como 'criar/crie', "
            "'escrever/escreva', 'desenvolver/desenvolva' ou 'gerar/gere' "
            "pedindo um script, programa, função ou aplicação NOVOS.\n"
            "- AÇÃO DE ANÁLISE: contém um verbo como 'explicar/explique', "
            "'resumir/resuma', 'analisar/analise', 'auditar/audite' ou "
            "'comentar/comente' sobre um arquivo, código ou projeto "
            "EXISTENTE.\n\n"
            "PASSO 2 — EXECUTE conforme a classificação:\n"
            "- Se CRIAÇÃO: escreva o código completo e funcional. NÃO faça "
            "introdução, desenvolvimento em texto ou conclusão. Vá direto "
            "ao código.\n"
            "- Se ANÁLISE: escreva um relatório técnico em prosa (título, "
            "seções, conclusão) explicando o que foi encontrado nos "
            "achados do Pesquisador. NUNCA responda com um script/"
            "programa — isso é sempre errado numa Ação de Análise, mesmo "
            "que o assunto seja um arquivo de código."
        ),
        expected_output=(
            "Para Ação de Criação: código cru 100% funcional, sem texto ao "
            "redor. Para Ação de Análise: relatório técnico em prosa, sem "
            "nenhum bloco de código."
        ),
        agent=escritor,
        context=[pesquisa], # O Escritor usa o que o Pesquisador encontrou
    )

    # TAREFA 3: O Filtro Final (também aplica a Regra de Intencionalidade)
    revisao = Task(
        description=(
            f"Revise o trabalho do Escritor para o pedido: '{tema}'.\n"
            "1) Classifique de novo o pedido original: AÇÃO DE CRIAÇÃO ou "
            "AÇÃO DE ANÁLISE (mesmos critérios da Tarefa 2).\n"
            "2) Se CRIAÇÃO: remova o texto explicativo/saudações que estão "
            "AO REDOR do código e devolva APENAS o código. NUNCA apague o "
            "código gerado pelo Escritor — copie TODAS as linhas dele "
            "exatamente. Se ficar em dúvida, prefira devolver o código do "
            "Escritor inteiro a devolver uma explicação sem código.\n"
            "3) Se ANÁLISE: garanta que a saída é um relatório técnico em "
            "prosa — SEM bloco de código. Se o Escritor entregou código "
            "por engano, reescreva como texto corrido usando os achados "
            "do Pesquisador (disponíveis no contexto desta tarefa)."
        ),
        expected_output=(
            "Para Ação de Criação: a versão final do código, absolutamente "
            "limpa e pronta para uso. Para Ação de Análise: o relatório "
            "técnico final em prosa, sem nenhum código."
        ),
        agent=revisor,
        context=[pesquisa, redacao], # Precisa da pesquisa para reescrever em prosa se o Escritor errar
    )

    kwargs_crew = {"verbose": True}
    if usar_memoria:
        # Só configura memória (e o embedder local) quando pedida — com ela
        # desligada, nem o embedder nomic-embed-text é necessário.
        kwargs_crew["memory"] = True
        kwargs_crew["embedder"] = CONFIG_EMBEDDER

    if usar_gerente:
        # MODO GERENTE (hierárquico): um Gerente auto-criado (manager_llm) lê a
        # tarefa e delega aos 5 especialistas via ferramentas de delegação (que,
        # com supports_function_calling=False, rodam pelo caminho ReAct — testado
        # e funcional com o qwen local). Uma ÚNICA tarefa descreve o objetivo e
        # o fluxo; o Gerente decide quem convocar.
        entrega = Task(
            description=(
                f"{bloco_contexto}"
                f"Pedido do usuário: '{tema}'.\n\n"
                "Coordene a equipe para entregar o melhor resultado, seguindo a "
                "REGRA DE INTENCIONALIDADE:\n"
                "- CRIAÇÃO (criar/escrever/desenvolver/gerar um script, programa "
                "ou função): a entrega final é APENAS código cru e funcional, "
                "sem texto explicativo ao redor.\n"
                "- ANÁLISE (explicar/resumir/analisar/auditar algo existente): a "
                "entrega final é um relatório técnico em prosa, SEM bloco de "
                "código.\n\n"
                "FLUXO SUGERIDO (delegue só o que fizer sentido):\n"
                "1. Analista de Requisitos: transforma o pedido num briefing "
                "técnico (objetivo, dúvidas, requisitos) antes de tudo.\n"
                "2. Pesquisador: levanta fatos (web/disco/busca semântica) e "
                "monta um dossiê com fontes.\n"
                "3. Arquiteto: SÓ para sistemas/infra complexos — rascunha "
                "bibliotecas e estrutura antes de programar.\n"
                "4. Escritor: produz o código ou a prosa.\n"
                "5. Especialista em Integrações: SÓ se envolver n8n/webhook/API — "
                "valida os pontos de integração do código.\n"
                "6. Engenheiro de Qualidade: limpa a saída final (tira tagarelice; "
                "NUNCA apaga o código numa Ação de Criação)."
            ),
            expected_output=(
                "A entrega final no formato certo pela Regra de Intencionalidade: "
                "código cru (Criação) ou relatório em prosa (Análise)."
            ),
        )
        kwargs_crew.update(
            agents=[analista, pesquisador, arquiteto, escritor,
                    especialista_integracoes, revisor],
            tasks=[entrega],
            process=Process.hierarchical,
            manager_llm=llm,
        )
    else:
        # MODO SEQUENCIAL (padrão): Analista -> Pesquisador -> Escritor -> Revisor.
        kwargs_crew.update(
            agents=[analista, pesquisador, escritor, revisor],
            tasks=[analise_requisitos, pesquisa, redacao, revisao],
            process=Process.sequential,
        )
    return Crew(**kwargs_crew)

# ---------------------------------------------------------------------------
# PROCESSAMENTO DE SAÍDA E SALVAMENTO DE ARQUIVOS
# ---------------------------------------------------------------------------
def _limpar_artefatos(texto: str) -> str:
    """Remove os vazamentos INTERNOS do modelo (rótulos "Final Answer",
    classificações, tool-calls JSON, andaime ReAct) — SEM tocar nas crases
    de markdown. É a base compartilhada: para o painel colorir, o conteúdo
    precisa manter os blocos ```lang, mas nunca pode carregar esse lixo.
    """
    # 1. Vazamentos do pensamento lógico do modelo (ex: "Final Answer:")
    texto = re.sub(
        r"^[\s*#]*(resposta final|final answer)[ \t]*:?[ \t]*\**[ \t]*(?:\r?\n)*",
        "",
        texto.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # 2. Rótulos internos de classificação e memória vazados
    texto = re.sub(
        r"^[\s*#]*(Classifica[çc][ãa]o|A[CÇ][ÃA]O DE AN[ÁA]LISE|A[CÇ][ÃA]O DE CRIA[CÇ][ÃA]O|Mem[óo]rias Relevantes|Mem[óo]rias Relevantes Fim|A[çc][ãa]o de An[áa]lise Conclu[íi]da|Resposta Final)[ \t]*:?[ \t]*\**.*?(?:\r?\n)*",
        "",
        texto,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # 3. Tool-calls internas vazadas no FINAL (ex.: JSON '{"name":
    # "search_memory", ...}'). Só remove se citar uma ferramenta conhecida —
    # um dict legítimo do código do usuário nunca é afetado.
    ferramentas_internas = (
        r"search_memory|busca[_ ]?na[_ ]?web|read_a_file'?s?_?content"
        r"|list_files_in_directory|executar_comando"
    )
    texto = re.sub(
        rf'(?:```(?:json)?[ \t]*\r?\n?)?\{{\s*"name"\s*:\s*"(?:{ferramentas_internas})"'
        rf"[\s\S]{{0,800}}?\}}\s*(?:```)?\s*$",
        "",
        texto,
        flags=re.IGNORECASE,
    )

    # 4. Andaime ReAct vazado no final. Só dispara a partir de "Thought:" (o
    # início canônico do laço) para não apagar código que tenha comentários
    # como "# Observation: ...".
    texto = re.sub(r"(?:^|\n)[ \t]*Thought[ \t]*:[\s\S]*$", "", texto)

    return texto.strip()


def limpar_saida(texto: str) -> str:
    """Versão para DISCO: limpa os artefatos internos E arranca as crases de
    markdown, para o arquivo salvo (.py/.sh/...) ficar cru e executável.
    O painel usa outra via (com as crases) — ver conteudo_para_exibicao."""
    texto = _limpar_artefatos(texto)
    texto = re.sub(r"^```[a-zA-Z0-9]*\n", "", texto)   # crase de abertura
    texto = re.sub(r"\n```$", "", texto)               # crase de fechamento
    return texto.strip()


def conteudo_para_exibicao(bruto: str, tipo: str, linguagem: str) -> str:
    """Versão do conteúdo para o PAINEL / editor de código: sem os artefatos
    internos do modelo, mas COM as crases de markdown para o Streamlit ativar
    o syntax highlighting. Para código, devolve exatamente UM bloco cercado
    com a linguagem certa (mesmo que o modelo não tenha cercado). Para prosa,
    devolve o texto com o markdown preservado. É a contrapartida do arquivo
    em disco: aqui as crases FICAM; em salvar_relatorio elas são removidas."""
    if tipo == "codigo":
        return f"```{linguagem}\n{limpar_saida(bruto)}\n```"
    return _limpar_artefatos(bruto)


def _extrair_bloco_codigo(texto: str) -> str:
    """Se houver um bloco cercado (```lang ... ```), devolve só o miolo;
    senão devolve o texto inteiro. Usado ao recuperar o rascunho do Escritor."""
    m = re.search(r"```[a-zA-Z0-9]*\n(.*?)\n```", texto, flags=re.DOTALL)
    return m.group(1) if m else texto


def _parece_codigo(texto: str) -> bool:
    """Heurística: o texto tem cara de código-fonte (vs. prosa explicativa)?
    Usada para detectar quando o Revisor apagou o código e sobrou só texto."""
    t = texto.strip()
    if not t:
        return False
    if "```" in t:
        return True
    marcadores = (
        "def ", "import ", "class ", "function ", "#!/", "</", "select ",
        "print(", "console.log", "public ", "package ", "=>", "});",
        "#include", "sudo ", "apt ", "$(", "echo ",
    )
    baixo = t.lower()
    if any(m in baixo for m in marcadores):
        return True
    # Relatório em prosa com fontes é sinal FORTE de que NÃO é código
    # (evita classificar um dossiê de pesquisa como .py — as linhas de
    # links markdown terminam em ")" e enganavam a heurística abaixo).
    if re.search(r"\]\(https?://", t) or re.search(r"(?im)^\s*fontes?\s*:", t):
        return False
    linhas = [ln for ln in t.splitlines() if ln.strip()]
    if linhas:
        # Só pontuação FORTE de código conta (; { }); ":" e ")" foram removidos
        # por serem ambíguos (cabeçalhos de prosa "Fontes:" e links "](url)").
        # Python/shell continuam pegos pela indentação e pelos marcadores acima.
        sinais = sum(
            1 for ln in linhas
            if ln[:1] in " \t" or ln.rstrip().endswith((";", "{", "}"))
        )
        if sinais / len(linhas) >= 0.4:
            return True
    return False


# Verbos de CRIAÇÃO de código (palavra inteira). Usados como segundo sinal
# de que se esperava código, para a rede de segurança atuar mesmo sem um
# "salve como <arquivo>" explícito no pedido.
_VERBOS_CRIACAO = (
    "crie", "criar", "cria",
    "escreva", "escrever", "escreve",
    "desenvolva", "desenvolver", "desenvolve",
    "gere", "gerar", "gera",
    "programe", "programar",
    "implemente", "implementar",
)
_RE_VERBO_CRIACAO = re.compile(
    r"\b(?:" + "|".join(_VERBOS_CRIACAO) + r")\b", re.IGNORECASE
)


def _pede_criacao(tema: str) -> bool:
    """True se o pedido usa um verbo de criação (crie/desenvolva/gere/...)."""
    return bool(_RE_VERBO_CRIACAO.search(tema))


def escolher_saida(crew_output, tema: str) -> str:
    """Rede de segurança do Bug 1: numa tarefa de CRIAÇÃO, se a saída FINAL do
    Revisor perdeu o código e virou só prosa (ex.: 'A função X foi criada
    para...'), recupera o rascunho do Escritor (ou do Pesquisador) via
    tasks_output. Em tarefa de análise, ou quando o código está presente,
    devolve a saída final normalmente.

    "Espera código" quando há um nome de arquivo de código pedido OU quando o
    tema usa um verbo de criação — assim a rede atua mesmo sem "salve como"
    (ex.: 'Crie a função de crossfade...'). É seguro deixar amplo: a recuperação
    só acontece se o rascunho REALMENTE parecer código (_parece_codigo), então
    'escreva um artigo' (prosa) nunca vira resgate indevido."""
    final = getattr(crew_output, "raw", None) or str(crew_output)

    nome = nome_arquivo_pedido(tema)
    espera_codigo = (
        (bool(nome) and Path(nome).suffix.lower() in LINGUAGEM_POR_EXT)
        or _pede_criacao(tema)
    )
    if not espera_codigo or _parece_codigo(final):
        return final

    # O final não tem código, mas era p/ ter: varre as saídas anteriores (todas
    # menos a última, que é o próprio 'final') de TRÁS pra FRENTE e recupera o
    # rascunho de código mais recente. A varredura reversa é robusta ao tamanho
    # do pipeline (não depende de índice fixo): com o Analista de Requisitos, o
    # rascunho do Escritor deixou de ser sempre o índice 1.
    try:
        tarefas = list(getattr(crew_output, "tasks_output", []) or [])
    except Exception:
        tarefas = []
    for tarefa in reversed(tarefas[:-1]):
        bruto = getattr(tarefa, "raw", "") or ""
        if _parece_codigo(bruto):
            return _extrair_bloco_codigo(bruto)
    return final

def nome_arquivo_pedido(tema: str) -> str | None:
    """Extrai o nome do arquivo de SAÍDA pedido, se houver.

    Prioriza uma menção logo após uma deixa explícita ("salve/salvar/
    gravar como", "save as"). Sem essa deixa, descarta qualquer menção
    precedida por um verbo de LEITURA (ex.: "leia o README.md") antes
    de aplicar o fallback antigo (última menção restante) — sem isso,
    o nome do arquivo de ENTRADA seria confundido com o de saída (bug
    real: aconteceu ao testar a leitura de disco do Nível 1, e teria
    sobrescrito output/README.md). Heurística por palavras-chave, não
    é um parser de linguagem natural: frases fora desses padrões caem
    no fallback com timestamp, que é o comportamento seguro.
    """
    # Alternação construída a partir de EXT_SAIDA (fonte única) — assim .sh,
    # .js, .sql etc. também são reconhecidos como nome de saída válido.
    alt_ext = "|".join(sorted((e[1:] for e in EXT_SAIDA), key=len, reverse=True))
    extensao = rf"[a-zA-Z0-9_-]+\.(?:{alt_ext})"

    # NOVO: Adicionado .*? para lidar com palavras extras entre o comando e a extensão.
    # Ex: Captura com sucesso "salve o código como servidor_teste.py"
    cue_salvar = re.search(
        rf"(?:salve|salvar|grave|gravar|nomeie).*?como\s+({extensao})"
        rf"|save\s+as\s+({extensao})",
        tema,
        flags=re.IGNORECASE,
    )
    if cue_salvar:
        return cue_salvar.group(1) or cue_salvar.group(2)

    # Mesmos verbos de "Ação de Análise" da Regra de Intencionalidade
    # (montar_equipe): qualquer um deles antes de um nome de arquivo indica
    # ENTRADA, não o nome de saída pedido.
    verbos_leitura = (
        "leia", "ler", "lido", "abra", "abrir", "read", "open", "veja",
        "analise", "analisar", "resuma", "resumir", "explique", "explicar",
        "audite", "auditar", "comente", "comentar",
    )
    # Olha a frase INTEIRA antes do nome do arquivo, não só uma janela
    # curta: "Explique ... no arquivo main.py" tem o verbo bem longe do
    # nome do arquivo. Efeito colateral aceitável: um pedido tipo "Leia X
    # e crie Y.py" (sem "salve como") também cairia no fallback com
    # timestamp em vez de nomear como Y.py — comportamento seguro, só
    # perde o nome "bonito".
    candidatos = [
        m.group(0)
        for m in re.finditer(extensao, tema)
        if not any(v in tema[:m.start()].lower() for v in verbos_leitura)
    ]
    return candidatos[-1] if candidatos else None


def salvar_relatorio(conteudo: str, tema: str) -> Path:
    """Analisa o prompt do usuário para decidir o nome e a extensão do arquivo,
    criando a pasta de destino automaticamente se não existir."""

    conteudo = limpar_saida(conteudo)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nome_arquivo = nome_arquivo_pedido(tema)
    if nome_arquivo is None:
        # Fallback inteligente: se o usuário não deu o nome, adivinha a
        # extensão pelo CONTEÚDO que a agência gerou (não pelo verbo do
        # pedido). Motivo: "escreva um artigo" tem verbo de criação mas
        # produz PROSA — salvar como .py estaria errado; _parece_codigo
        # olha o que foi de fato produzido. Código -> .py (ou .sh se tiver
        # shebang de shell); prosa -> .txt.
        carimbo = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if _parece_codigo(conteudo):
            primeira = conteudo.lstrip().splitlines()[0] if conteudo.strip() else ""
            if primeira.startswith("#!") and ("bash" in primeira or "sh" in primeira):
                nome_arquivo = f"script_{carimbo}.sh"
            else:
                nome_arquivo = f"script_{carimbo}.py"
        else:
            nome_arquivo = f"relatorio_{carimbo}.txt"

    arquivo = OUTPUT_DIR / nome_arquivo
    arquivo.write_text(conteudo + "\n", encoding="utf-8")
    return arquivo

# ---------------------------------------------------------------------------
# NÚCLEO DE EXECUÇÃO (MAIN)
# ---------------------------------------------------------------------------
def main() -> int:
    # Captura tudo que você digitou no terminal após 'python main.py'
    tema = " ".join(sys.argv[1:]).strip() or TEMA_PADRAO

    problema = diagnostico_ollama()
    if problema:
        print(f"[Cortexus] ERRO: {problema}")
        return 1

    # Inicia a conexão com o Ollama local (fábrica compartilhada com a api.py)
    llm = criar_llm()

    print(f"[Cortexus] Iniciando a equipe para o tema: {tema}\n")
    equipe = montar_equipe(tema, llm)
    
    try:
        crew_output = equipe.kickoff()
    except Exception as erro:
        print(f"\n[Cortexus] Falha crítica ao executar a equipe: {erro}")
        return 1

    # escolher_saida recupera o código se o Revisor o tiver apagado (Bug 1);
    # salvar_relatorio já aplica limpar_saida internamente.
    arquivo = salvar_relatorio(escolher_saida(crew_output, tema), tema)
    print(f"\n[Cortexus] Concluído! Arquivo salvo em: {arquivo}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())