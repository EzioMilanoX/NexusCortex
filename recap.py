"""R.E.C.A.P. — revisão semanal dos projetos pessoais (conceito SYS-16).

Duas metades, com a fronteira no lugar de sempre deste projeto:

  1. COLETA (determinística, sem LLM): para cada repositório de
     PersonalProjects, os commits da janela (data + mensagem), as alterações
     locais sem commit e a linha do projeto no PROJECTS.md. Vira um resumo em
     texto com teto de tamanho — o qwen 7B roda com num_ctx 8192, e o resumo
     entra no prompt de dois agentes.
  2. EQUIPE (LLM): Revisor de Projetos escreve o relatório, o Crítico de
     Texto o confere contra os mesmos dados (corta progresso que não está nos
     commits) e, no formato "roteiro", o Roteirista adapta para duas vozes no
     formato que o Estúdio de Podcast narra.

O palpite de próximo passo é só texto: nada aqui escreve no PROJECTS.md.

Uso:
  python recap.py                      # relatório dos últimos 7 dias
  python recap.py --dias 14 --roteiro  # roteiro para o Estúdio de Podcast
  python recap.py --so-coletar         # só a coleta (sem Ollama), na tela
  python recap.py --raiz D:/outros     # outra pasta de projetos

A pasta de projetos padrão é a pasta-mãe deste projeto (PersonalProjects);
a variável NCX_PROJETOS troca. A API (POST /recap) usa SEMPRE essa pasta
configurada no servidor — nunca um caminho vindo do cliente, porque a API não
tem autenticação e git log de pasta arbitrária seria vazamento de dados.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from crewai import LLM, Crew, Process, Task

import main
import podcast as estudio
from especialistas import (
    criar_critico_texto,
    criar_revisor_projetos,
    criar_roteirista_semanal,
)

FORMATOS = ("relatorio", "roteiro")
DIAS_MAX = 60

# Teto do resumo que vai para o prompt. ~4500 caracteres ≈ 1,3 mil tokens; o
# resumo aparece em DUAS tarefas (Revisor e Crítico), e ainda precisa sobrar
# contexto para backstory, saída do Revisor e resposta dentro dos 8k.
LIMITE_RESUMO = 4500
LIMITE_PROXIMO_PASSO = 200
LIMITE_MENSAGEM = 90

_STATUS_POR_EMOJI = {
    "🟢": "ativo",
    "🟡": "protótipo",
    "🔵": "lançado",
    "⚪": "pausado",
    "🟣": "ideia",
}
# Status em que ficar sem commit é o esperado — não entra como "parado".
_STATUS_QUIETOS = frozenset({"pausado", "ideia"})


class ErroRecap(RuntimeError):
    """Falha da revisão semanal com mensagem pronta para o usuário."""


# ---------------------------------------------------------------------------
# 1. COLETA
# ---------------------------------------------------------------------------
@dataclass
class EstadoProjeto:
    nome: str
    pasta: str
    tem_git: bool
    status: str = ""
    proximo_passo: str = ""
    commits: list[tuple[str, str]] = field(default_factory=list)  # (data, mensagem)
    ultimo_commit: str = ""
    alteracoes_locais: int = 0

    @property
    def mexido(self) -> bool:
        """Teve commit na janela OU tem trabalho local ainda sem commit."""
        return bool(self.commits) or self.alteracoes_locais > 0

    @property
    def parado(self) -> bool:
        return self.tem_git and not self.mexido and self.status not in _STATUS_QUIETOS

    def como_dict(self) -> dict:
        dados = asdict(self)
        dados["commits"] = [{"data": d, "mensagem": m} for d, m in self.commits]
        dados["parado"] = self.parado
        return dados


Git = Callable[[Path, list[str]], str]


def executar_git(repo: Path, args: list[str]) -> str:
    """Roda um git de LEITURA no repositório e devolve o stdout ("" se falhar).

    Mesmas cercas do terminal do main.py: shell=False, stdin fechado, timeout.
    Nunca levanta — um repositório quebrado não pode derrubar a coleta inteira.
    """
    try:
        resultado = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            shell=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ""
    return resultado.stdout if resultado.returncode == 0 else ""


def raiz_padrao() -> Path:
    """Pasta de projetos: NCX_PROJETOS, ou a pasta-mãe do NexusCortex."""
    return Path(os.environ.get("NCX_PROJETOS") or main.BASE_DIR.parent)


def _extrair_proximo(celula: str) -> str:
    """Da célula 'Próximo passo' (que acumula histórico), fica o último
    'Próximo: ...' quando existe — é a parte que interessa ao revisor."""
    texto = celula.replace("**", "").strip()
    achados = list(re.finditer(r"Pr[óo]ximo[^:|]{0,20}:\s*", texto, flags=re.IGNORECASE))
    if achados:
        texto = texto[achados[-1].end():]
    texto = " ".join(texto.split())
    if len(texto) > LIMITE_PROXIMO_PASSO:
        texto = texto[: LIMITE_PROXIMO_PASSO - 1].rstrip() + "…"
    return texto


def ler_projects_md(caminho: Path) -> dict[str, dict]:
    """Lê a tabela do PROJECTS.md: {nome: {"status", "proximo_passo"}}.

    Colunas esperadas: Projeto | Status | Stack | O que é | Último commit |
    Próximo passo. Uma barra "|" dentro do texto do próximo passo não quebra a
    leitura: tudo a partir da 6ª coluna é juntado de volta.
    """
    if not caminho.is_file():
        return {}
    projetos: dict[str, dict] = {}
    for linha in caminho.read_text(encoding="utf-8", errors="replace").splitlines():
        linha = linha.strip()
        # A chave é o ALVO do link (o nome da pasta), não o texto: "[Sistema
        # de Estudos Automatizado (S.E.A.)](<...Automatizado(S.E.A.)>/)" tem
        # um espaço a mais no texto do que na pasta.
        # Alvo entre <> pode ter parênteses ("<NexusCortex(NCX)>/"); sem <>,
        # vai até o primeiro ")".
        m = re.match(r"^\|\s*\[([^\]]+)\]\((?:<([^>]*)>|([^)<>]*))/?\)", linha)
        if not m:
            continue
        celulas = [c.strip() for c in linha.strip("|").split("|")]
        if len(celulas) < 6:
            continue
        emoji = celulas[1].strip()
        alvo = (m.group(2) or m.group(3) or "").strip().rstrip("/")
        pasta = alvo or m.group(1).strip()
        projetos[pasta] = {
            "status": _STATUS_POR_EMOJI.get(emoji, emoji),
            "proximo_passo": _extrair_proximo("|".join(celulas[5:])),
        }
    return projetos


def coletar_estado(raiz: Path, dias: int = 7, git: Git | None = None) -> list[EstadoProjeto]:
    """Junta, por projeto, o git da janela e a linha do PROJECTS.md.

    Projeto = subpasta da raiz (menos as que começam com '_' ou '.', como
    _templates). `git` é injetável para os testes não dependerem de repositório
    de verdade.
    """
    git = git or executar_git
    raiz = Path(raiz)
    if not raiz.is_dir():
        raise ErroRecap(f"Pasta de projetos não encontrada: {raiz}")

    tabela = ler_projects_md(raiz / "PROJECTS.md")
    estados: list[EstadoProjeto] = []
    for pasta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        if pasta.name.startswith(("_", ".")):
            continue
        linha = tabela.get(pasta.name, {})
        estado = EstadoProjeto(
            nome=pasta.name,
            pasta=str(pasta),
            tem_git=(pasta / ".git").exists(),
            status=linha.get("status", ""),
            proximo_passo=linha.get("proximo_passo", ""),
        )
        if estado.tem_git:
            log = git(pasta, [
                "log", f"--since={dias}.days.ago", "--no-merges",
                "--date=short", "--pretty=format:%ad%x09%s",
            ])
            for registro in log.splitlines():
                data, _, mensagem = registro.partition("\t")
                if mensagem.strip():
                    estado.commits.append((data.strip(), mensagem.strip()))
            estado.ultimo_commit = git(
                pasta, ["log", "-1", "--date=short", "--pretty=format:%ad"]
            ).strip()
            estado.alteracoes_locais = sum(
                1 for ln in git(pasta, ["status", "--porcelain"]).splitlines() if ln.strip()
            )
        estados.append(estado)
    return estados


def _bloco_projeto(estado: EstadoProjeto, max_commits: int) -> str:
    linhas = [f"### {estado.nome} (status: {estado.status or 'sem registro'})"]
    if estado.proximo_passo:
        linhas.append(f"Próximo passo anotado: {estado.proximo_passo}")
    for data, mensagem in estado.commits[:max_commits]:
        if len(mensagem) > LIMITE_MENSAGEM:
            mensagem = mensagem[: LIMITE_MENSAGEM - 1] + "…"
        linhas.append(f"- {data} {mensagem}")
    if len(estado.commits) > max_commits:
        linhas.append(f"- (+{len(estado.commits) - max_commits} commits)")
    if not estado.commits:
        linhas.append("- nenhum commit na janela")
    if estado.alteracoes_locais:
        linhas.append(f"- {estado.alteracoes_locais} arquivo(s) alterado(s) sem commit")
    return "\n".join(linhas)


def resumo_para_prompt(
    estados: list[EstadoProjeto], dias: int, limite: int = LIMITE_RESUMO
) -> str:
    """Texto que vai para o prompt: ativos em detalhe, parados numa linha.

    Para caber no teto, corta commits por projeto (6 -> 4 -> 2 -> 1) antes de
    truncar o texto — perder o 5º commit de um projeto é melhor que perder um
    projeto inteiro do fim da lista.
    """
    mexidos = sorted(
        (e for e in estados if e.mexido),
        key=lambda e: (-len(e.commits), e.nome.lower()),
    )
    parados = [e for e in estados if e.parado]
    quietos = [e for e in estados if e.tem_git and not e.mexido and not e.parado]
    sem_git = [e for e in estados if not e.tem_git]

    def rodape() -> list[str]:
        partes = []
        if parados:
            partes.append("## Parados (sem commit e sem alteração local)\n" + "\n".join(
                f"- {e.nome}: último commit {e.ultimo_commit or 'desconhecido'}"
                + (f"; próximo passo anotado: {e.proximo_passo}" if e.proximo_passo else "")
                for e in parados
            ))
        if quietos:
            partes.append("Pausados/ideias sem movimento (esperado): "
                          + ", ".join(e.nome for e in quietos))
        if sem_git:
            partes.append("Sem git (fora da revisão): " + ", ".join(e.nome for e in sem_git))
        return partes

    cabecalho = (
        f"Janela: últimos {dias} dias (até {datetime.now():%Y-%m-%d}). "
        f"{len(mexidos)} projeto(s) com movimento, {len(parados)} parado(s)."
    )
    texto = ""
    for max_commits in (6, 4, 2, 1):
        corpo = [cabecalho, "## Projetos com movimento"]
        corpo += [_bloco_projeto(e, max_commits) for e in mexidos] or ["(nenhum)"]
        texto = "\n\n".join(corpo + rodape())
        if len(texto) <= limite:
            return texto
    return texto[: limite - 30].rstrip() + "\n[... resumo truncado ...]"


# ---------------------------------------------------------------------------
# 2. EQUIPE
# ---------------------------------------------------------------------------
def _bloco_dados(resumo: str) -> str:
    return f"<<<DADOS DA SEMANA>>>\n{resumo}\n<<<FIM DOS DADOS>>>\n\n"


def montar_equipe_recap(
    resumo: str,
    llm: LLM,
    formato: str = "relatorio",
    minutos: int = 5,
) -> Crew:
    """Revisor -> Crítico (-> Roteirista, no formato 'roteiro'). Sequencial.

    O Crítico recebe os DADOS de novo (e não só o relatório): conferir contra
    a fonte é o trabalho dele. É o que ataca o risco anotado no blueprint —
    "um 7B confunde projetos parecidos e inventa progresso".
    """
    if formato not in FORMATOS:
        raise ErroRecap(f"Formato desconhecido: {formato}. Use: {', '.join(FORMATOS)}")

    revisor = criar_revisor_projetos(llm)
    critico = criar_critico_texto(llm)

    relatorio = Task(
        description=(
            _bloco_dados(resumo)
            + "Com base SOMENTE nos dados acima, monte a revisão semanal em "
            "markdown, neste formato:\n"
            "# Revisão semanal\n"
            "## Resumo da semana\n(2 ou 3 frases)\n"
            "## Projetos com movimento\n"
            "### <nome exato do projeto>\n"
            "- Feito: <o que os commits mostram>\n"
            "- Em relação ao próximo passo anotado: <andou / desviou / não tocou>\n"
            "- Palpite de próximo passo: <uma ação concreta>\n"
            "## Parados\n- <projeto>: <último commit> — <o que faltava, se anotado>\n"
            "## Foco sugerido\n(um projeto e uma frase do porquê)"
        ),
        expected_output="A revisão semanal em markdown, no formato pedido.",
        agent=revisor,
    )
    conferencia = Task(
        description=(
            _bloco_dados(resumo)
            + "Confira a revisão semanal do Revisor de Projetos (no contexto) "
            "contra os DADOS acima:\n"
            "- corte qualquer progresso que não esteja nos commits ou nas "
            "alterações locais;\n"
            "- corrija nome de projeto trocado ou inventado;\n"
            "- garanta que todo projeto da lista de parados está em 'Parados';\n"
            "- mantenha os palpites de próximo passo, marcados como palpite.\n"
            "Devolva APENAS a revisão final corrigida, no mesmo formato "
            "markdown, sem comentar a revisão que você fez."
        ),
        expected_output="A revisão semanal corrigida, em markdown, sem comentários.",
        agent=critico,
        context=[relatorio],
    )

    agentes = [revisor, critico]
    tarefas = [relatorio, conferencia]
    if formato == "roteiro":
        roteirista = criar_roteirista_semanal(llm)
        nomes = [a.nome for a in estudio.APRESENTADORES_PADRAO]
        palavras = max(120, int(minutos * estudio.PALAVRAS_POR_MINUTO))
        roteiro = Task(
            description=(
                "Adapte a revisão semanal corrigida (no contexto) num diálogo "
                f"de cerca de {minutos} minutos (~{palavras} palavras) entre "
                f"{nomes[0]} e {nomes[1]}.\n"
                f"FORMATO OBRIGATÓRIO: uma fala por linha, '{nomes[0]}: texto' "
                f"ou '{nomes[1]}: texto'. Nada além das falas — sem título, sem "
                "rubrica, sem markdown, sem lista.\n"
                "Ordem: abertura da revisão da semana; os projetos com "
                "movimento; uma menção aos parados; fechamento com o foco "
                "sugerido. Use só os fatos da revisão."
            ),
            expected_output=f"Falas no formato '{nomes[0]}: texto', uma por linha.",
            agent=roteirista,
            context=[conferencia],
        )
        agentes.append(roteirista)
        tarefas.append(roteiro)

    return Crew(agents=agentes, tasks=tarefas, process=Process.sequential, verbose=True)


def gerar_recap(
    dias: int = 7,
    formato: str = "relatorio",
    *,
    minutos: int = 5,
    raiz: Path | None = None,
    llm: LLM | None = None,
    git: Git | None = None,
    pasta_saida: Path | None = None,
) -> dict:
    """Coleta, roda a equipe e salva. Fonte única do CLI e do POST /recap."""
    if not 1 <= dias <= DIAS_MAX:
        raise ErroRecap(f"dias deve ficar entre 1 e {DIAS_MAX}.")
    if formato not in FORMATOS:
        raise ErroRecap(f"Formato desconhecido: {formato}. Use: {', '.join(FORMATOS)}")

    estados = coletar_estado(raiz or raiz_padrao(), dias, git)
    if not any(e.tem_git for e in estados):
        raise ErroRecap("Nenhum repositório git encontrado na pasta de projetos.")
    resumo = resumo_para_prompt(estados, dias)

    equipe = montar_equipe_recap(resumo, llm or main.criar_llm(), formato, minutos)
    saida = equipe.kickoff()
    bruto = main.limpar_saida(getattr(saida, "raw", None) or str(saida))

    pasta = pasta_saida or (main.OUTPUT_DIR / "recap")
    pasta.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    resultado = {
        "formato": formato,
        "dias": dias,
        "resumo": resumo,
        "com_movimento": [e.nome for e in estados if e.mexido],
        "parados": [e.nome for e in estados if e.parado],
    }
    if formato == "roteiro":
        # O parser do Estúdio é a garantia de formato (o 7B desobedece às
        # vezes): o que se salva é exatamente o que o /podcast vai narrar.
        falas = estudio.parse_roteiro(bruto)
        conteudo = "\n".join(f"{f.locutor}: {f.texto}" for f in falas) if falas else bruto
        arquivo = pasta / f"recap_roteiro_{carimbo}.txt"
        resultado.update(roteiro=conteudo, total_falas=len(falas))
    else:
        conteudo = bruto
        arquivo = pasta / f"recap_{carimbo}.md"
    arquivo.write_text(conteudo + "\n", encoding="utf-8")
    resultado.update(conteudo=conteudo, arquivo=str(arquivo))
    return resultado


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python recap.py",
        description="Revisão semanal dos projetos (R.E.C.A.P.) pela equipe local.",
    )
    parser.add_argument("--dias", type=int, default=7, help="janela em dias (padrão 7)")
    parser.add_argument("--roteiro", action="store_true",
                        help="sai como roteiro de podcast (Nome: fala) em vez de relatório")
    parser.add_argument("--minutos", type=int, default=5, help="duração-alvo do roteiro")
    parser.add_argument("--raiz", default=None,
                        help="pasta dos projetos (padrão: NCX_PROJETOS ou a pasta-mãe)")
    parser.add_argument("--so-coletar", action="store_true",
                        help="só mostra os dados coletados do git; não chama o Ollama")
    args = parser.parse_args(argv)
    raiz = Path(args.raiz) if args.raiz else raiz_padrao()

    try:
        if args.so_coletar:
            print(resumo_para_prompt(coletar_estado(raiz, args.dias), args.dias))
            return 0

        problema = main.diagnostico_ollama()
        if problema:
            print(f"[Recap] ERRO: {problema}")
            return 1
        res = gerar_recap(
            args.dias,
            "roteiro" if args.roteiro else "relatorio",
            minutos=args.minutos,
            raiz=raiz,
        )
    except ErroRecap as erro:
        print(f"[Recap] ERRO: {erro}")
        return 1

    print(f"\n[Recap] Concluído! Arquivo salvo em: {res['arquivo']}")
    if args.roteiro:
        print(f"[Recap] Para narrar: python podcast.py \"{res['arquivo']}\" --motor piper")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
