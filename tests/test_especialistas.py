"""Testes dos especialistas: revisão semanal (recap.py), Crítico de Texto
(critica.py), as fábricas de agentes (especialistas.py) e as rotas novas.

Herméticos como o resto da suíte: o git é um dublê injetado (exceto num teste
opcional que usa um repositório de verdade em tmp_path, pulado sem git), a
crew é trocada por um objeto falso no kickoff, e o diagnóstico do Ollama é
mockado nas rotas. Nada de rede, nada de LLM."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import critica
import especialistas
import main
import recap

PROJECTS_MD = """\
# PROJECTS.md

| Projeto | Status | Stack | O que é | Último commit | Próximo passo (palpite) |
| --- | --- | --- | --- | --- | --- |
| [Alfa](Alfa/) | 🟢 | Python | Jogo A. | 2026-11-20 | **v1 (2026-11-01):** coisas velhas. Próximo: música por fase |
| [Beta Projeto](<Beta Projeto>/) | 🟡 | Flutter | App B. | 2026-10-01 | Publicar a v0.2 |
| [Gama](Gama/) | ⚪ | Python | Pausado. | 2026-01-01 | Retomar algum dia |
| [Sem Git (S.G.)](<SemGit(S.G.)>/) | 🟣 | — | Rascunho. | — (sem git) | Decidir |
"""


def _montar_raiz(tmp_path: Path) -> Path:
    """Pasta de projetos falsa: Alfa (com commits), Beta (parado), Gama
    (pausado), SemGit (sem .git) e _templates (ignorado)."""
    raiz = tmp_path / "PersonalProjects"
    for nome in ("Alfa", "Beta Projeto", "Gama"):
        (raiz / nome / ".git").mkdir(parents=True)
    (raiz / "SemGit(S.G.)").mkdir()
    (raiz / "_templates" / ".git").mkdir(parents=True)
    (raiz / "PROJECTS.md").write_text(PROJECTS_MD, encoding="utf-8")
    return raiz


def _git_falso(respostas: dict[str, dict[str, str]]):
    """Dublê do executar_git: responde por (nome da pasta, subcomando)."""
    chamadas: list[tuple[str, list[str]]] = []

    def git(repo: Path, args: list[str]) -> str:
        chamadas.append((repo.name, args))
        sub = "ultimo" if args[:2] == ["log", "-1"] else args[0]
        return respostas.get(repo.name, {}).get(sub, "")

    git.chamadas = chamadas
    return git


GIT_PADRAO = {
    "Alfa": {
        "log": "2026-11-22\tfeat: chefe novo\n2026-11-20\tfix: colisão no canto",
        "ultimo": "2026-11-22",
        "status": " M alfa/jogo.py\n?? notas.md\n",
    },
    "Beta Projeto": {"ultimo": "2026-10-01"},
    "Gama": {"ultimo": "2026-01-01"},
}


# --------------------------------------------------------------------------
# Coleta (determinística)
# --------------------------------------------------------------------------
def test_ler_projects_md_usa_a_pasta_e_o_ultimo_proximo(tmp_path):
    raiz = _montar_raiz(tmp_path)
    tabela = recap.ler_projects_md(raiz / "PROJECTS.md")
    assert tabela["Alfa"] == {"status": "ativo", "proximo_passo": "música por fase"}
    # chave = alvo do link (nome da pasta), não o texto do link
    assert "SemGit(S.G.)" in tabela and "Sem Git (S.G.)" not in tabela
    assert tabela["Beta Projeto"]["status"] == "protótipo"
    assert tabela["Gama"]["status"] == "pausado"


def test_ler_projects_md_inexistente_devolve_vazio(tmp_path):
    assert recap.ler_projects_md(tmp_path / "nao_existe.md") == {}


def test_extrair_proximo_trunca():
    longo = "Próximo: " + "x" * 500
    assert len(recap._extrair_proximo(longo)) == recap.LIMITE_PROXIMO_PASSO


def test_coletar_estado_classifica_projetos(tmp_path):
    raiz = _montar_raiz(tmp_path)
    git = _git_falso(GIT_PADRAO)
    estados = {e.nome: e for e in recap.coletar_estado(raiz, 7, git)}

    assert "_templates" not in estados  # pastas com _ ficam de fora
    alfa = estados["Alfa"]
    assert alfa.commits == [("2026-11-22", "feat: chefe novo"), ("2026-11-20", "fix: colisão no canto")]
    assert alfa.alteracoes_locais == 2 and alfa.mexido and not alfa.parado
    assert estados["Beta Projeto"].parado            # protótipo sem movimento
    assert not estados["Gama"].parado                # pausado: quieto é o esperado
    assert not estados["SemGit(S.G.)"].tem_git
    # git só é chamado em repositório, com a janela pedida
    assert all(nome != "SemGit(S.G.)" for nome, _ in git.chamadas)
    assert any("--since=7.days.ago" in args for _, args in git.chamadas)


def test_coletar_estado_raiz_inexistente(tmp_path):
    with pytest.raises(recap.ErroRecap):
        recap.coletar_estado(tmp_path / "nada", 7, _git_falso({}))


def test_resumo_separa_movimento_parados_e_sem_git(tmp_path):
    estados = recap.coletar_estado(_montar_raiz(tmp_path), 7, _git_falso(GIT_PADRAO))
    resumo = recap.resumo_para_prompt(estados, 7)
    assert "### Alfa (status: ativo)" in resumo
    assert "feat: chefe novo" in resumo and "2 arquivo(s) alterado(s) sem commit" in resumo
    assert "## Parados" in resumo and "Beta Projeto: último commit 2026-10-01" in resumo
    assert "Pausados/ideias sem movimento (esperado): Gama" in resumo
    assert "Sem git (fora da revisão): SemGit(S.G.)" in resumo


def test_resumo_respeita_o_teto_cortando_commits_primeiro():
    estados = [
        recap.EstadoProjeto(
            nome=f"P{i}", pasta=".", tem_git=True, status="ativo",
            commits=[("2026-11-2%d" % (j % 10), f"commit {j} " + "y" * 60) for j in range(8)],
        )
        for i in range(6)
    ]
    resumo = recap.resumo_para_prompt(estados, 7, limite=2500)
    assert len(resumo) <= 2500
    # todos os projetos sobrevivem; o que some são os commits excedentes
    assert all(f"### P{i} " in resumo for i in range(6))
    assert "commits)" in resumo


def test_executar_git_nunca_levanta(tmp_path, monkeypatch):
    def _explode(*a, **k):
        raise OSError("git não instalado")

    monkeypatch.setattr(recap.subprocess, "run", _explode)
    assert recap.executar_git(tmp_path, ["status"]) == ""


def test_raiz_padrao_le_a_variavel(monkeypatch, tmp_path):
    monkeypatch.setenv("NCX_PROJETOS", str(tmp_path))
    assert recap.raiz_padrao() == tmp_path
    monkeypatch.delenv("NCX_PROJETOS")
    assert recap.raiz_padrao() == main.BASE_DIR.parent


@pytest.mark.skipif(shutil.which("git") is None, reason="git não instalado")
def test_coleta_com_repositorio_de_verdade(tmp_path):
    raiz = tmp_path / "proj"
    repo = raiz / "Real"
    repo.mkdir(parents=True)

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=repo, check=True, capture_output=True, stdin=subprocess.DEVNULL,
        )

    git("init", "-q")
    (repo / "a.txt").write_text("1", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-q", "-m", "feat: primeiro passo")
    (repo / "b.txt").write_text("2", encoding="utf-8")  # alteração local

    (estado,) = recap.coletar_estado(raiz, 7)
    assert [m for _, m in estado.commits] == ["feat: primeiro passo"]
    assert estado.alteracoes_locais == 1 and estado.ultimo_commit


# --------------------------------------------------------------------------
# Agentes e equipes (montadas de verdade, sem kickoff)
# --------------------------------------------------------------------------
def test_fabricas_de_agentes_sem_ferramentas():
    llm = main.criar_llm()
    agentes = [
        especialistas.criar_revisor_projetos(llm),
        especialistas.criar_critico_texto(llm),
        especialistas.criar_roteirista_semanal(llm),
    ]
    assert [a.role for a in agentes] == [
        "Revisor de Projetos", "Crítico de Texto", "Roteirista do Podcast Semanal",
    ]
    # Sem ferramenta e sem delegação: os fatos chegam pelo prompt.
    assert all(not a.tools and not a.allow_delegation for a in agentes)


def test_equipe_recap_relatorio_revisor_e_critico():
    equipe = recap.montar_equipe_recap("DADOS-XYZ", main.criar_llm())
    assert [a.role for a in equipe.agents] == ["Revisor de Projetos", "Crítico de Texto"]
    assert len(equipe.tasks) == 2
    # o Crítico recebe os dados de novo — conferir contra a fonte é o trabalho dele
    assert all("DADOS-XYZ" in t.description for t in equipe.tasks)
    assert equipe.tasks[1].context == [equipe.tasks[0]]


def test_equipe_recap_roteiro_tem_roteirista_com_os_apresentadores():
    equipe = recap.montar_equipe_recap("d", main.criar_llm(), "roteiro", minutos=3)
    assert equipe.agents[-1].role == "Roteirista do Podcast Semanal"
    descricao = equipe.tasks[-1].description
    for apresentador in recap.estudio.APRESENTADORES_PADRAO:
        assert f"{apresentador.nome}: texto" in descricao


def test_equipe_recap_formato_invalido():
    with pytest.raises(recap.ErroRecap):
        recap.montar_equipe_recap("d", main.criar_llm(), "video")


def test_equipe_critica_com_referencia_e_foco():
    equipe = critica.montar_equipe_critica(
        "texto ALVO", main.criar_llm(), foco="encurtar", referencia="fato REF"
    )
    assert [a.role for a in equipe.agents] == ["Crítico de Texto"]
    descricao = equipe.tasks[0].description
    assert "texto ALVO" in descricao and "fato REF" in descricao and "encurtar" in descricao


def test_equipe_critica_sem_referencia_pede_marcacao():
    descricao = critica.montar_equipe_critica("t", main.criar_llm()).tasks[0].description
    assert "[verificar]" in descricao and "<<<REFERENCIA>>>" not in descricao


# --------------------------------------------------------------------------
# gerar_recap / gerar_critica (crew falsa)
# --------------------------------------------------------------------------
class _SaidaFalsa:
    def __init__(self, raw):
        self.raw = raw


class _EquipeFalsa:
    def __init__(self, raw):
        self._raw = raw

    def kickoff(self):
        return _SaidaFalsa(self._raw)


def test_gerar_recap_relatorio_salva_md(tmp_path, monkeypatch):
    recebido = {}

    def montar(resumo, llm, formato, minutos):
        recebido.update(resumo=resumo, formato=formato)
        return _EquipeFalsa("Final Answer:\n# Revisão semanal\n## Resumo da semana\nAndou.")

    monkeypatch.setattr(recap, "montar_equipe_recap", montar)
    res = recap.gerar_recap(
        7, raiz=_montar_raiz(tmp_path), llm=object(), git=_git_falso(GIT_PADRAO),
        pasta_saida=tmp_path / "saida",
    )
    assert recebido["formato"] == "relatorio" and "Alfa" in recebido["resumo"]
    assert res["conteudo"].startswith("# Revisão semanal")  # limpar_saida aplicado
    assert res["arquivo"].endswith(".md") and Path(res["arquivo"]).is_file()
    assert res["com_movimento"] == ["Alfa"] and res["parados"] == ["Beta Projeto"]


def test_gerar_recap_roteiro_normaliza_pelo_parser_do_estudio(tmp_path, monkeypatch):
    bruto = "```\n**Rafa:** Bom dia, revisão da semana!\nBruno: O Alfa ganhou um chefe novo.\n```"
    monkeypatch.setattr(recap, "montar_equipe_recap", lambda *a, **k: _EquipeFalsa(bruto))
    res = recap.gerar_recap(
        7, "roteiro", raiz=_montar_raiz(tmp_path), llm=object(),
        git=_git_falso(GIT_PADRAO), pasta_saida=tmp_path,
    )
    assert res["total_falas"] == 2
    assert res["roteiro"].splitlines()[0] == "Rafa: Bom dia, revisão da semana!"
    assert res["arquivo"].endswith(".txt")
    assert Path(res["arquivo"]).read_text(encoding="utf-8").startswith("Rafa:")


@pytest.mark.parametrize("kwargs", [{"dias": 0}, {"dias": 999}, {"formato": "video"}])
def test_gerar_recap_valida_entrada(tmp_path, kwargs):
    with pytest.raises(recap.ErroRecap):
        recap.gerar_recap(raiz=tmp_path, llm=object(), git=_git_falso({}), **kwargs)


def test_gerar_recap_sem_repositorio(tmp_path):
    (tmp_path / "SoPasta").mkdir()
    with pytest.raises(recap.ErroRecap, match="Nenhum repositório"):
        recap.gerar_recap(raiz=tmp_path, llm=object(), git=_git_falso({}))


def test_gerar_critica_salva(tmp_path, monkeypatch):
    saida = "## Problemas encontrados\nNenhum problema relevante.\n## Versão revisada\nOi."
    monkeypatch.setattr(critica, "montar_equipe_critica", lambda *a, **k: _EquipeFalsa(saida))
    res = critica.gerar_critica("Oi.", llm=object(), pasta_saida=tmp_path)
    assert "## Versão revisada" in res["conteudo"]
    assert Path(res["arquivo"]).parent == tmp_path


@pytest.mark.parametrize(
    "texto,referencia",
    [("   ", ""), ("a" * critica.MAX_CARACTERES, "b")],
)
def test_validar_pedido_critica(texto, referencia):
    with pytest.raises(critica.ErroCritica):
        critica.validar_pedido(texto, referencia)


# --------------------------------------------------------------------------
# CLIs (sem Ollama)
# --------------------------------------------------------------------------
def test_cli_recap_so_coletar_nao_chama_ollama(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(recap, "executar_git", _git_falso(GIT_PADRAO))
    monkeypatch.setattr(main, "diagnostico_ollama", lambda: pytest.fail("não devia checar o Ollama"))
    assert recap.main_cli(["--so-coletar", "--raiz", str(_montar_raiz(tmp_path))]) == 0
    assert "### Alfa" in capsys.readouterr().out


def test_cli_recap_ollama_fora(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "diagnostico_ollama", lambda: "Ollama fora do ar")
    assert recap.main_cli(["--raiz", str(tmp_path)]) == 1
    assert "Ollama fora do ar" in capsys.readouterr().out


def test_cli_critica_arquivo_inexistente(tmp_path, capsys):
    assert critica.main_cli([str(tmp_path / "nao_existe.md")]) == 1
    assert "não consegui ler" in capsys.readouterr().out


def test_cli_critica_roda_e_salva(tmp_path, monkeypatch, capsys):
    alvo = tmp_path / "texto.md"
    alvo.write_text("Um texto qualquer.", encoding="utf-8")
    monkeypatch.setattr(main, "diagnostico_ollama", lambda: None)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(critica, "montar_equipe_critica", lambda *a, **k: _EquipeFalsa("ok"))
    monkeypatch.setattr(main, "criar_llm", lambda: object())
    assert critica.main_cli([str(alvo), "--foco", "clareza"]) == 0
    assert (tmp_path / "out" / "critica").is_dir()
    assert "Concluído" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Rotas: /recap/coleta, /recap, /critica
# --------------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: None)
    return TestClient(api.app)


def test_rota_recap_coleta_usa_a_pasta_do_servidor(client, monkeypatch, tmp_path):
    monkeypatch.setenv("NCX_PROJETOS", str(_montar_raiz(tmp_path)))
    monkeypatch.setattr(recap, "executar_git", _git_falso(GIT_PADRAO))
    r = client.get("/recap/coleta", params={"dias": 14})
    assert r.status_code == 200
    d = r.json()
    assert d["dias"] == 14 and d["parados"] == ["Beta Projeto"]
    alfa = next(p for p in d["projetos"] if p["nome"] == "Alfa")
    assert alfa["commits"][0] == {"data": "2026-11-22", "mensagem": "feat: chefe novo"}


def test_rota_recap_coleta_pasta_inexistente(client, monkeypatch, tmp_path):
    monkeypatch.setenv("NCX_PROJETOS", str(tmp_path / "nada"))
    assert client.get("/recap/coleta").status_code == 404


@pytest.mark.parametrize("dias", [0, 61])
def test_rota_recap_coleta_dias_fora_da_faixa(client, dias):
    assert client.get("/recap/coleta", params={"dias": dias}).status_code == 400


def test_rota_recap_cria_job_e_conclui(client, monkeypatch):
    def gerar(dias, formato, minutos):
        return {"conteudo": "# Revisão", "arquivo": "output/recap/r.md",
                "formato": formato, "parados": ["Beta"]}

    monkeypatch.setattr(recap, "gerar_recap", gerar)
    r = client.post("/recap", json={"dias": 7, "formato": "relatorio"})
    assert r.status_code == 200
    job = client.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "concluido" and job["tipo"] == "recap"
    assert job["arquivo_salvo"] == "output/recap/r.md" and job["parados"] == ["Beta"]


def test_rota_recap_erro_vira_status_erro(client, monkeypatch):
    def gerar(*a, **k):
        raise recap.ErroRecap("Nenhum repositório git encontrado")

    monkeypatch.setattr(recap, "gerar_recap", gerar)
    r = client.post("/recap", json={})
    job = client.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "erro" and "Nenhum repositório" in job["detalhe"]


@pytest.mark.parametrize(
    "corpo",
    [{"formato": "video"}, {"dias": 0}, {"minutos": 0}, {"webhook_url": "ftp://x"}],
)
def test_rota_recap_valida_entrada(client, corpo):
    assert client.post("/recap", json=corpo).status_code == 400


def test_rota_recap_ollama_fora(client, monkeypatch):
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: "Ollama fora do ar")
    assert client.post("/recap", json={}).status_code == 503


def test_rota_critica_cria_job_e_conclui(client, monkeypatch):
    recebido = {}

    def gerar(texto, foco, referencia):
        recebido.update(texto=texto, foco=foco, referencia=referencia)
        return {"conteudo": "## Versão revisada\nOk.", "arquivo": "output/critica/c.md"}

    monkeypatch.setattr(critica, "gerar_critica", gerar)
    r = client.post("/critica", json={"texto": "Olá", "foco": "tom", "referencia": "ref"})
    assert r.status_code == 200
    job = client.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "concluido" and job["tipo"] == "critica"
    assert job["conteudo"].startswith("## Versão revisada")
    assert recebido == {"texto": "Olá", "foco": "tom", "referencia": "ref"}


@pytest.mark.parametrize(
    "corpo",
    [{"texto": "  "}, {"texto": "x" * (critica.MAX_CARACTERES + 1)}],
)
def test_rota_critica_valida_texto(client, corpo):
    assert client.post("/critica", json=corpo).status_code == 400


def test_rota_critica_ollama_fora(client, monkeypatch):
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: "Ollama fora do ar")
    assert client.post("/critica", json={"texto": "oi"}).status_code == 503
