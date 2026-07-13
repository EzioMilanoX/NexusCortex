"""Testes da camada determinística do main.py (sem LLM, sem Ollama, sem rede).

O que NÃO é testado aqui (de propósito): a saída do modelo Qwen em si e a
execução real da crew (kickoff) — são não-determinísticas, lentas e exigem
Ollama. Isso é avaliação manual/integração, não teste unitário. Aqui cobrimos
toda a lógica ao redor: classificação, limpeza, heurísticas, nomes de arquivo,
rede de segurança, sandbox e as ferramentas (com rede mockada)."""

from __future__ import annotations

import re

import pytest

import main


# --------------------------------------------------------------------------
# classificar_saida
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "arquivo,esperado",
    [
        ("x.py", ("codigo", "python")),
        ("setup_kvm.sh", ("codigo", "bash")),
        ("app.js", ("codigo", "javascript")),
        ("dados.json", ("codigo", "json")),
        ("relatorio.md", ("analise", "")),
        ("relatorio.txt", ("analise", "")),
        ("dados.csv", ("analise", "")),
    ],
)
def test_classificar_saida(arquivo, esperado):
    assert main.classificar_saida(arquivo) == esperado


# --------------------------------------------------------------------------
# _limpar_artefatos / limpar_saida / conteudo_para_exibicao
# --------------------------------------------------------------------------
def test_limpar_saida_remove_final_answer_e_crases():
    bruto = 'Final Answer:\n```python\ndef f():\n    return 1\n```'
    assert main.limpar_saida(bruto) == "def f():\n    return 1"


def test_limpar_saida_remove_toolcall_search_memory():
    bruto = (
        'Relatorio.\n\n{"name": "search_memory", "arguments": {"q": "x"}}'
    )
    out = main.limpar_saida(bruto)
    assert "search_memory" not in out and out.startswith("Relatorio")


def test_limpar_saida_react_thought_removido_mas_nao_observation_em_codigo():
    # "Thought:" (início do ReAct) é removido...
    assert "Thought" not in main.limpar_saida("ok\nThought: continuo\nAction: x")
    # ...mas um comentário shell "# Observation:" NÃO destrói o resto.
    sh = "#!/bin/bash\n# Observation: reinicia\nsystemctl restart x\necho ok"
    out = main.limpar_saida(sh)
    assert "systemctl restart x" in out and "echo ok" in out


def test_conteudo_para_exibicao_codigo_tem_crases_e_prosa_nao():
    exib = main.conteudo_para_exibicao("print(1)", "codigo", "python")
    assert exib == "```python\nprint(1)\n```"
    prosa = main.conteudo_para_exibicao("## Titulo\nTexto.", "analise", "")
    assert "```" not in prosa and prosa.startswith("## Titulo")


# --------------------------------------------------------------------------
# _parece_codigo
# --------------------------------------------------------------------------
def test_parece_codigo_detecta_codigo():
    assert main._parece_codigo("def f():\n    return 1") is True
    assert main._parece_codigo("#!/bin/bash\necho oi") is True


def test_parece_codigo_dossie_com_links_e_prosa():
    dossie = (
        "O with garante liberação de recursos.\n\nFontes:\n"
        "1. [Guia](https://pyai.io/x)\n2. [Doc](https://docs.python.org/y)"
    )
    assert main._parece_codigo(dossie) is False
    assert main._parece_codigo("Isto é uma frase de prosa comum.") is False


def test_extrair_bloco_codigo():
    assert main._extrair_bloco_codigo("intro\n```py\nX=1\n```\nfim") == "X=1"
    assert main._extrair_bloco_codigo("sem cerca") == "sem cerca"


# --------------------------------------------------------------------------
# nome_arquivo_pedido
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tema,esperado",
    [
        ("Crie e salve como app.py", "app.py"),
        ("crie um script bash e salve como setup_kvm.sh", "setup_kvm.sh"),
        ("Read config.json and save as summary.txt", "summary.txt"),
        ("Leia o README.md e resuma", None),          # verbo de leitura -> entrada
        ("Explique o que faz o main.py", None),        # idem
        ("Escreva um artigo sobre IA", None),          # sem nome de arquivo
    ],
)
def test_nome_arquivo_pedido(tema, esperado):
    assert main.nome_arquivo_pedido(tema) == esperado


# --------------------------------------------------------------------------
# _pede_criacao
# --------------------------------------------------------------------------
def test_pede_criacao():
    assert main._pede_criacao("Crie a função de crossfade") is True
    assert main._pede_criacao("desenvolva um endpoint") is True
    assert main._pede_criacao("Explique o crossfade") is False
    # palavra inteira: 'gere' não casa dentro de 'gerenciar'
    assert main._pede_criacao("gerenciar usuários") is False


# --------------------------------------------------------------------------
# salvar_relatorio (usa OUTPUT_DIR — redirecionado para tmp_path)
# --------------------------------------------------------------------------
@pytest.fixture
def output_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    return tmp_path


def test_salvar_nome_explicito(output_tmp):
    p = main.salvar_relatorio("print(1)", "salve como a.py")
    assert p.name == "a.py" and p.read_text(encoding="utf-8").strip() == "print(1)"


def test_salvar_fallback_codigo_vira_py(output_tmp):
    p = main.salvar_relatorio("def f():\n    return 42", "crie uma função")
    assert re.fullmatch(r"script_.+\.py", p.name)


def test_salvar_fallback_prosa_vira_txt(output_tmp):
    p = main.salvar_relatorio("A inteligência artificial é um campo.", "Escreva um artigo")
    assert re.fullmatch(r"relatorio_.+\.txt", p.name)


def test_salvar_fallback_shell_vira_sh(output_tmp):
    p = main.salvar_relatorio("#!/bin/bash\necho ola", "crie um script")
    assert re.fullmatch(r"script_.+\.sh", p.name)


# --------------------------------------------------------------------------
# escolher_saida (rede de segurança do "código some")
# --------------------------------------------------------------------------
class _FakeTask:
    def __init__(self, raw):
        self.raw = raw


class _FakeCrew:
    def __init__(self, final, tasks):
        self.raw = final
        self.tasks_output = [_FakeTask(t) for t in tasks]


def test_escolher_saida_recupera_codigo_com_pipeline_de_4():
    codigo = "def smooth_crossfade(a, b, d):\n    return a.append(b, crossfade=d)"
    prosa = "A função smooth_crossfade foi criada para suavizar."
    # analise, dossie(prosa), redacao(codigo), revisao(final=prosa)
    crew = _FakeCrew(final=prosa, tasks=["briefing", "dossie", codigo, prosa])
    out = main.escolher_saida(crew, "Crie a função de crossfade")
    assert "def smooth_crossfade" in out


def test_escolher_saida_sem_verbo_criacao_mantem_final():
    prosa = "Crossfade é uma técnica de transição."
    crew = _FakeCrew(final=prosa, tasks=["x", prosa])
    assert main.escolher_saida(crew, "Explique crossfade") == prosa


def test_escolher_saida_codigo_no_final_retorna_final():
    codigo = "def f():\n    return 1"
    crew = _FakeCrew(final=codigo, tasks=["x", codigo])
    assert main.escolher_saida(crew, "crie uma função") == codigo


# --------------------------------------------------------------------------
# Sandbox do terminal: _dentro_do_projeto / _validar_argv
# --------------------------------------------------------------------------
def test_dentro_do_projeto():
    assert main._dentro_do_projeto("main.py") is True
    assert main._dentro_do_projeto("output") is True
    assert main._dentro_do_projeto("C:/Windows/win.ini") is False
    assert main._dentro_do_projeto("../fora.py") is False


@pytest.mark.parametrize(
    "argv,ok",
    [
        (["git", "status"], True),
        (["git", "commit", "-m", "msg"], True),
        (["python", "output/x.py"], True),
        (["pytest", "tests"], True),
        (["git", "push", "origin", "main"], False),      # subcomando proibido
        (["git", "checkout", "--", "."], False),
        (["git", "-c", "core.x=y", "status"], False),    # injeção de config
        (["git", "log", "--output=/tmp/x"], False),
        (["git", "commit"], False),                      # sem -m abriria editor
        (["python", "-c", "import os"], False),          # flag bloqueada
        (["python", "C:/Windows/x.py"], False),          # fora do projeto
        (["rm", "-rf", "/"], False),                     # executável não permitido
        (["curl", "http://x"], False),
    ],
)
def test_validar_argv(argv, ok):
    resultado = main._validar_argv(argv)
    assert (resultado is None) == ok, resultado


# --------------------------------------------------------------------------
# busca_web (rede mockada)
# --------------------------------------------------------------------------
def _rodar_tool(tool, **kwargs):
    """Chama uma ferramenta CrewAI (@tool) de forma robusta a versões."""
    if hasattr(tool, "run"):
        return tool.run(**kwargs)
    return tool.func(**kwargs)  # fallback


def test_busca_web_sucesso(monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results=5):
            return [{"title": "T1", "href": "http://a", "body": "corpo"}]
    monkeypatch.setattr(main, "DDGS", FakeDDGS)
    out = _rodar_tool(main.busca_web, query="algo")
    assert "T1" in out and "http://a" in out


def test_busca_web_erro_nao_propaga(monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results=5):
            raise RuntimeError("rede caiu")
    monkeypatch.setattr(main, "DDGS", FakeDDGS)
    out = _rodar_tool(main.busca_web, query="algo")
    assert "falhou" in out.lower()  # devolve texto, não levanta


def test_busca_web_vazio(monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results=5):
            return []
    monkeypatch.setattr(main, "DDGS", FakeDDGS)
    out = _rodar_tool(main.busca_web, query="algo")
    assert "Nenhum resultado" in out


# --------------------------------------------------------------------------
# diagnostico_ollama (urlopen mockado)
# --------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, payload: bytes):
        self._p = payload
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def read(self, *a):
        return self._p


def test_diagnostico_ollama_ok(monkeypatch):
    payload = b'{"models":[{"name":"qwen2.5-8k:latest"}]}'
    monkeypatch.setattr(main.urllib.request, "urlopen", lambda *a, **k: _FakeResp(payload))
    assert main.diagnostico_ollama() is None


def test_diagnostico_ollama_variante_faltando(monkeypatch):
    payload = b'{"models":[{"name":"qwen2.5:7b"}]}'  # tem a base, falta a -8k
    monkeypatch.setattr(main.urllib.request, "urlopen", lambda *a, **k: _FakeResp(payload))
    msg = main.diagnostico_ollama()
    assert msg and "8k" in msg


def test_diagnostico_ollama_fora_do_ar(monkeypatch):
    def _boom(*a, **k):
        raise OSError("conexão recusada")
    monkeypatch.setattr(main.urllib.request, "urlopen", _boom)
    msg = main.diagnostico_ollama()
    assert msg and "Ollama" in msg


# --------------------------------------------------------------------------
# criar_llm (fix do tool-calling) — sem rede
# --------------------------------------------------------------------------
def test_criar_llm_forca_react():
    llm = main.criar_llm()
    assert llm.supports_function_calling() is False
