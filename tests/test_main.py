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
# --------------------------------------------------------------------------
# ler_pagina_web — sandbox de URL e limpeza (sem rede: o raspador é mockado)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url,trecho",
    [
        ("file:///C:/Windows/win.ini", "Esquema 'file' bloqueado"),
        ("ftp://exemplo.com/x", "Esquema 'ftp' bloqueado"),
        ("nao-e-url", "Esquema 'ausente' bloqueado"),
        ("https://", "sem domínio"),
    ],
)
def test_ler_pagina_web_recusa_esquema_nao_http(url, trecho):
    """Sem este bloqueio, 'file:///...' usaria esta ferramenta para contornar o
    confinamento a BASE_DIR das ferramentas de disco."""
    saida = main.ler_pagina_web.run(url=url)
    assert saida.startswith("BLOQUEADO") and trecho in saida


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/api/tags",   # o próprio Ollama
        "http://127.0.0.1:8000/status/x",    # a própria API do Cortexus
        "http://192.168.0.1/admin",          # roteador da rede doméstica
        "http://10.0.0.5/",
        "http://169.254.169.254/latest/meta-data/",  # metadados de nuvem
    ],
)
def test_ler_pagina_web_recusa_endereco_interno(url):
    """SSRF: o agente processa texto NÃO confiável da web, então uma injeção
    de prompt poderia mandá-lo buscar um endereço interno."""
    saida = main.ler_pagina_web.run(url=url)
    assert saida.startswith("BLOQUEADO") and "interno" in saida


def test_ler_pagina_web_dominio_inexistente_nao_estoura(monkeypatch):
    def _falha(*a, **k):
        raise OSError("getaddrinfo failed")
    monkeypatch.setattr(main.socket, "getaddrinfo", _falha)
    saida = main.ler_pagina_web.run(url="https://dominio-que-nao-existe.invalid/x")
    assert "resolver o domínio" in saida


# --------------------------------------------------------------------------
# busca_web — agora LÊ as fontes sozinha (rede mockada)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url,esperado",
    [
        ("https://pt.wikipedia.org/wiki/X", True),
        ("https://docs.python.org/3/", True),
        ("https://blog.aleatorio.com/post", True),
        ("https://www.youtube.com/watch?v=abc", False),   # vídeo: sem prosa
        ("https://www.instagram.com/p/abc", False),
        ("https://exemplo.com/manual.pdf", False),
        ("ftp://exemplo.com/x", False),
    ],
)
def test_vale_ler_descarta_video_e_binario(url, esperado):
    assert main._vale_ler(url) is esperado


@pytest.mark.parametrize(
    "url",
    [
        "https://prezi.com/p/abc/teoria",
        "https://pt.slideshare.net/x",
        "https://www.scribd.com/doc/1",
        "https://brainly.com.br/tarefa/1",
        "https://pt.quora.com/x",
    ],
)
def test_vale_ler_descarta_agregador_de_slides_e_respostas(url):
    """Têm prosa, mas prosa sem revisão. Um deles (`prezi.com`) chegou a virar
    fonte citada numa pauta de teste."""
    assert main._vale_ler(url) is False


@pytest.mark.parametrize(
    "url,referencia",
    [
        ("https://pt.wikipedia.org/wiki/X", True),
        ("https://plato.stanford.edu/entries/game-theory/", True),
        ("https://www.gov.br/pagina", True),
        ("https://docs.python.org/3/", True),
        ("https://www.fm2s.com.br/blog/teoria-dos-jogos", False),
        ("https://scopi.com.br/blog/x", False),
    ],
)
def test_eh_referencia_separa_enciclopedia_de_blog(url, referencia):
    assert main._eh_referencia(url) is referencia


def test_busca_web_rotula_o_nivel_de_cada_fonte(monkeypatch, raspador):
    """O único erro factual que sobreviveu num teste real veio de um blog de
    consultoria lido junto da Wikipédia — o modelo não tinha como saber qual
    das duas pesava mais."""
    resultados = [
        {"title": "Blog", "href": "https://www.fm2s.com.br/blog/x", "body": ".."},
        {"title": "Wiki", "href": "https://pt.wikipedia.org/wiki/X", "body": ".."},
    ]
    monkeypatch.setattr(
        main, "DDGS", lambda: type("D", (), {"text": lambda s, *a, **k: resultados})()
    )
    raspador("Uma frase de conteudo com tamanho suficiente para ser mantida.\n")

    saida = main.busca_web.run(query="x")
    assert "[FONTE DE REFERÊNCIA] TEXTO DE https://pt.wikipedia.org" in saida
    assert "[FONTE SECUNDÁRIA] TEXTO DE https://www.fm2s.com.br" in saida
    assert "DATAS, NOMES PRÓPRIOS e DEFINIÇÕES" in saida
    # A de referência é lida primeiro (ordenação por credibilidade).
    assert saida.index("pt.wikipedia.org/wiki/X:") < saida.index("fm2s.com.br/blog/x:")


def test_busca_web_avisa_quando_so_ha_fonte_secundaria(monkeypatch, raspador):
    resultados = [{"title": "Blog", "href": "https://blog.qualquer.com/x", "body": ".."}]
    monkeypatch.setattr(
        main, "DDGS", lambda: type("D", (), {"text": lambda s, *a, **k: resultados})()
    )
    raspador("Uma frase de conteudo com tamanho suficiente para ser mantida.\n")
    saida = main.busca_web.run(query="x")
    assert "Nenhuma fonte de referência foi encontrada" in saida
    assert "incertos" in saida


def test_credibilidade_ordena_enciclopedia_antes_de_blog():
    urls = [
        "https://blog.qualquer.com/a",
        "https://pt.wikipedia.org/wiki/A",
        "https://x.edu/a",
    ]
    ordenado = sorted(urls, key=main._nota_de_credibilidade)
    assert "wikipedia" in ordenado[0]
    assert "blog" in ordenado[-1]


def test_busca_web_anexa_o_texto_das_fontes(monkeypatch, raspador):
    """O ponto central: MEDIMOS que o 7B não chama a ferramenta de leitura
    (uma execução de ferramenta no log, só a busca). Fundindo busca e leitura,
    a decisão sai das mãos do modelo — ele recebe prosa queira ou não."""
    resultados = [
        {"title": "Vídeo", "href": "https://www.youtube.com/watch?v=1", "body": ".."},
        {"title": "Wiki", "href": "https://pt.wikipedia.org/wiki/Nash", "body": ".."},
    ]
    monkeypatch.setattr(main, "DDGS", lambda: type("D", (), {"text": lambda s, *a, **k: resultados})())
    raspador(
        "O equilíbrio de Nash é a situação em que nenhum jogador ganha "
        "mudando de estratégia sozinho.\n"
    )

    saida = main.busca_web.run(query="equilibrio de nash")
    assert "TEXTO DAS FONTES" in saida
    assert "nenhum jogador ganha" in saida
    # A lista de resultados continua ali; o texto é um acréscimo.
    assert "https://pt.wikipedia.org/wiki/Nash" in saida
    # O YouTube foi descartado antes de qualquer requisição.
    assert "TEXTO DE https://www.youtube.com" not in saida


def test_busca_web_avisa_quando_nenhuma_fonte_tem_texto(monkeypatch, raspador):
    resultados = [{"title": "V", "href": "https://youtu.be/x", "body": ".."}]
    monkeypatch.setattr(main, "DDGS", lambda: type("D", (), {"text": lambda s, *a, **k: resultados})())
    raspador("")
    saida = main.busca_web.run(query="x")
    assert "NENHUMA das páginas devolveu texto legível" in saida
    assert "NÃO invente" in saida


def test_busca_web_respeita_o_teto_de_fontes(monkeypatch, raspador):
    resultados = [
        {"title": f"F{i}", "href": f"https://site{i}.org/a", "body": ".."}
        for i in range(6)
    ]
    monkeypatch.setattr(main, "DDGS", lambda: type("D", (), {"text": lambda s, *a, **k: resultados})())
    raspador("Uma frase de conteúdo com tamanho suficiente para valer. " * 3)
    saida = main.busca_web.run(query="x")
    assert saida.count("TEXTO DE https://") <= main.MAX_FONTES_LIDAS


def test_busca_web_fonte_que_explode_nao_derruba_a_busca(monkeypatch, raspador):
    resultados = [{"title": "A", "href": "https://a.org/x", "body": ".."}]
    monkeypatch.setattr(main, "DDGS", lambda: type("D", (), {"text": lambda s, *a, **k: resultados})())

    def _explode(**k):
        raise RuntimeError("conexão caiu")

    raspador(_explode)
    saida = main.busca_web.run(query="x")
    assert "https://a.org/x" in saida            # a lista sobrevive
    assert "NENHUMA das páginas" in saida        # e o aviso aparece


def test_busca_web_falha_de_rede_vira_texto(monkeypatch):
    def _boom():
        raise RuntimeError("sem rede")
    monkeypatch.setattr(main, "DDGS", _boom)
    assert "A busca falhou" in main.busca_web.run(query="x")


class _RaspadorFalso:
    """Substitui o `main._raspador` inteiro nos testes.

    Trocar só o método `run` não funciona: a ScrapeWebsiteTool é um modelo
    pydantic e recusa atributo que não esteja no schema ("object has no field
    'run'"). Substituir o objeto do módulo é mais simples e não depende do
    formato interno da ferramenta.
    """

    def __init__(self, resposta):
        self._resposta = resposta

    def run(self, **kwargs):
        return self._resposta(**kwargs)


@pytest.fixture
def raspador(monkeypatch):
    """Devolve um instalador de resposta falsa, com a URL já liberada."""
    monkeypatch.setattr(main, "_url_segura", lambda url: None)

    def instalar(resposta):
        alvo = resposta if callable(resposta) else (lambda **k: resposta)
        monkeypatch.setattr(main, "_raspador", _RaspadorFalso(alvo))

    return instalar


def test_ler_pagina_web_devolve_texto_limpo(raspador):
    raspador(
        "The following text is scraped website content:\n"
        "Equilíbrio de Nash – Wikipédia\n"                      # título, sem ponto
        "Página principal Conteúdo destacado Eventos atuais\n"   # menu longo
        "العربية Azərbaycanca Български Català Čeština\n"        # lista de idiomas
        "O equilíbrio de Nash é uma situação em que nenhum jogador ganha "
        "mudando de estratégia unilateralmente.[1]\n"
        "Doar\n"
    )
    saida = main.ler_pagina_web.run(url="https://pt.wikipedia.org/wiki/X")
    assert "equilíbrio de Nash é uma situação" in saida
    # O critério é pontuação de fim de frase, não tamanho: o menu longo cai.
    assert "Página principal" not in saida
    assert "Azərbaycanca" not in saida
    assert "Doar" not in saida
    assert "[1]" not in saida          # marcador de referência não se pronuncia
    assert "scraped website content" not in saida


def test_ler_pagina_web_avisa_quando_nao_ha_prosa(raspador):
    """Foi o caso que gerou o roteiro alucinado: a busca devolveu vídeos do
    YouTube, cuja página não tem uma única frase para ler."""
    raspador("Titulo do video - YouTube About Press Copyright Contact us")
    saida = main.ler_pagina_web.run(url="https://www.youtube.com/watch?v=abc")
    assert "não devolveu texto legível" in saida


def test_ler_pagina_web_trunca_no_limite(raspador):
    raspador("Esta é uma frase de conteúdo com tamanho suficiente. " * 400)
    saida = main.ler_pagina_web.run(url="https://exemplo.com")
    assert "texto truncado" in saida
    assert len(saida) < main.LIMITE_PAGINA_WEB + 300  # + cabeçalho e aviso


def test_ler_pagina_web_erro_do_raspador_vira_texto(raspador):
    """A ferramenta nunca levanta: o laço ReAct precisa do erro como texto
    para tentar a próxima fonte."""

    def _explode(**k):
        raise RuntimeError("timeout")

    raspador(_explode)
    saida = main.ler_pagina_web.run(url="https://exemplo.com")
    assert "Não consegui ler a página" in saida and "timeout" in saida


def test_pesquisador_recebe_a_ferramenta_de_leitura(monkeypatch):
    """Regressão: a ferramenta existir sem estar ligada ao agente seria pior
    que não existir — o sintoma (alucinação) continuaria idêntico.

    A DirectorySearchTool é substituída porque o construtor dela cria um
    cliente/coleção ChromaDB em %LOCALAPPDATA% — efeito colateral que não cabe
    numa suíte hermética."""
    # Reaproveita uma ferramenta já instanciada: o Agent do CrewAI valida que
    # `tools` contém BaseTool, então o dublê não pode ser um objeto qualquer.
    monkeypatch.setattr(main, "DirectorySearchTool", lambda *a, **k: main.leitor_pastas)

    equipe = main.montar_equipe("tema qualquer", llm=main.criar_llm())
    pesquisador = next(a for a in equipe.agents if "Pesquisador" in a.role)
    nomes = {getattr(t, "name", "") for t in pesquisador.tools}
    assert "Ler Página Web" in nomes
    assert "Busca na Web" in nomes


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
