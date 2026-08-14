"""Testes do contrato HTTP da api.py (FastAPI TestClient).

A crew NÃO é executada: mockamos `_gerar_resultado` e `diagnostico_ollama`,
então testamos o formato das respostas (JSON/SSE), os códigos de erro e o
roteamento — tudo sem LLM, sem Ollama e sem rede."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api

RESULTADO_FAKE = {
    "conteudo": "```python\nprint(1)\n```",
    "conteudo_limpo": "print(1)",
    "arquivo": "output/x.py",
    "tipo": "codigo",
    "linguagem": "python",
}


@pytest.fixture
def client(monkeypatch):
    # Ambiente "saudável" por padrão e crew instantânea (sem LLM).
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: None)
    monkeypatch.setattr(api, "_gerar_resultado", lambda *a, **k: dict(RESULTADO_FAKE))
    return TestClient(api.app)


# --------------------------------------------------------------------------
# Funções puras
# --------------------------------------------------------------------------
def test_extrair_tema_ultimo_user():
    body = {"messages": [
        {"role": "system", "content": "voce e um assistente"},
        {"role": "user", "content": "primeira"},
        {"role": "user", "content": "Explique o README"},
    ]}
    assert api._extrair_tema(body) == "Explique o README"


def test_extrair_tema_content_em_partes():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "parte A"}, {"type": "image", "x": 1},
    ]}]}
    assert api._extrair_tema(body) == "parte A"


def test_extrair_tema_vazio():
    assert api._extrair_tema({"messages": []}) == ""


def test_frame_chunk_formato_sse():
    frame = api._frame_chunk("id1", "m", 123, {"content": "oi"})
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    obj = json.loads(frame[len("data: "):].strip())
    assert obj["object"] == "chat.completion.chunk"
    assert obj["choices"][0]["delta"]["content"] == "oi"


# --------------------------------------------------------------------------
# GET /v1/models
# --------------------------------------------------------------------------
def test_v1_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "list" and d["data"][0]["id"] == "cortexus-crew"


# --------------------------------------------------------------------------
# POST /v1/nexus-assist
# --------------------------------------------------------------------------
def test_nexus_assist_ok(client):
    r = client.post("/v1/nexus-assist", json={"tarefa": "explique X", "tipo": "pesquisa"})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "concluido" and d["relatorio"] == "print(1)"


def test_nexus_assist_tarefa_vazia(client):
    r = client.post("/v1/nexus-assist", json={"tarefa": "   "})
    assert r.status_code == 400


def test_nexus_assist_ollama_fora(client, monkeypatch):
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: "Ollama fora do ar")
    r = client.post("/v1/nexus-assist", json={"tarefa": "algo"})
    assert r.status_code == 503


# --------------------------------------------------------------------------
# POST /v1/chat/completions
# --------------------------------------------------------------------------
def test_chat_completions_nao_streaming(client):
    r = client.post("/v1/chat/completions", json={
        "model": "cortexus-crew", "stream": False,
        "messages": [{"role": "user", "content": "oi"}],
    })
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "chat.completion"
    assert d["choices"][0]["message"]["content"] == RESULTADO_FAKE["conteudo"]
    assert d["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_streaming(client):
    r = client.post("/v1/chat/completions", json={
        "model": "cortexus-crew", "stream": True,
        "messages": [{"role": "user", "content": "oi"}],
    })
    assert r.status_code == 200
    corpo = r.text
    assert "data: [DONE]" in corpo
    # deve conter um frame com o conteúdo
    assert "chat.completion.chunk" in corpo and "print(1)" in corpo


# --------------------------------------------------------------------------
# POST /agencia + GET /status/{id}  (job assíncrono; background roda mockado)
# --------------------------------------------------------------------------
def test_agencia_cria_job_e_conclui(client):
    r = client.post("/agencia", json={"tema": "crie um script"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"].startswith("Recebido")
    # O TestClient roda a background task; o status deve existir.
    s = client.get(f"/status/{job_id}")
    assert s.status_code == 200
    assert s.json()["status"] in ("na_fila", "executando", "concluido")


def test_status_desconhecido(client):
    assert client.get("/status/inexistente").status_code == 404


def test_agencia_ollama_fora(client, monkeypatch):
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: "Ollama fora")
    r = client.post("/agencia", json={"tema": "x"})
    assert r.status_code == 503


# --------------------------------------------------------------------------
# _disparar_webhook (urlopen mockado) — ordem dos excepts importa
# --------------------------------------------------------------------------
def test_disparar_webhook_sucesso(monkeypatch):
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert api._disparar_webhook("http://x", {"ok": 1}) is None


def test_disparar_webhook_conexao_falha(monkeypatch):
    def _boom(*a, **k):
        raise api.urllib.error.URLError("recusado")
    monkeypatch.setattr(api.urllib.request, "urlopen", _boom)
    msg = api._disparar_webhook("http://x", {"ok": 1})
    assert msg and "alcanc" in msg.lower()


# --------------------------------------------------------------------------
# ESTÚDIO DE PODCAST
#
# Usam o motor 'tom' (não exige voz baixada) e redirecionam a pasta de saída
# para o tmp_path, senão a suíte sujaria o output/ do projeto. O TestClient
# executa a BackgroundTask ANTES de devolver o controle, então quando o POST
# retorna o job já está concluído — não é preciso fazer polling aqui.
# --------------------------------------------------------------------------
ROTEIRO_FAKE = "Host1: Bom dia, isso é um teste.\nHost2: E esta é a resposta."


class _LLMFalso:
    """Substitui o LLM nas chamadas por bloco do modo 'blocos'.

    Sem isto os testes de podcast por tema chamariam o Ollama de verdade — a
    suíte tem que continuar hermética. Guarda os prompts recebidos para que o
    teste possa afirmar quantos blocos foram pedidos e em que ordem."""

    def __init__(self, falas_por_bloco: int = 2):
        self.prompts: list[str] = []
        self.falas_por_bloco = falas_por_bloco

    def call(self, mensagens, **kwargs):
        self.prompts.append(mensagens if isinstance(mensagens, str) else str(mensagens))
        n = len(self.prompts)
        # Texto diferente por bloco: se fosse igual, o _remover_repeticoes
        # cortaria as emendas e o teste mediria a coisa errada.
        return "\n".join(
            f"{'Rafa' if i % 2 else 'Bruno'}: Fala {i} do bloco numero {n}, "
            f"com tamanho suficiente para nao ser confundida com interjeicao."
            for i in range(1, self.falas_por_bloco + 1)
        )


@pytest.fixture
def studio(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api.estudio, "PODCAST_DIR", tmp_path / "podcast")
    falso = _LLMFalso()
    monkeypatch.setattr(api, "criar_llm", lambda: falso)
    client.llm_falso = falso          # o teste alcança os prompts por aqui
    return client


def test_podcast_com_roteiro_colado_gera_episodio(studio):
    r = studio.post("/podcast", json={"roteiro": ROTEIRO_FAKE, "motor": "tom"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    job = studio.get(f"/status/{job_id}").json()
    assert job["status"] == "concluido", job.get("detalhe")
    assert job["total_falas"] == 2
    assert job["duracao"] > 0
    assert job["origem_roteiro"] == "colado"
    assert Path(job["arquivo_wav"]).is_file()


def test_podcast_com_roteiro_dispensa_o_ollama(studio, monkeypatch):
    """O ponto central do estúdio: com um roteiro pronto, o áudio sai mesmo
    com o Ollama desligado e nenhum modelo de chat baixado."""
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: "Ollama fora do ar")
    r = studio.post("/podcast", json={"roteiro": ROTEIRO_FAKE, "motor": "tom"})
    assert r.status_code == 200
    assert studio.get(f"/status/{r.json()['job_id']}").json()["status"] == "concluido"


def test_podcast_por_tema_exige_ollama(studio, monkeypatch):
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: "Ollama fora do ar")
    r = studio.post("/podcast", json={"tema": "energia solar", "motor": "tom"})
    assert r.status_code == 503


def test_podcast_por_tema_passe_unico(studio, monkeypatch):
    """Caminho antigo: a crew escreve num pedido só. O que a crew devolve é
    mockado (é o LLM, não determinístico); o resto é real."""
    monkeypatch.setattr(
        api,
        "_gerar_resultado",
        lambda *a, **k: {**RESULTADO_FAKE, "conteudo_limpo": ROTEIRO_FAKE},
    )
    r = studio.post(
        "/podcast", json={"tema": "energia solar", "motor": "tom", "modo": "unico"}
    )
    assert r.status_code == 200

    job = studio.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "concluido", job.get("detalhe")
    assert job["origem_roteiro"] == "crew"
    assert job["total_falas"] == 2
    assert job["blocos"] == 1
    # No passe único a crew é chamada, e nenhuma chamada direta ao LLM acontece.
    assert studio.llm_falso.prompts == []


def test_podcast_avisa_quando_a_crew_nao_devolve_dialogo(studio):
    """O 7B às vezes ignora o formato e devolve prosa/código (o RESULTADO_FAKE
    do fixture é justamente código). O job precisa falhar EXPLICANDO o formato,
    não com um traceback."""
    r = studio.post(
        "/podcast", json={"tema": "energia solar", "motor": "tom", "modo": "unico"}
    )
    assert r.status_code == 200
    job = studio.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "erro" and "Locutor: texto" in job["detalhe"]


# --- modo "blocos" (padrão) ------------------------------------------------
PAUTA_FAKE = (
    "1. Abertura | o que e o tema, quem estuda\n"
    "2. Exemplos | dilema do prisioneiro, leilao\n"
    "3. Fecho | onde isso aparece no dia a dia\n"
)


@pytest.fixture
def studio_pauta(studio, monkeypatch):
    """Crew devolvendo uma PAUTA bem formada (o modo blocos espera isso)."""
    monkeypatch.setattr(
        api,
        "_gerar_resultado",
        lambda *a, **k: {**RESULTADO_FAKE, "conteudo_limpo": PAUTA_FAKE},
    )
    return studio


def test_roteiro_em_blocos_faz_uma_chamada_por_bloco(studio_pauta):
    r = studio_pauta.post(
        "/podcast/roteiro", json={"tema": "teoria dos jogos", "minutos": 5}
    )
    assert r.status_code == 200
    job = studio_pauta.get(f"/status/{r.json()['job_id']}").json()

    assert job["status"] == "concluido", job.get("detalhe")
    # 5 minutos -> 3 blocos -> 3 chamadas curtas ao LLM (a pauta veio da crew).
    assert job["blocos"] == 3
    assert len(studio_pauta.llm_falso.prompts) == 3
    # 3 blocos x 2 falas do LLM falso: as emendas não foram deduplicadas à toa.
    assert job["total_falas"] == 6
    assert job["pauta"] == PAUTA_FAKE


def test_roteiro_em_blocos_passa_o_assunto_certo_a_cada_bloco(studio_pauta):
    studio_pauta.post(
        "/podcast/roteiro", json={"tema": "teoria dos jogos", "minutos": 5}
    )
    p1, p2, p3 = studio_pauta.llm_falso.prompts
    assert "Abertura" in p1 and "bloco 1 de 3" in p1
    assert "Exemplos" in p2 and "bloco 2 de 3" in p2
    assert "Fecho" in p3 and "bloco 3 de 3" in p3
    # Continuidade: o bloco do meio recebe as falas anteriores e a proibição
    # de cumprimentar; o primeiro, não.
    assert "NÃO repita estas falas" in p2 and "proibido cumprimentar" in p2
    assert "NÃO repita estas falas" not in p1


def test_roteiro_em_blocos_sobrevive_a_um_bloco_que_falha(studio_pauta, monkeypatch):
    """Um bloco que estoura não pode derrubar o episódio inteiro: o que já foi
    escrito continua valendo e o usuário completa no editor."""

    class LLMQuePifaNoSegundo:
        def __init__(self):
            self.n = 0

        def call(self, mensagens, **kwargs):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("contexto estourou")
            return f"Rafa: Esta e a fala do bloco numero {self.n}, com tamanho normal."

    monkeypatch.setattr(api, "criar_llm", LLMQuePifaNoSegundo)
    r = studio_pauta.post(
        "/podcast/roteiro", json={"tema": "teoria dos jogos", "minutos": 5}
    )
    job = studio_pauta.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "concluido", job.get("detalhe")
    assert job["total_falas"] == 2          # blocos 1 e 3 sobreviveram


def test_roteiro_em_blocos_reporta_a_etapa_atual(studio_pauta):
    """O estúdio mostra 'bloco 3 de 5'; sem isso a barra parece travada."""
    r = studio_pauta.post("/podcast/roteiro", json={"tema": "x", "minutos": 5})
    job = studio_pauta.get(f"/status/{r.json()['job_id']}").json()
    # Ao terminar, a última etapa registrada é o último bloco.
    assert job["etapa"] == "bloco 3 de 3"


def test_podcast_sem_tema_e_sem_roteiro(studio):
    r = studio.post("/podcast", json={})
    assert r.status_code == 400 and "roteiro" in r.json()["detail"]


def test_podcast_motor_desconhecido(studio):
    r = studio.post("/podcast", json={"roteiro": ROTEIRO_FAKE, "motor": "elevenlabs"})
    assert r.status_code == 400 and "piper" in r.json()["detail"]


def test_podcast_voz_desconhecida(studio):
    r = studio.post(
        "/podcast",
        json={"roteiro": ROTEIRO_FAKE, "motor": "tom", "vozes": ["pt_BR-inexistente"]},
    )
    assert r.status_code == 400 and "desconhecida" in r.json()["detail"].lower()


def test_podcast_minutos_fora_da_faixa(studio):
    r = studio.post("/podcast", json={"tema": "x", "minutos": 999})
    assert r.status_code == 400


@pytest.mark.parametrize(
    "trilha,esperado",
    [
        ("../fora_do_projeto.wav", 400),   # escapa da pasta do projeto
        ("README.md", 400),                # extensão errada
        ("web/nao_existe.wav", 404),       # dentro do projeto, mas ausente
    ],
)
def test_podcast_recusa_trilha_invalida(studio, trilha, esperado):
    r = studio.post(
        "/podcast", json={"roteiro": ROTEIRO_FAKE, "motor": "tom", "trilha": trilha}
    )
    assert r.status_code == esperado


def test_podcast_audio_entrega_o_arquivo(studio):
    r = studio.post("/podcast", json={"roteiro": ROTEIRO_FAKE, "motor": "tom"})
    job_id = r.json()["job_id"]
    audio = studio.get(f"/podcast/{job_id}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] in ("audio/mpeg", "audio/wav")
    assert len(audio.content) > 44  # 44 = tamanho de um cabeçalho WAV vazio


def test_podcast_audio_antes_de_concluir_da_409(studio):
    job_id = api._registrar_job("x", tipo="podcast")  # fica em "na_fila"
    assert studio.get(f"/podcast/{job_id}/audio").status_code == 409


def test_podcast_audio_job_desconhecido(studio):
    assert studio.get("/podcast/naoexiste/audio").status_code == 404


def test_podcast_roteiro_sem_tema(studio):
    assert studio.post("/podcast/roteiro", json={}).status_code == 400


def test_vozes_lista_catalogo_e_disponibilidade(studio):
    d = studio.get("/podcast/vozes").json()
    ids = {v["id"] for v in d["vozes"]}
    assert "pt_BR-faber-medium" in ids
    assert all("baixada" in v and "nasal" in v for v in d["vozes"])
    assert len(d["padrao"]) == 2


def test_ambiente_reporta_ollama_e_ffmpeg(studio):
    d = studio.get("/podcast/ambiente").json()
    assert d["ollama_ok"] is True          # o fixture client mocka o diagnóstico
    assert isinstance(d["ffmpeg"], bool)


def test_estudio_serve_o_html(studio):
    r = studio.get("/estudio")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Estúdio de Podcast" in r.text
