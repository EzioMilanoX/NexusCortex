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
    # Sem FFmpeg de verdade: com ele no PATH a suíte dispararia um subprocesso
    # por episódio (lento e dependente da máquina). O episódio sai em WAV.
    monkeypatch.setattr(api.estudio, "exportar_mp3", lambda *a, **k: None)
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


# --- mais cobertura das rotas do estúdio (tudo offline) -------------------
class _MotorFalso(api.estudio.MotorTom):
    """Finge ser o Piper: aceita qualquer voz e sintetiza um tom.

    Exercita o caminho do motor de PRODUÇÃO (validação no POST + síntese no
    background) sem piper-tts instalado e sem os .onnx em disco."""

    nome = "piper"

    def __init__(self):
        super().__init__()
        self.sintetizadas: list[str] = []

    def verificar(self, apresentadores):
        return None

    def sintetizar(self, texto, apresentador, destino):
        self.sintetizadas.append(texto)
        return super().sintetizar(texto, apresentador, destino)


def test_podcast_motor_piper_stubado_gera_episodio(studio, monkeypatch):
    motor = _MotorFalso()
    monkeypatch.setattr(api.estudio, "criar_motor", lambda *a, **k: motor)
    r = studio.post("/podcast", json={"roteiro": ROTEIRO_FAKE, "motor": "piper"})
    assert r.status_code == 200

    job = studio.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "concluido", job.get("detalhe")
    assert job["motor"] == "piper"
    assert len(motor.sintetizadas) == 2
    assert job["arquivo_mp3"] is None and "FFmpeg" in job["aviso_mp3"]


def test_podcast_por_tema_em_blocos_gera_audio(studio_pauta):
    """O caminho completo do POST /podcast por tema (modo padrão): pauta da
    crew (mockada) -> blocos do LLM (falso) -> síntese 'tom'."""
    r = studio_pauta.post("/podcast", json={"tema": "teoria dos jogos", "motor": "tom"})
    job = studio_pauta.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "concluido", job.get("detalhe")
    assert job["origem_roteiro"] == "crew" and job["blocos"] == 3
    assert job["total_falas"] == 6
    assert job["audio_em"] == f"/podcast/{job['job_id']}/audio"


def test_podcast_dispara_webhook_ao_concluir(studio, monkeypatch):
    recebidos: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        api, "_disparar_webhook", lambda url, payload: recebidos.append((url, payload))
    )
    r = studio.post(
        "/podcast",
        json={"roteiro": ROTEIRO_FAKE, "motor": "tom", "webhook_url": "http://n8n/x"},
    )
    job_id = r.json()["job_id"]
    assert recebidos and recebidos[0][0] == "http://n8n/x"
    assert recebidos[0][1]["status"] == "concluido"
    assert recebidos[0][1]["job_id"] == job_id
    assert studio.get(f"/status/{job_id}").json()["webhook_entregue"] is True


def test_podcast_recusa_webhook_sem_http(studio):
    r = studio.post(
        "/podcast",
        json={"roteiro": ROTEIRO_FAKE, "motor": "tom", "webhook_url": "file:///etc"},
    )
    assert r.status_code == 400


def test_podcast_roteiro_por_tema_devolve_texto_normalizado(studio, monkeypatch):
    bruto = "**Host1:** Olá, [risos] tudo bem?\nHost2: Tudo ótimo, e você?"
    monkeypatch.setattr(
        api,
        "_gerar_resultado",
        lambda *a, **k: {**RESULTADO_FAKE, "conteudo_limpo": bruto},
    )
    r = studio.post("/podcast/roteiro", json={"tema": "x", "modo": "unico"})
    assert r.status_code == 200 and "job_id" in r.json()
    job = studio.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "concluido", job.get("detalhe")
    assert job["roteiro_bruto"] == bruto
    # O que volta para o editor já passou pelo parser: sem markdown nem rubrica.
    assert "**" not in job["roteiro"] and "[risos]" not in job["roteiro"]
    assert job["total_falas"] == 2


def test_podcast_roteiro_exige_ollama(studio, monkeypatch):
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: "Ollama fora do ar")
    assert studio.post("/podcast/roteiro", json={"tema": "x"}).status_code == 503


def test_podcast_roteiro_minutos_fora_da_faixa(studio):
    r = studio.post("/podcast/roteiro", json={"tema": "x", "minutos": 0})
    assert r.status_code == 400


def test_podcast_audio_prefere_o_mp3(studio, monkeypatch):
    def _mp3_falso(wav, destino=None, **k):
        mp3 = wav.with_suffix(".mp3")
        mp3.write_bytes(b"ID3" + b"\x00" * 64)
        return mp3

    monkeypatch.setattr(api.estudio, "exportar_mp3", _mp3_falso)
    r = studio.post("/podcast", json={"roteiro": ROTEIRO_FAKE, "motor": "tom"})
    audio = studio.get(f"/podcast/{r.json()['job_id']}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/mpeg"
    assert audio.content.startswith(b"ID3")


def test_podcast_audio_arquivo_apagado_da_404(studio, tmp_path):
    job_id = api._registrar_job("x", tipo="podcast")
    api._atualizar_job(
        job_id, status="concluido", arquivo_wav=str(tmp_path / "sumiu.wav")
    )
    assert studio.get(f"/podcast/{job_id}/audio").status_code == 404


def test_vozes_reflete_o_que_esta_em_disco(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(api.estudio, "VOZES_DIR", tmp_path)
    (tmp_path / "pt_BR-faber-medium.onnx").write_bytes(b"")
    d = studio.get("/podcast/vozes").json()
    baixadas = {v["id"] for v in d["vozes"] if v["baixada"]}
    assert baixadas == {"pt_BR-faber-medium"}
    assert d["pasta"] == str(tmp_path)
    assert "tom" in d["motores"] and "piper" in d["motores"]


def test_ambiente_ollama_fora_e_sem_ffmpeg(studio, monkeypatch):
    monkeypatch.setattr(api, "diagnostico_ollama", lambda: "Ollama fora do ar")
    monkeypatch.setattr(api.shutil, "which", lambda nome: None)
    d = studio.get("/podcast/ambiente").json()
    assert d == {
        "ollama_ok": False,
        "ollama_detalhe": "Ollama fora do ar",
        "ffmpeg": False,
    }


def test_estudio_sem_a_pagina_da_404(studio, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "WEB_DIR", tmp_path)
    assert studio.get("/estudio").status_code == 404


# --------------------------------------------------------------------------
# _caminho_de_projeto — guarda contra path traversal
#
# A raiz do "projeto" é trocada por uma pasta temporária: assim os arquivos de
# teste não sujam o repositório e dá para criar alvos FORA da raiz de verdade.
# --------------------------------------------------------------------------
@pytest.fixture
def projeto(tmp_path, monkeypatch):
    raiz = (tmp_path / "proj").resolve()
    (raiz / "trilhas").mkdir(parents=True)
    (raiz / "trilhas" / "fundo.wav").write_bytes(b"RIFF")
    (raiz / "trilhas" / "ALTO.WAV").write_bytes(b"RIFF")
    (tmp_path / "segredo.wav").write_bytes(b"RIFF")
    # Irmã com o MESMO prefixo: pega quem comparasse caminhos como string.
    (tmp_path / "proj_vizinho").mkdir()
    (tmp_path / "proj_vizinho" / "x.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(api.estudio, "BASE_DIR", raiz)
    return raiz


def test_caminho_de_projeto_aceita_relativo_dentro(projeto):
    alvo = api._caminho_de_projeto("trilhas/fundo.wav", (".wav",))
    assert alvo == projeto / "trilhas" / "fundo.wav"


def test_caminho_de_projeto_aceita_absoluto_dentro(projeto):
    absoluto = str(projeto / "trilhas" / "fundo.wav")
    assert api._caminho_de_projeto(absoluto, (".wav",)).is_file()


def test_caminho_de_projeto_extensao_ignora_caixa(projeto):
    assert api._caminho_de_projeto("trilhas/ALTO.WAV", (".wav",)).is_file()


@pytest.mark.parametrize(
    "bruto",
    [
        "../segredo.wav",                 # sobe um nível
        "trilhas/../../segredo.wav",      # desce e sobe além da raiz
        "../proj_vizinho/x.wav",          # prefixo igual, pasta irmã
        ".",                              # a própria raiz não é arquivo
    ],
)
def test_caminho_de_projeto_recusa_quem_escapa(projeto, bruto):
    with pytest.raises(api.HTTPException) as erro:
        api._caminho_de_projeto(bruto, (".wav",))
    assert erro.value.status_code == 400


@pytest.mark.skipif(api.os.name != "nt", reason="barra invertida só separa no Windows")
def test_caminho_de_projeto_recusa_barra_invertida_do_windows(projeto):
    with pytest.raises(api.HTTPException) as erro:
        api._caminho_de_projeto("..\\segredo.wav", (".wav",))
    assert erro.value.status_code == 400


def test_caminho_de_projeto_recusa_absoluto_fora(projeto):
    fora = str(projeto.parent / "segredo.wav")
    with pytest.raises(api.HTTPException) as erro:
        api._caminho_de_projeto(fora, (".wav",))
    assert erro.value.status_code == 400 and "fora" in erro.value.detail.lower()


def test_caminho_de_projeto_recusa_symlink_para_fora(projeto):
    link = projeto / "trilhas" / "atalho.wav"
    try:
        link.symlink_to(projeto.parent / "segredo.wav")
    except (OSError, NotImplementedError):
        pytest.skip("sem permissão para criar symlink nesta máquina")
    with pytest.raises(api.HTTPException) as erro:
        api._caminho_de_projeto("trilhas/atalho.wav", (".wav",))
    assert erro.value.status_code == 400


def test_caminho_de_projeto_extensao_errada_e_ausente(projeto):
    (projeto / "notas.txt").write_text("x", encoding="utf-8")
    with pytest.raises(api.HTTPException) as errada:
        api._caminho_de_projeto("notas.txt", (".wav",))
    assert errada.value.status_code == 400
    with pytest.raises(api.HTTPException) as ausente:
        api._caminho_de_projeto("trilhas/nao_existe.wav", (".wav",))
    assert ausente.value.status_code == 404


# --------------------------------------------------------------------------
# _escrever_em_blocos — chamado direto, com crew e LLM falsos
# --------------------------------------------------------------------------
@pytest.fixture
def blocos(monkeypatch):
    """Crew devolvendo a PAUTA_FAKE e um LLM falso; guarda o que a crew recebeu."""
    chamadas_crew: list[dict] = []

    def _crew(prompt, **kwargs):
        chamadas_crew.append({"prompt": prompt, **kwargs})
        return {
            **RESULTADO_FAKE,
            "conteudo_limpo": PAUTA_FAKE,
            "arquivo": "output/pauta.txt",
        }

    falso = _LLMFalso()
    monkeypatch.setattr(api, "_gerar_resultado", _crew)
    monkeypatch.setattr(api, "criar_llm", lambda: falso)
    return chamadas_crew, falso


def _rodar_blocos(**campos):
    pedido = api.PedidoPodcast(**{"tema": "teoria dos jogos", "minutos": 5, **campos})
    job_id = api._registrar_job(pedido.tema, tipo="roteiro")
    saida = api._escrever_em_blocos(job_id, pedido, api.estudio.APRESENTADORES_PADRAO)
    return job_id, saida


def test_escrever_em_blocos_devolve_o_contrato(blocos):
    _, falso = blocos
    job_id, saida = _rodar_blocos()
    assert set(saida) == {"falas", "bruto", "pauta", "blocos", "arquivo_roteiro"}
    assert saida["blocos"] == 3 and len(falso.prompts) == 3
    assert saida["pauta"] == PAUTA_FAKE
    assert saida["arquivo_roteiro"] == "output/pauta.txt"
    assert len(saida["falas"]) == 6
    assert all(isinstance(f, api.estudio.Fala) for f in saida["falas"])
    # O bruto junta os blocos com linha em branco, na ordem.
    assert saida["bruto"].count("\n\n") == 2
    assert saida["bruto"].index("bloco numero 1") < saida["bruto"].index("bloco numero 3")
    job = api._jobs[job_id]
    assert job["status"] == "escrevendo_roteiro" and job["etapa"] == "bloco 3 de 3"


def test_escrever_em_blocos_repassa_contexto_e_memoria_a_crew(blocos):
    chamadas_crew, _ = blocos
    _rodar_blocos(contexto_extra="anexo do usuário", usar_memoria=True)
    assert len(chamadas_crew) == 1                 # a pesquisa roda UMA vez
    assert chamadas_crew[0]["contexto_extra"] == "anexo do usuário"
    assert chamadas_crew[0]["usar_memoria"] is True


def test_escrever_em_blocos_marca_o_bloco_que_falhou(blocos, monkeypatch):
    class _LLMQueFalhaNoPrimeiro(_LLMFalso):
        def call(self, mensagens, **kwargs):
            if not self.prompts:
                self.prompts.append("falhou")
                raise RuntimeError("timeout do Ollama")
            return super().call(mensagens, **kwargs)

    monkeypatch.setattr(api, "criar_llm", _LLMQueFalhaNoPrimeiro)
    _, saida = _rodar_blocos()
    assert saida["bruto"].startswith("[bloco 1 falhou: timeout do Ollama]")
    assert len(saida["falas"]) == 4                 # blocos 2 e 3 sobreviveram


def test_escrever_em_blocos_pauta_curta_limita_as_chamadas(blocos, monkeypatch):
    _, falso = blocos
    pauta_curta = "1. Abertura | um fato\n2. Fecho | outro fato\n"
    monkeypatch.setattr(
        api,
        "_gerar_resultado",
        lambda *a, **k: {**RESULTADO_FAKE, "conteudo_limpo": pauta_curta},
    )
    _, saida = _rodar_blocos(minutos=10)
    assert saida["blocos"] == 2 and len(falso.prompts) == 2


# --------------------------------------------------------------------------
# Configuração do servidor (host / porta / reload)
# --------------------------------------------------------------------------
def test_servidor_padrao_e_loopback_sem_reload(monkeypatch):
    for var in ("NCX_HOST", "NCX_PORT", "NCX_RELOAD"):
        monkeypatch.delenv(var, raising=False)
    config = api.configuracao_servidor([])        # lê o os.environ de verdade
    assert config.host == "127.0.0.1"
    assert config.port == 8000
    assert config.reload is False
    assert api.aviso_de_exposicao(config.host, config.port) is None


def test_servidor_le_variaveis_de_ambiente():
    config = api.configuracao_servidor(
        [], {"NCX_HOST": "0.0.0.0", "NCX_PORT": "9001", "NCX_RELOAD": "sim"}
    )
    assert (config.host, config.port, config.reload) == ("0.0.0.0", 9001, True)


def test_servidor_linha_de_comando_vence_o_ambiente():
    config = api.configuracao_servidor(
        ["--host", "127.0.0.1", "--port", "8100", "--no-reload"],
        {"NCX_HOST": "0.0.0.0", "NCX_PORT": "9001", "NCX_RELOAD": "1"},
    )
    assert (config.host, config.port, config.reload) == ("127.0.0.1", 8100, False)


def test_servidor_variavel_vazia_cai_no_padrao():
    config = api.configuracao_servidor(
        [], {"NCX_HOST": "", "NCX_PORT": "", "NCX_RELOAD": "0"}
    )
    assert (config.host, config.port, config.reload) == ("127.0.0.1", 8000, False)


def test_servidor_porta_invalida_da_erro_amigavel():
    with pytest.raises(SystemExit):
        api.configuracao_servidor([], {"NCX_PORT": "oito mil"})


@pytest.mark.parametrize(
    "host,loopback",
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.0.10", False),
        ("minha-maquina.local", False),   # nome qualquer: na dúvida, exposto
    ],
)
def test_host_e_loopback(host, loopback):
    assert api.host_e_loopback(host) is loopback


def test_aviso_quando_sai_do_loopback():
    aviso = api.aviso_de_exposicao("0.0.0.0", 8000)
    assert aviso and "0.0.0.0:8000" in aviso and "/agencia" in aviso
