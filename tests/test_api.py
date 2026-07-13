"""Testes do contrato HTTP da api.py (FastAPI TestClient).

A crew NÃO é executada: mockamos `_gerar_resultado` e `diagnostico_ollama`,
então testamos o formato das respostas (JSON/SSE), os códigos de erro e o
roteamento — tudo sem LLM, sem Ollama e sem rede."""

from __future__ import annotations

import json

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
