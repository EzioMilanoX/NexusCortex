"""Testes do cliente CLI nexus_bridge.py (com requests mockado, sem rede)."""

from __future__ import annotations

import requests

import nexus_bridge


class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
    def json(self):
        return self._payload


def test_detalhe_json():
    r = _FakeResp(500, {"detail": "erro X"})
    assert nexus_bridge._detalhe(r) == "erro X"


def test_detalhe_texto_cru():
    class R:
        text = "erro cru sem json"
        def json(self):
            raise ValueError("no json")
    assert "erro cru" in nexus_bridge._detalhe(R())


def test_bridge_pesquisar_sucesso(monkeypatch, capsys):
    def fake_post(url, json, timeout):
        # confirma que envia tarefa/tipo corretos
        assert json["tipo"] == "pesquisa" and json["tarefa"] == "o que e X"
        return _FakeResp(200, {"relatorio": "RESPOSTA DO NEXUS", "arquivo_salvo": "output/x.txt"})
    monkeypatch.setattr(nexus_bridge.requests, "post", fake_post)
    monkeypatch.setattr("sys.argv", ["nexus_bridge.py", "--pesquisar", "o que e X"])
    cod = nexus_bridge.main()
    assert cod == 0
    assert "RESPOSTA DO NEXUS" in capsys.readouterr().out


def test_bridge_documentar_tipo(monkeypatch):
    capturado = {}
    def fake_post(url, json, timeout):
        capturado.update(json)
        return _FakeResp(200, {"relatorio": "doc"})
    monkeypatch.setattr(nexus_bridge.requests, "post", fake_post)
    monkeypatch.setattr("sys.argv", ["nexus_bridge.py", "--documentar", "modulo Y"])
    assert nexus_bridge.main() == 0
    assert capturado["tipo"] == "documentacao"


def test_bridge_api_fora(monkeypatch):
    def fake_post(*a, **k):
        raise requests.exceptions.ConnectionError("recusado")
    monkeypatch.setattr(nexus_bridge.requests, "post", fake_post)
    monkeypatch.setattr("sys.argv", ["nexus_bridge.py", "--pesquisar", "x"])
    assert nexus_bridge.main() == 2


def test_bridge_503(monkeypatch):
    monkeypatch.setattr(nexus_bridge.requests, "post",
                        lambda *a, **k: _FakeResp(503, {"detail": "ollama fora"}))
    monkeypatch.setattr("sys.argv", ["nexus_bridge.py", "--pesquisar", "x"])
    assert nexus_bridge.main() == 3


def test_bridge_exige_um_modo(monkeypatch):
    import pytest
    monkeypatch.setattr("sys.argv", ["nexus_bridge.py"])
    with pytest.raises(SystemExit):  # argparse: um dos dois é obrigatório
        nexus_bridge.main()
