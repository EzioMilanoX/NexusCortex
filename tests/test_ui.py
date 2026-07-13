"""Testes do painel Streamlit (ui.py).

O render é testado com o AppTest oficial do Streamlit (headless); as funções
auxiliares puras são testadas direto."""

from __future__ import annotations

import io

from streamlit.testing.v1 import AppTest

import ui


# --------------------------------------------------------------------------
# Funções auxiliares puras
# --------------------------------------------------------------------------
class _FakeUpload:
    def __init__(self, name, data: bytes):
        self.name = name
        self._d = data
    def getvalue(self):
        return self._d


def test_montar_contexto_extra_concatena_e_rotula():
    arqs = [_FakeUpload("a.txt", "conteúdo çã".encode("utf-8")),
            _FakeUpload("b.py", b"print(1)")]
    ctx = ui.montar_contexto_extra(arqs)
    assert "### Arquivo: a.txt ###" in ctx and "conteúdo çã" in ctx
    assert "### Arquivo: b.py ###" in ctx and "print(1)" in ctx


def test_montar_contexto_extra_vazio():
    assert ui.montar_contexto_extra([]) == ""


def test_montar_contexto_extra_trunca():
    grande = [_FakeUpload("big.txt", b"x" * (ui.MAX_CONTEXTO_CHARS + 5000))]
    ctx = ui.montar_contexto_extra(grande)
    assert "truncado" in ctx and len(ctx) <= ui.MAX_CONTEXTO_CHARS + 200


def test_listar_gerados(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "OUTPUT_DIR", tmp_path)
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / "a.py").write_text("print(1)")
    (tmp_path / "b.txt").write_text("texto")
    nomes = [p.name for p in ui.listar_gerados()]
    assert set(nomes) == {"a.py", "b.txt"}  # ignora o .gitkeep


# --------------------------------------------------------------------------
# Render headless (AppTest)
# --------------------------------------------------------------------------
def test_painel_renderiza_com_controles():
    at = AppTest.from_file("ui.py").run(timeout=30)
    assert not at.exception
    assert any("Modo" in r.label for r in at.radio)
    assert any("Memória" in t.label for t in at.sidebar.toggle)
    assert any("Timeout" in n.label for n in at.number_input)
    assert any("Busca Semântica" in t.label for t in at.text_input)
    assert len(at.get("file_uploader")) == 1


def test_painel_renderiza_resposta_no_historico():
    at = AppTest.from_file("ui.py")
    at.session_state["historico"] = [
        {"papel": "assistant", "conteudo": "```python\nprint('oi')\n```",
         "conteudo_limpo": "print('oi')", "tipo": "codigo", "linguagem": "python",
         "arquivo_nome": "oi.py", "payload": {"tema": "t", "usar_memoria": False}},
    ]
    at.run(timeout=30)
    assert not at.exception
    assert len(at.tabs) == 2  # abas Análise/Código
    assert any("Baixar oi.py" in b.label for b in at.get("download_button"))
