"""Teste de fumaça de INTEGRAÇÃO ponta a ponta (exige Ollama + LLM real).

Fora da suíte hermética: roda um kickoff() de verdade no pipeline sequencial
(Analista → Pesquisador → Escritor → Revisor) para confirmar que o motor
completa o ciclo na GPU sem estourar a memória.

Marcado com 'llm' (opt-in) e PULADO automaticamente se o Ollama estiver
offline ou se a variante do modelo não existir — assim o `pytest` do dia a
dia continua rápido e nunca trava.

  pytest              # rápido: NÃO roda este teste (default -m "not llm")
  pytest -m llm -v    # roda ESTE teste (ou SKIPPED se o Ollama estiver off)
"""

from __future__ import annotations

import pytest

import main


def _motivo_para_pular() -> str | None:
    """Reaproveita o diagnóstico do projeto: retorna None se o ambiente está
    pronto (Ollama no ar E a variante qwen2.5-8k criada), ou a razão do skip.
    Mais robusto que só bater em /api/tags — evita um kickoff que falharia
    porque o modelo não foi baixado/criado."""
    try:
        return main.diagnostico_ollama() or None
    except Exception as erro:  # rede/instalação — melhor pular do que quebrar
        return f"não consegui checar o Ollama: {erro}"


_MOTIVO = _motivo_para_pular()


@pytest.mark.llm
@pytest.mark.skipif(_MOTIVO is not None, reason=_MOTIVO or "Ollama/modelo indisponível")
def test_pipeline_crewai_ponta_a_ponta():
    """Envia um pedido pequeno e fechado e verifica que a agência completa o
    ciclo e devolve uma resposta com contexto (não vazia, não alucinada)."""
    # Tema de ANÁLISE (verbo 'explique') curto — evita a classificação de
    # criação de código e não demora minutos à toa.
    tema = "Explique em uma frase curta o que é um teste de software."

    # Sequencial (usar_gerente=False): mais rápido e mantém a rede de segurança.
    # montar_equipe EXIGE o llm — usamos a fábrica do projeto (fix de tool-calling).
    equipe = main.montar_equipe(
        tema=tema, llm=main.criar_llm(), usar_memoria=False, usar_gerente=False
    )
    resultado = equipe.kickoff()

    # Extrai como o pipeline real faz (rede de segurança do código incluída).
    texto = main.escolher_saida(resultado, tema)

    assert texto is not None, "O modelo não retornou resposta."
    assert len(texto.strip()) > 10, f"Resposta vazia ou curta demais: {texto!r}"

    palavras_chave = (
        "teste", "software", "verificar", "verifica", "garantir", "garante",
        "código", "codigo", "qualidade", "sistema", "programa", "bug", "erro",
    )
    tem_contexto = any(p in texto.lower() for p in palavras_chave)
    assert tem_contexto, f"Respondeu, mas perdeu o contexto. Resposta: {texto!r}"
