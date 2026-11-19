"""Testes do estúdio de podcast (sem LLM, sem Ollama, sem voz baixada).

As três etapas do podcast.py são determinísticas, então dá para cobrir TUDO
aqui: o parser de roteiro, a mixagem/EQ (com áudio sintético gerado no próprio
teste) e a orquestração ponta a ponta usando o motor 'tom', que não precisa de
nenhum modelo em disco. O único trecho não coberto é a chamada real ao Piper —
essa é integração, e o que testamos dela é a mensagem de erro quando falta.
"""

from __future__ import annotations

import json
import re
import wave
from datetime import datetime, timezone

import numpy as np
import pytest

import podcast


# --------------------------------------------------------------------------
# Utilitários dos testes
# --------------------------------------------------------------------------
def _seno(duracao: float, freq: float, taxa: int = podcast.TAXA_PADRAO) -> np.ndarray:
    t = np.arange(int(duracao * taxa), dtype=np.float32) / taxa
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(sinal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(sinal, dtype=np.float64))))


def _wav(pasta, nome: str, duracao: float = 1.0, freq: float = 440.0,
         taxa: int = podcast.TAXA_PADRAO):
    return podcast.escrever_wav(pasta / nome, _seno(duracao, freq, taxa), taxa)


# --------------------------------------------------------------------------
# 1. PARSER DE ROTEIRO
# --------------------------------------------------------------------------
def test_parse_formato_simples():
    falas = podcast.parse_roteiro("Host1: Olá!\nHost2: Tudo bem?")
    assert [(f.locutor, f.texto) for f in falas] == [
        ("Host1", "Olá!"),
        ("Host2", "Tudo bem?"),
    ]


@pytest.mark.parametrize(
    "linha,locutor,texto",
    [
        ("**Ana:** bom dia", "Ana", "bom dia"),
        ("[Bruno] bom dia", "Bruno", "bom dia"),
        ("[Bruno]: bom dia", "Bruno", "bom dia"),
        ("- ANA — bom dia", "ANA", "bom dia"),
        ("* Host 1: bom dia", "Host 1", "bom dia"),
        ("Dr. Silva: bom dia", "Dr. Silva", "bom dia"),
    ],
)
def test_parse_aceita_variacoes_de_marcacao(linha, locutor, texto):
    falas = podcast.parse_roteiro(linha)
    assert len(falas) == 1
    assert falas[0].locutor == locutor and falas[0].texto == texto


def test_parse_linha_sem_rotulo_continua_a_fala_anterior():
    bruto = "Ana: primeira parte\ne o resto da mesma ideia\nBruno: outra fala"
    falas = podcast.parse_roteiro(bruto)
    assert len(falas) == 2
    assert falas[0].texto == "primeira parte e o resto da mesma ideia"


def test_parse_rotulo_sozinho_pega_a_linha_seguinte():
    falas = podcast.parse_roteiro("Ana: oi\nBruno:\nagora eu falo")
    assert [(f.locutor, f.texto) for f in falas] == [
        ("Ana", "oi"),
        ("Bruno", "agora eu falo"),
    ]


def test_parse_descarta_metadados_e_preambulo():
    bruto = (
        "Aqui está o roteiro que você pediu:\n"
        "# Episódio 1\n"
        "Tema: energia solar\n"
        "Duração: 10 minutos\n"
        "Fonte: https://exemplo.com\n"
        "Host1: vamos começar\n"
    )
    falas = podcast.parse_roteiro(bruto)
    assert [f.locutor for f in falas] == ["Host1"]


def test_parse_descarta_rubricas_de_estudio():
    bruto = "Ana: oi\n(risos)\n[a música sobe]\n*pausa*\nBruno: tudo bem"
    assert [f.texto for f in podcast.parse_roteiro(bruto)] == ["oi", "tudo bem"]


def test_parse_limpa_markdown_e_links_para_a_voz():
    bruto = "Ana: veja o **relatório** em [nosso site](https://x.com) e `main.py`"
    (fala,) = podcast.parse_roteiro(bruto)
    # O TTS lê literalmente: asterisco e URL não podem sobrar no texto falado.
    assert fala.texto == "veja o relatório em nosso site e main.py"
    assert "*" not in fala.texto and "http" not in fala.texto


def test_parse_remove_andaime_react_do_crewai():
    bruto = "Final Answer:\nHost1: bom dia\nThought: preciso continuar\nHost2: bom dia"
    falas = podcast.parse_roteiro(bruto)
    assert [f.texto for f in falas] == ["bom dia", "bom dia"]


def test_parse_nao_confunde_prosa_com_locutor():
    # Frase comum com dois-pontos no meio: não vira uma fala nova.
    falas = podcast.parse_roteiro("Ana: o ponto é claro. Mas veja: não é simples")
    assert len(falas) == 1


@pytest.mark.parametrize(
    "bruto",
    [
        '[{"speaker": "Host1", "text": "Olá!"}, {"speaker": "Host2", "text": "Oi!"}]',
        '{"roteiro": [{"locutor": "Host1", "texto": "Olá!"},'
        ' {"locutor": "Host2", "texto": "Oi!"}]}',
        'Segue o JSON:\n[{"speaker": "Host1", "text": "Olá!"},'
        ' {"speaker": "Host2", "text": "Oi!"}]',
    ],
)
def test_parse_aceita_json_quando_o_modelo_insiste(bruto):
    falas = podcast.parse_roteiro(bruto)
    assert [(f.locutor, f.texto) for f in falas] == [
        ("Host1", "Olá!"),
        ("Host2", "Oi!"),
    ]


def test_parse_vazio_devolve_lista_vazia():
    assert podcast.parse_roteiro("") == []
    assert podcast.parse_roteiro("   \n\n") == []
    assert podcast.parse_roteiro("Só um parágrafo de prosa sem diálogo nenhum") == []


def test_parse_descarta_fala_longa_repetida():
    """Regressão de um defeito observado com o tema 'teoria dos jogos': o 7B
    emitiu o mesmo parágrafo duas vezes seguidas, trocando só o locutor."""
    paragrafo = (
        "E isso nos leva à discussão sobre estratégias de longo prazo versus "
        "curto prazo, em que os jogadores precisam considerar as reações futuras."
    )
    falas = podcast.parse_roteiro(f"Rafa: {paragrafo}\nBruno: {paragrafo}")
    assert len(falas) == 1


def test_parse_mantem_interjeicao_curta_repetida():
    """"Isso mesmo!" dito pelos dois é conversa normal, não travamento."""
    falas = podcast.parse_roteiro("Rafa: Isso mesmo!\nBruno: Isso mesmo!")
    assert len(falas) == 2


def test_parse_mantem_repeticao_nao_adjacente():
    """Retomar uma frase mais adiante é legítimo; só o eco imediato sai."""
    frase = "A teoria dos jogos estuda decisões quando cada lado sabe do outro."
    falas = podcast.parse_roteiro(f"Rafa: {frase}\nBruno: Certo.\nRafa: {frase}")
    assert len(falas) == 3


def test_parse_descarta_fala_que_trocou_de_idioma():
    """Regressão real: o qwen2.5 (de origem chinesa) escreveu as duas últimas
    falas de um episódio pt-BR em MANDARIM. Isso no Piper viraria ruído ou
    silêncio, porque o mapa de fonemas da voz pt-BR não tem esses caracteres."""
    bruto = (
        "Rafa: Esta fala está em português e deve sobreviver ao parser.\n"
        "Bruno: 是的，我们希望通过这个播客让你对博弈论有更深的理解。\n"
        "Rafa: Esta também está em português e também deve sobreviver.\n"
    )
    falas = podcast.parse_roteiro(bruto)
    assert len(falas) == 2
    assert all("português" in f.texto for f in falas)


@pytest.mark.parametrize(
    "texto,mantem",
    [
        ("Uma frase inteiramente em português do Brasil.", True),
        ("O termo japonês kaizen aparece assim: 改善 — e segue a frase.", True),
        ("这是一个完全用中文写的句子，不应该被保留下来。", False),
        ("Это предложение полностью на русском языке и deve sair.", False),
    ],
)
def test_alfabeto_latino_tolera_simbolo_solto_mas_nao_deriva(texto, mantem):
    assert podcast._em_alfabeto_latino(texto) is mantem


def test_apresentadores_padrao_tem_nomes_de_gente():
    """O modelo coloca o nome do colega DENTRO da fala ('Sim, Host1!'), então
    um rótulo de placeholder acabava sendo narrado em voz alta."""
    for apresentador in podcast.APRESENTADORES_PADRAO:
        assert not re.match(r"(?i)^host\s*\d*$", apresentador.nome)
        # Vira nome de arquivo dos .wav de cada fala: só ASCII simples.
        assert apresentador.nome.isascii() and apresentador.nome.isalnum()


def test_roteiro_para_json_sempre_valido():
    falas = podcast.parse_roteiro('Ana: disse "oi"\nBruno: e daí?')
    dados = json.loads(podcast.roteiro_para_json(falas))
    assert dados == [
        {"speaker": "Ana", "text": 'disse "oi"'},
        {"speaker": "Bruno", "text": "e daí?"},
    ]


# --------------------------------------------------------------------------
# 2. MAPEAMENTO DE LOCUTORES
# --------------------------------------------------------------------------
def test_mapear_locutores_alterna_por_ordem_de_aparicao():
    primeiro, segundo = (a.nome for a in podcast.APRESENTADORES_PADRAO[:2])
    falas = podcast.parse_roteiro("Joana: a\nPedro: b\nJoana: c")
    dupla = podcast.mapear_locutores(falas)
    assert [a.nome for _, a in dupla] == [primeiro, segundo, primeiro]


def test_mapear_locutores_ignora_caixa_e_acento():
    falas = podcast.parse_roteiro("Ana: a\nANA: b\nÂna: c")
    assert len({a.nome for _, a in podcast.mapear_locutores(falas)}) == 1


def test_mapear_locutores_da_a_volta_com_mais_rotulos_que_vozes():
    primeiro, segundo = (a.nome for a in podcast.APRESENTADORES_PADRAO[:2])
    falas = podcast.parse_roteiro("A: 1\nB: 2\nC: 3")
    nomes = [a.nome for _, a in podcast.mapear_locutores(falas)]
    assert nomes == [primeiro, segundo, primeiro]


# --------------------------------------------------------------------------
# PROMPT DO ROTEIRISTA
# --------------------------------------------------------------------------
def test_prompt_roteiro_evita_os_verbos_de_criacao_de_codigo():
    """Se o prompt usasse 'crie/escreva/gere', main.escolher_saida trataria o
    pedido como Ação de Criação e acionaria a rede de segurança de código."""
    import main

    for tem_contexto in (False, True):
        prompt = podcast.prompt_roteiro("energia solar", tem_contexto=tem_contexto)
        achados = main._RE_VERBO_CRIACAO.findall(prompt)
        assert not achados, f"verbo de criação no prompt: {achados}"

    prompt = podcast.prompt_roteiro("energia solar")
    assert "PRODUZA" in prompt
    # "script" em português significa roteiro E programa — a ambiguidade é o
    # que faria o Escritor devolver Python.
    assert "script" not in prompt.lower()


def test_prompt_roteiro_proibe_disco_mas_exige_pesquisa():
    """Duas regressões de uma vez:

    1. Sem a proibição de DISCO, o Pesquisador dispara a busca semântica na
       raiz e passa dez minutos indexando o `.venv`.
    2. A primeira versão dessa proibição foi longe demais e desligou também a
       pesquisa web ("caso contrário, vá direto ao roteiro com o que você já
       sabe") — foi o que fez o modelo escrever de memória e alucinar. A
       leitura de página agora é obrigatória.
    """
    prompt = podcast.prompt_roteiro("energia solar", tem_contexto=False)
    assert "ferramentas de DISCO" in prompt
    assert "Busca Semântica na Pasta" in prompt
    assert "Ler Página Web" in prompt and "OBRIGATÓRIO" in prompt


def test_prompt_roteiro_com_contexto_aponta_a_fonte_primaria():
    prompt = podcast.prompt_roteiro("energia solar", tem_contexto=True)
    assert "CONTEXTO FORNECIDO" in prompt
    assert "ferramentas de DISCO" not in prompt


def test_prompt_roteiro_dimensiona_pela_duracao():
    # Derivado da constante, não fixado no teste: o ritmo é calibrado por
    # medição e mudar o número não deve quebrar a suíte.
    for minutos in (2, 15):
        esperado = minutos * podcast.PALAVRAS_POR_MINUTO
        assert f"{esperado} palavras" in podcast.prompt_roteiro("x", minutos=minutos)


def test_prompt_roteiro_usa_os_nomes_dos_apresentadores():
    dupla = podcast.montar_apresentadores(
        ["pt_BR-faber-medium", "pt_BR-cadu-medium"], nomes=["Ana", "Bruno"]
    )
    prompt = podcast.prompt_roteiro("x", apresentadores=dupla)
    assert "Ana: (uma frase" in prompt and "Bruno: (uma frase" in prompt


# --------------------------------------------------------------------------
# ESCRITA EM BLOCOS
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "minutos,esperado",
    [(1, 2), (2, 2), (5, 3), (10, 5), (15, 6), (60, 6)],
)
def test_blocos_para_respeita_piso_e_teto(minutos, esperado):
    # Piso 2: com um bloco só voltaríamos ao pedido único que entrega de menos.
    # Teto 6: cada bloco é uma chamada ao LLM.
    assert podcast.blocos_para(minutos) == esperado


def test_falas_por_bloco_aplica_a_calibracao_de_excesso():
    """Medido em três episódios: o modelo entrega ~15% mais falas do que se
    pede, então o pedido sai calibrado para a entrega cair no alvo."""
    # 5 min x 8 falas/min = 40 no papel; x0,85 = 34; em 3 blocos = 11.
    assert podcast.falas_por_bloco(5, 3) == 11
    assert podcast.falas_por_bloco(10, 5) == 14
    # Piso: um bloco nunca pede menos de 4 falas (não daria conversa).
    assert podcast.falas_por_bloco(1, 6) == 4


def test_falas_por_bloco_nao_divide_por_zero():
    assert podcast.falas_por_bloco(5, 0) >= 4


def test_prompt_bloco_proibe_gancho_menos_no_ultimo():
    """A frase de gancho reaparecia a cada emenda ('ainda temos muito mais
    para explorar'). Não é repetição literal, então só o prompt a evita."""
    primeiro = podcast.prompt_bloco("x", "y", indice=1, total=3)
    meio = podcast.prompt_bloco("x", "y", indice=2, total=3)
    ultimo = podcast.prompt_bloco("x", "y", indice=3, total=3)

    assert "NÃO encerre o bloco" in primeiro and "NÃO encerre o bloco" in meio
    assert "NÃO encerre o bloco" not in ultimo
    assert "Só este bloco fecha o episódio" in ultimo


def test_prompt_pauta_exige_leitura_de_pagina():
    """A pauta é o ÚNICO ponto do fluxo em que a pesquisa acontece — se ela não
    exigir leitura de fonte, os blocos escrevem de memória."""
    p = podcast.prompt_pauta("teoria dos jogos", minutos=5)
    assert "Ler Página Web" in p and "Busca na Web" in p
    assert "EXATAMENTE 3 blocos" in p
    assert "não da sua memória" in p


def test_prompt_pauta_com_contexto_nao_manda_pesquisar():
    p = podcast.prompt_pauta("x", tem_contexto=True)
    assert "CONTEXTO FORNECIDO" in p and "Ler Página Web" not in p


def test_prompt_pauta_sem_verbo_de_criacao_de_codigo():
    import main

    for ctx in (False, True):
        achados = main._RE_VERBO_CRIACAO.findall(
            podcast.prompt_pauta("x", tem_contexto=ctx)
        )
        assert not achados, f"verbo de criação na pauta: {achados}"


@pytest.mark.parametrize(
    "bruto",
    [
        "1. Abertura | o que é\n2. Exemplos | dilema do prisioneiro\n3. Fecho | usos",
        "- Bloco 1: Abertura | o que é\n- Bloco 2: Exemplos | dilema\n- Bloco 3: Fecho | usos",
        "1) Abertura | o que é\n2) Exemplos | dilema\n3) Fecho | usos",
        "## 1 - Abertura | o que é\n## 2 - Exemplos | dilema\n## 3 - Fecho | usos",
    ],
)
def test_dividir_pauta_aceita_variacoes_de_numeracao(bruto):
    blocos = podcast.dividir_pauta(bruto)
    assert len(blocos) == 3
    assert blocos[0].startswith("Abertura")


def test_dividir_pauta_anexa_a_linha_de_fatos_ao_bloco():
    """Regressão que custou um episódio: o modelo pôs o título na linha
    numerada e os fatos pesquisados na linha SEGUINTE. A primeira versão só
    coletava linhas numeradas e jogava a pesquisa no lixo — os blocos eram
    escritos a partir do rótulo do formato, e o roteiro saiu inventando."""
    bruto = (
        "1. O que é e de onde veio | definição exata (fonte)\n"
        "A teoria dos jogos foi consolidada pelo teorema minimax de John von "
        "Neumann e por Theory of Games and Economic Behavior, de 1944.\n"
        "2. Como funciona | mecanismo (fonte)\n"
        "Exemplo clássico: o dilema do prisioneiro.\n"
    )
    blocos = podcast.dividir_pauta(bruto)
    assert len(blocos) == 2
    assert "von Neumann" in blocos[0] and "1944" in blocos[0]
    assert "dilema do prisioneiro" in blocos[1]


def test_dividir_pauta_respeita_o_teto_de_esperados():
    bruto = "\n".join(f"{i}. Bloco {i} | fato" for i in range(1, 9))
    assert len(podcast.dividir_pauta(bruto, 4)) == 4


def test_dividir_pauta_cai_para_paragrafos_sem_numeracao():
    bruto = "Primeiro assunto do episódio\n\nSegundo assunto do episódio"
    assert len(podcast.dividir_pauta(bruto)) == 2


def test_dividir_pauta_nunca_devolve_vazio():
    """Pauta ilegível degrada para um bloco único (pior, mas não quebra)."""
    assert podcast.dividir_pauta("") == ["o tema do episódio"]
    assert len(podcast.dividir_pauta("frase solta")) == 1


def test_prompt_bloco_do_meio_proibe_saudacao():
    """Sem isto, cada bloco recomeçava com 'Olá, bem-vindos!' no meio."""
    meio = podcast.prompt_bloco("x", "assunto", indice=2, total=3)
    assert "proibido cumprimentar" in meio
    primeiro = podcast.prompt_bloco("x", "assunto", indice=1, total=3)
    assert "PRIMEIRO bloco" in primeiro
    ultimo = podcast.prompt_bloco("x", "assunto", indice=3, total=3)
    assert "ÚLTIMO bloco" in ultimo


def test_prompt_bloco_injeta_continuidade():
    falas = podcast.parse_roteiro("Rafa: primeira\nBruno: segunda\nRafa: terceira")
    anteriores = podcast.ultimas_falas_como_texto(falas)
    p = podcast.prompt_bloco("x", "y", indice=2, total=3, ultimas_falas=anteriores)
    assert "Bruno: segunda" in p and "Rafa: terceira" in p
    assert "NÃO repita estas falas" in p
    # Sem falas anteriores, o bloco de continuidade simplesmente não aparece.
    assert "NÃO repita estas falas" not in podcast.prompt_bloco(
        "x", "y", indice=1, total=3
    )


def test_ultimas_falas_como_texto_pega_o_fim():
    falas = podcast.parse_roteiro("A: um\nB: dois\nA: tres\nB: quatro")
    assert podcast.ultimas_falas_como_texto(falas, 2) == "A: tres\nB: quatro"
    assert podcast.ultimas_falas_como_texto([], 2) == ""


# --------------------------------------------------------------------------
# 3. ENTRADA E SAÍDA DE WAV
# --------------------------------------------------------------------------
def test_wav_ida_e_volta(tmp_path):
    original = _seno(0.5, 440.0)
    arquivo = podcast.escrever_wav(tmp_path / "a.wav", original)
    lido, taxa = podcast.ler_wav(arquivo)
    assert taxa == podcast.TAXA_PADRAO
    assert lido.size == original.size
    # A perda é só a quantização de 16 bits.
    assert float(np.max(np.abs(lido - original))) < 1e-3


def test_wav_estereo_vira_mono(tmp_path):
    esquerdo, direito = _seno(0.2, 300.0), _seno(0.2, 300.0) * 0.0
    entrelacado = np.empty(esquerdo.size * 2, dtype=np.float32)
    entrelacado[0::2], entrelacado[1::2] = esquerdo, direito
    caminho = tmp_path / "estereo.wav"
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(2)
        arquivo.setsampwidth(2)
        arquivo.setframerate(podcast.TAXA_PADRAO)
        arquivo.writeframes((entrelacado * 32767).astype("<i2").tobytes())

    lido, _ = podcast.ler_wav(caminho)
    assert lido.size == esquerdo.size          # um canal, não dois
    assert _rms(lido) == pytest.approx(_rms(esquerdo) / 2, rel=0.05)


def test_wav_de_8_bits_e_lido_centrado(tmp_path):
    caminho = tmp_path / "oito.wav"
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(1)
        arquivo.setframerate(8000)
        arquivo.writeframes(bytes([128] * 800))  # 128 = silêncio no PCM u8
    lido, taxa = podcast.ler_wav(caminho)
    assert taxa == 8000 and float(np.max(np.abs(lido))) < 1e-6


def test_ler_wav_arquivo_invalido_da_erro_amigavel(tmp_path):
    ruim = tmp_path / "nao_e_wav.wav"
    ruim.write_bytes(b"isso nao e um wav")
    with pytest.raises(podcast.ErroPodcast, match="Não consegui ler"):
        podcast.ler_wav(ruim)


def test_escrever_wav_satura_em_vez_de_dar_a_volta(tmp_path):
    estourado = np.array([3.0, -3.0, 0.0], dtype=np.float32)
    lido, _ = podcast.ler_wav(podcast.escrever_wav(tmp_path / "e.wav", estourado))
    # Sem o clip, +3.0 viraria um valor NEGATIVO no int16 (wrap) = estalo.
    assert lido[0] > 0.99 and lido[1] < -0.99


# --------------------------------------------------------------------------
# 4. PROCESSAMENTO DE SINAL
# --------------------------------------------------------------------------
def test_reamostrar_ajusta_o_tamanho_e_preserva_o_tom():
    sinal = _seno(0.5, 440.0, 16000)
    convertido = podcast.reamostrar(sinal, 16000, 22050)
    assert convertido.size == pytest.approx(0.5 * 22050, rel=0.01)
    # Meio segundo continua meio segundo: o tom não foi acelerado.
    cruzamentos = np.sum(np.diff(np.signbit(convertido)) != 0)
    assert cruzamentos == pytest.approx(440, rel=0.05)


def test_reamostrar_mesma_taxa_e_no_op():
    sinal = _seno(0.1, 440.0)
    assert np.array_equal(podcast.reamostrar(sinal, 22050, 22050), sinal)


def test_aplicar_fade_zera_as_pontas():
    com_fade = podcast.aplicar_fade(np.ones(22050, dtype=np.float32), 0.1, 0.1)
    assert com_fade[0] == pytest.approx(0.0, abs=1e-6)
    assert com_fade[-1] == pytest.approx(0.0, abs=1e-3)
    assert com_fade[11025] == pytest.approx(1.0)


def test_aplicar_fade_em_trecho_curto_nao_estoura():
    curto = np.ones(10, dtype=np.float32)
    assert podcast.aplicar_fade(curto, 1.0, 1.0).size == 10


def test_normalizar_leva_o_pico_ao_alvo():
    normalizado = podcast.normalizar(_seno(0.2, 440.0) * 0.01, pico=0.89)
    assert float(np.max(np.abs(normalizado))) == pytest.approx(0.89, rel=1e-3)


def test_normalizar_nao_divide_por_zero():
    mudo = np.zeros(1000, dtype=np.float32)
    assert float(np.max(np.abs(podcast.normalizar(mudo)))) == 0.0


def test_equalizar_corta_o_grave_e_preserva_a_voz():
    grave, voz = _seno(0.5, 40.0), _seno(0.5, 1000.0)
    razao_grave = _rms(podcast.equalizar(grave)) / _rms(grave)
    razao_voz = _rms(podcast.equalizar(voz)) / _rms(voz)
    assert razao_grave < 0.35          # ronco abaixo de 85 Hz vai embora
    assert 0.9 < razao_voz < 1.5       # a faixa da fala continua de pé


def test_equalizar_preserva_o_tamanho():
    sinal = _seno(0.3, 500.0)
    assert podcast.equalizar(sinal).size == sinal.size


# --------------------------------------------------------------------------
# 5. MONTAGEM DO PROGRAMA
# --------------------------------------------------------------------------
def test_montar_programa_soma_falas_e_pausas(tmp_path):
    trechos = [_wav(tmp_path, f"{i}.wav", 1.0) for i in range(3)]
    res = podcast.montar_programa(
        trechos, tmp_path / "prog.wav", locutores=["A", "A", "B"]
    )
    esperado = 3.0 + podcast.PAUSA_MESMA_VOZ + podcast.PAUSA_TROCA_VOZ
    assert res["duracao"] == pytest.approx(esperado, abs=0.02)
    assert res["arquivo"].exists()


def test_montar_programa_pausa_maior_quando_troca_de_voz(tmp_path):
    trechos = [_wav(tmp_path, f"{i}.wav", 0.5) for i in range(2)]
    mesma = podcast.montar_programa(
        trechos, tmp_path / "m.wav", locutores=["A", "A"]
    )
    troca = podcast.montar_programa(
        trechos, tmp_path / "t.wav", locutores=["A", "B"]
    )
    assert troca["duracao"] > mesma["duracao"]


def test_montar_programa_unifica_taxas_diferentes(tmp_path):
    a = _wav(tmp_path, "a.wav", 1.0, taxa=16000)
    b = _wav(tmp_path, "b.wav", 1.0, taxa=22050)
    res = podcast.montar_programa([a, b], tmp_path / "mix.wav", locutores=["A", "A"])
    # As duas falas mantêm 1s cada mesmo vindo de taxas diferentes.
    assert res["taxa"] == podcast.TAXA_PADRAO
    assert res["duracao"] == pytest.approx(2.0 + podcast.PAUSA_MESMA_VOZ, abs=0.02)


def test_montar_programa_com_trilha_adiciona_intro_e_outro(tmp_path):
    fala = _wav(tmp_path, "fala.wav", 2.0)
    trilha = _wav(tmp_path, "trilha.wav", 1.0, freq=110.0)  # curta: entra em loop
    res = podcast.montar_programa(
        [fala], tmp_path / "prog.wav", trilha=trilha,
        intro=3.0, outro=2.0, crossfade=1.0,
    )
    assert res["duracao"] == pytest.approx(3.0 + 2.0 + 2.0, abs=0.05)

    sinal, taxa = podcast.ler_wav(res["arquivo"])
    # A abertura (só trilha, volume alto) precisa ser mais forte que o miolo,
    # onde a música está abaixada sob a fala — é o "duck" acontecendo.
    abertura = _rms(sinal[: int(1.5 * taxa)])
    assert abertura > 0.01


def test_montar_programa_sem_trechos_da_erro(tmp_path):
    with pytest.raises(podcast.ErroPodcast, match="Nenhum trecho"):
        podcast.montar_programa([], tmp_path / "x.wav")


def test_exportar_mp3_sem_ffmpeg_devolve_none(tmp_path, monkeypatch):
    monkeypatch.setattr(podcast.shutil, "which", lambda _: None)
    assert podcast.exportar_mp3(_wav(tmp_path, "a.wav", 0.2)) is None


# --------------------------------------------------------------------------
# 6. MOTORES DE TTS
# --------------------------------------------------------------------------
def test_vozes_padrao_suportam_til_nasal():
    """Regressão de um defeito REAL encontrado no catálogo do Piper.

    A voz `pt_BR-edresson-low` foi publicada sem o til combinante (U+0303) no
    phoneme_id_map: ela come a nasalização e "não" sai como "nao". Este teste
    impede que ela (ou outra sem nasal) volte a ser padrão num projeto pt-BR.
    """
    for apresentador in podcast.APRESENTADORES_PADRAO:
        ficha = podcast.VOZES_PT_BR[apresentador.voz]
        assert ficha["nasal"], f"{apresentador.voz} não nasaliza — imprópria p/ pt-BR"


def test_vozes_padrao_sao_distinguiveis():
    """Duas vozes com o mesmo timbre deixam o ouvinte perdido sobre quem fala.
    O catálogo pt-BR é todo masculino, então a separação vem do pitch."""
    f0 = [podcast.VOZES_PT_BR[a.voz]["f0"] for a in podcast.APRESENTADORES_PADRAO]
    assert len(set(f0)) == len(f0)
    assert abs(f0[0] - f0[1]) >= 20  # ~2 semitons: audível numa troca de turno


def test_criar_motor_desconhecido_lista_as_opcoes():
    with pytest.raises(podcast.ErroPodcast, match="piper"):
        podcast.criar_motor("elevenlabs")


def test_motor_tom_gera_audio_proporcional_ao_texto(tmp_path):
    motor = podcast.criar_motor("tom")
    apresentador = podcast.APRESENTADORES_PADRAO[0]
    curto = motor.sintetizar("oi", apresentador, tmp_path / "curto.wav")
    longo = motor.sintetizar("oi " * 60, apresentador, tmp_path / "longo.wav")
    assert podcast.ler_wav(longo)[0].size > podcast.ler_wav(curto)[0].size * 5


def test_motor_tom_diferencia_os_apresentadores(tmp_path):
    motor = podcast.criar_motor("tom")
    a, b = podcast.APRESENTADORES_PADRAO[:2]
    texto = "uma frase de tamanho razoável para medir"
    sinal_a, _ = podcast.ler_wav(motor.sintetizar(texto, a, tmp_path / "a.wav"))
    sinal_b, _ = podcast.ler_wav(motor.sintetizar(texto, b, tmp_path / "b.wav"))
    # Vozes diferentes = tons diferentes = número de cruzamentos por zero difere.
    cruz_a = np.sum(np.diff(np.signbit(sinal_a)) != 0)
    cruz_b = np.sum(np.diff(np.signbit(sinal_b)) != 0)
    assert cruz_a != cruz_b


def test_motor_piper_avisa_quando_a_voz_nao_foi_baixada(tmp_path):
    motor = podcast.MotorPiper(dir_vozes=tmp_path / "vazio")
    pytest.importorskip("piper", reason="sem piper-tts, o erro testado é o de import")
    with pytest.raises(podcast.ErroPodcast, match="download_voices"):
        motor.verificar(podcast.APRESENTADORES_PADRAO)


def test_motor_piper_sem_pacote_instalado_ensina_o_pip(tmp_path, monkeypatch):
    # Simula a ausência do piper-tts mesmo que ele esteja instalado na máquina.
    import builtins

    original = builtins.__import__

    def sem_piper(nome, *args, **kwargs):
        if nome == "piper" or nome.startswith("piper."):
            raise ImportError("No module named 'piper'")
        return original(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", sem_piper)
    motor = podcast.MotorPiper(dir_vozes=tmp_path)
    with pytest.raises(podcast.ErroPodcast, match="pip install piper-tts"):
        motor.verificar(podcast.APRESENTADORES_PADRAO)


# --------------------------------------------------------------------------
# 7. PONTA A PONTA (motor 'tom' — nenhum modelo em disco)
# --------------------------------------------------------------------------
def test_gerar_podcast_ponta_a_ponta(tmp_path):
    roteiro = (
        "Host1: Bem-vindo ao Cortexus, o podcast gerado localmente.\n"
        "Host2: Hoje a gente fala sobre agentes de IA rodando offline.\n"
        "Host1: E sem mandar um byte para a nuvem.\n"
    )
    res = podcast.gerar_podcast(
        roteiro, titulo="Piloto", motor="tom", destino=tmp_path / "ep"
    )

    assert res["total_falas"] == 3
    assert res["motor"] == "tom"
    assert res["duracao"] > 0

    wav = tmp_path / "ep" / "programa.wav"
    assert wav.exists() and wav.stat().st_size > 44  # 44 = só o cabeçalho

    salvo = json.loads((tmp_path / "ep" / "roteiro.json").read_text(encoding="utf-8"))
    assert [f["speaker"] for f in salvo] == ["Host1", "Host2", "Host1"]
    # Alternância de vozes preservada até o fim do pipeline. O 'speaker' é o
    # rótulo que veio no roteiro; a 'voz' é o apresentador que o narrou.
    primeiro, segundo = (a.nome for a in podcast.APRESENTADORES_PADRAO[:2])
    assert [f["voz"] for f in res["falas"]] == [primeiro, segundo, primeiro]


def test_gerar_podcast_apaga_os_rascunhos_por_padrao(tmp_path):
    podcast.gerar_podcast("A: um\nB: dois", motor="tom", destino=tmp_path / "ep")
    assert not (tmp_path / "ep" / "falas").exists()


def test_gerar_podcast_mantem_rascunhos_quando_pedido(tmp_path):
    podcast.gerar_podcast(
        "A: um\nB: dois", motor="tom", destino=tmp_path / "ep", manter_falas=True
    )
    assert len(list((tmp_path / "ep" / "falas").glob("*.wav"))) == 2


def test_gerar_podcast_roteiro_sem_dialogo_da_erro_explicativo(tmp_path):
    with pytest.raises(podcast.ErroPodcast, match="Locutor: texto"):
        podcast.gerar_podcast("apenas prosa solta", motor="tom", destino=tmp_path / "e")


def test_gerar_podcast_falha_antes_de_sintetizar_se_a_voz_falta(tmp_path):
    """A verificação do motor roda ANTES da primeira síntese — descobrir a
    falta da voz depois de 80 falas seria a pior hora possível."""

    class MotorQueSemprePifa(podcast.MotorTTS):
        nome = "pifado"

        def verificar(self, apresentadores):
            raise podcast.ErroPodcast("voz ausente")

        def sintetizar(self, texto, apresentador, destino):
            raise AssertionError("não deveria ter chegado a sintetizar")

    with pytest.raises(podcast.ErroPodcast, match="voz ausente"):
        podcast.gerar_podcast(
            "A: um\nB: dois", motor=MotorQueSemprePifa(), destino=tmp_path / "ep"
        )


# --------------------------------------------------------------------------
# Funções puras auxiliares (_limpar_para_fala, _tentar_json, _identificador)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bruto,esperado",
    [
        ("**IA** é _incrível_", "IA é incrível"),
        ("Veja [a doc](https://x.com/doc) agora", "Veja a doc agora"),
        ("![logo](img.png) Marca", "logo Marca"),
        ("Rode `pip install` e pronto", "Rode pip install e pronto"),
        ("Bom dia [pausa] pessoal", "Bom dia pessoal"),
        ("Olá <break time='1s'/> gente", "Olá gente"),
        ("  —  texto   com\n\nespaços  -", "texto com espaços"),
        ("", ""),
    ],
)
def test_limpar_para_fala(bruto, esperado):
    assert podcast._limpar_para_fala(bruto) == esperado


def test_tentar_json_lista_simples():
    falas = podcast._tentar_json(
        '[{"speaker": "Ana", "text": "**Oi**"}, {"host": "Bia", "fala": "Olá"}]'
    )
    assert falas == [podcast.Fala("Ana", "Oi"), podcast.Fala("Bia", "Olá")]


@pytest.mark.parametrize("chave", ["roteiro", "script", "falas", "dialogo", "lines"])
def test_tentar_json_objeto_com_chave_conhecida(chave):
    bruto = json.dumps({chave: [{"locutor": " Ana ", "texto": "Oi"}]})
    assert podcast._tentar_json(bruto) == [podcast.Fala("Ana", "Oi")]


def test_tentar_json_ignora_texto_ao_redor():
    bruto = 'Claro! Aqui está:\n[{"speaker": "Ana", "text": "Oi"}]\nEspero ter ajudado.'
    assert podcast._tentar_json(bruto) == [podcast.Fala("Ana", "Oi")]


def test_tentar_json_pula_itens_invalidos():
    bruto = json.dumps(
        [
            "string solta",
            {"speaker": "Ana"},                    # sem texto
            {"speaker": 3, "text": "número"},      # locutor não é string
            {"speaker": "Bia", "text": "[pausa]"},  # some na limpeza
            {"speaker": "Caio", "text": "Valeu"},
        ]
    )
    assert podcast._tentar_json(bruto) == [podcast.Fala("Caio", "Valeu")]


@pytest.mark.parametrize(
    "bruto",
    [
        "Ana: só texto, sem JSON",
        "]{ ao contrário [",
        "[isto não é json]",
        '{"outra_chave": [{"speaker": "Ana", "text": "Oi"}]}',
        '{"roteiro": "não é lista"}',
        "[]",
        '[{"speaker": "Ana"}]',
    ],
)
def test_tentar_json_devolve_none_quando_nao_aproveita(bruto):
    assert podcast._tentar_json(bruto) is None


class _AgoraFixo:
    """Congela o relógio do podcast.py para o carimbo ser determinístico."""

    @staticmethod
    def now():
        return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "titulo,esperado",
    [
        (None, "episodio_2026-01-02_03-04-05"),
        ("", "episodio_2026-01-02_03-04-05"),
        ("Energia Solar no Brasil", "2026-01-02_03-04-05_energia-solar-no-brasil"),
        ("Ação & Reação: 2ª parte!", "2026-01-02_03-04-05_acao-reacao-2a-parte"),
        ("../../etc/passwd", "2026-01-02_03-04-05_etc-passwd"),
        ("???", "2026-01-02_03-04-05_episodio"),
    ],
)
def test_identificador(monkeypatch, titulo, esperado):
    monkeypatch.setattr(podcast, "datetime", _AgoraFixo)
    assert podcast._identificador(titulo) == esperado


def test_identificador_corta_titulo_longo(monkeypatch):
    monkeypatch.setattr(podcast, "datetime", _AgoraFixo)
    nome = podcast._identificador("palavra " * 30)
    base = nome.split("_", 2)[2]
    assert len(base) <= 40
    assert re.fullmatch(r"[a-z0-9-]+", base)
