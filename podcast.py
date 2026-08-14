"""
Cortexus — Estúdio de Podcast (roteiro -> vozes -> programa final).

Três motores, na ordem exata do pipeline:

  1. ROTEIRO  `parse_roteiro()`     texto do LLM   -> lista de `Fala`
  2. VOZES    `MotorTTS`            cada `Fala`    -> um .wav por fala
  3. ESTÚDIO  `montar_programa()`   vários .wav    -> um programa único (+MP3)

Decisões de projeto (o porquê, que é o que se esquece depois):

- POR QUE O MODELO NÃO CUSPIR JSON. O caminho óbvio seria pedir
  `[{"speaker": ..., "text": ...}]` ao LLM. Não fazemos: o qwen 7B roda no
  laço ReAct (`criar_llm` força `supports_function_calling = False`), e um
  roteiro de 15 minutos tem ~80 falas — uma vírgula perdida no meio derruba
  o job inteiro depois de minutos de geração. Em vez disso pedimos o formato
  de linha `Locutor: fala` (que qualquer modelo acerta) e a conversão é
  DETERMINÍSTICA, aqui em Python. É a mesma filosofia de `escolher_saida` e
  `_limpar_artefatos` no main.py: a rede de segurança mora no código, não na
  boa vontade do modelo. O JSON continua existindo — `roteiro_para_json()` o
  produz a partir das `Fala` já validadas, então ele é sempre válido.
  (Se o modelo mandar JSON mesmo assim, `parse_roteiro` aceita — ver
  `_tentar_json`.)

- POR QUE numpy + `wave`, E NÃO pydub. O pydub depende do módulo `audioop`,
  REMOVIDO no Python 3.13 (PEP 594) — funcionaria hoje (o projeto está preso
  ao 3.12 pelo CrewAI) e quebraria no dia da atualização. Ele ainda exigiria
  o binário do FFmpeg no PATH, que seria a primeira dependência não-pip do
  projeto, com instalação diferente no Windows e no Mint. Aqui a mixagem
  inteira é numpy (já instalado, dependência transitiva do chromadb) + o
  módulo `wave` da stdlib. O FFmpeg vira OPCIONAL: só para o encode final em
  MP3, e sem ele o programa sai em WAV normalmente.

- POR QUE PIPER É O MOTOR PADRÃO. Bark/XTTS/Tortoise querem 2-8 GB de VRAM,
  e a RTX 4060 desta máquina já está com ~6 dos 8 GB ocupados pelo
  `qwen2.5-8k`. Os dois não cabem juntos: ou se sequencia (descarregar o
  Ollama, carregar o TTS) ou o TTS transborda para a CPU e um episódio de 10
  minutos leva horas. O Piper roda em ONNX na CPU, mais rápido que tempo
  real, com ~60 MB por voz — zero conflito de VRAM, e o `onnxruntime` já
  está instalado. A troca é feita pelo adapter `MotorTTS`, então migrar para
  XTTS/ElevenLabs depois não toca no resto do pipeline.

- POR QUE ESTE MÓDULO NÃO IMPORTA O main.py. Importar o main puxa o CrewAI
  inteiro (segundos de import). Como as três etapas aqui são 100%
  determinísticas, mantê-las independentes deixa a suíte de testes rápida e
  permite montar/mixar áudio mesmo sem Ollama, sem modelo baixado e sem
  CrewAI funcionando. Quem cola as duas metades é a api.py.

Uso direto pelo terminal (funciona HOJE, sem Ollama e sem baixar voz):

    python podcast.py roteiro.txt --motor tom
    python podcast.py roteiro.txt --motor piper --trilha trilha.wav
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
import wave
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CAMINHOS E CONSTANTES
# ---------------------------------------------------------------------------
# Mesmo princípio cross-platform do main.py: todo caminho via pathlib, nunca
# string com "/" ou "\", para o código ser idêntico no Windows e no Mint.
BASE_DIR = Path(__file__).resolve().parent
PODCAST_DIR = BASE_DIR / "output" / "podcast"
# Onde ficam os .onnx do Piper. Fora do repo? Não: 60 MB por voz é pequeno e
# ter as vozes ao lado do projeto simplifica o dual-boot (a mesma pasta serve
# aos dois SOs). O .gitignore exclui a pasta.
VOZES_DIR = BASE_DIR / "vozes"

# 22050 Hz é a taxa das vozes "medium" do Piper e o padrão de fato de podcast
# falado (o dobro disso não acrescenta nada em voz humana e dobra o arquivo).
# Vozes "low" saem a 16000 Hz e são reamostradas para cá na mixagem.
TAXA_PADRAO = 22050

# Ritmo de conversa: a pausa é maior quando o microfone TROCA de apresentador
# (é o respiro que faz o diálogo soar natural em vez de metralhado).
PAUSA_MESMA_VOZ = 0.28
PAUSA_TROCA_VOZ = 0.55

# Fade de 8 ms coladinho em cada trecho. Sem isso, o corte seco no fim de um
# .wav e o começo do próximo produzem um "clique" audível na emenda.
FADE_EMENDA = 0.008

# Velocidade média de fala em pt-BR, usada só pelo motor de demonstração para
# estimar a duração de uma frase sem sintetizar nada.
CARACTERES_POR_SEGUNDO = 14.5


class ErroPodcast(RuntimeError):
    """Falha de pipeline com mensagem já pronta para mostrar ao usuário."""


# ---------------------------------------------------------------------------
# 1. O ROTEIRO — texto solto do LLM vira uma lista de falas (determinístico)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fala:
    """Uma linha de diálogo já limpa e pronta para virar áudio."""

    locutor: str
    texto: str

    def como_dict(self) -> dict:
        return {"speaker": self.locutor, "text": self.texto}


@dataclass(frozen=True)
class Apresentador:
    """Quem fala, com que voz e com que prosódia.

    `voz` é o nome do modelo Piper (ex.: "pt_BR-faber-medium"), sem extensão.
    `length_scale` acima de 1.0 deixa a fala MAIS LENTA (é um multiplicador de
    duração, não de velocidade) — 1.05 num dos dois apresentadores ajuda a
    diferenciá-los quando o catálogo de vozes pt-BR é curto.
    """

    nome: str
    voz: str
    length_scale: float = 1.0
    volume: float = 1.0
    # Só o motor de demonstração usa: frequência do tom que representa a voz.
    tom_hz: float = 210.0


# Catálogo pt-BR do Piper (são só estas quatro vozes), com os números MEDIDOS
# nesta máquina, não copiados da documentação:
#
#  - "nasal" é a descoberta que decide o padrão: a `edresson-low` foi
#    publicada SEM o til combinante (U+0303) no `phoneme_id_map`. Na prática
#    ela emite "Missing phoneme from id map" e COME a nasalização — "não"
#    sai como "nao", "coração" como "coracao". Num podcast em português isso
#    é inaceitável, então ela não entra no padrão (fica disponível para quem
#    quiser, com o aviso).
#  - "f0" é a frequência fundamental média (o grave/agudo da voz), medida por
#    autocorrelação na mesma frase. Quanto mais distantes as duas vozes, mais
#    fácil o ouvinte separar quem está falando — por isso o par padrão é
#    faber (165 Hz) + cadu (138 Hz), o de maior distância disponível.
#  - as quatro vozes do catálogo são masculinas; não existe voz feminina
#    pt-BR no Piper hoje. Se a dupla precisar de vozes de gêneros diferentes,
#    o caminho é o XTTS-v2 (com as ressalvas de VRAM e licença do README).
VOZES_PT_BR: dict[str, dict] = {
    "pt_BR-faber-medium": {"rotulo": "Faber", "f0": 165, "taxa": 22050, "nasal": True},
    "pt_BR-cadu-medium": {"rotulo": "Cadu", "f0": 138, "taxa": 22050, "nasal": True},
    "pt_BR-jeff-medium": {"rotulo": "Jeff", "f0": 148, "taxa": 22050, "nasal": True},
    "pt_BR-edresson-low": {
        "rotulo": "Edresson",
        "f0": 0,
        "taxa": 16000,
        "nasal": False,
        "aviso": "sem til nasal no modelo: 'não' soa como 'nao'",
    },
}

# Nomes dos apresentadores. NÃO são enfeite: o prompt manda o modelo usar
# esses nomes, e o modelo os coloca DENTRO das falas ("Sim, Host1!"). Com
# "Host1"/"Host2" o episódio saía com os apresentadores se chamando por um
# rótulo de placeholder em voz alta — observado no primeiro teste real. Nomes
# de gente resolvem, e precisam ser MASCULINOS: as quatro vozes pt-BR do
# Piper são masculinas, então "Ana" narrada pelo faber soaria errado.
# ASCII de propósito — o nome vira nome de arquivo dos .wav de cada fala.
NOMES_PADRAO: tuple[str, ...] = ("Rafa", "Bruno")

APRESENTADORES_PADRAO: tuple[Apresentador, ...] = (
    Apresentador(NOMES_PADRAO[0], "pt_BR-faber-medium", length_scale=1.0, tom_hz=180.0),
    Apresentador(NOMES_PADRAO[1], "pt_BR-cadu-medium", length_scale=1.0, tom_hz=250.0),
)


def montar_apresentadores(
    vozes: list[str] | None = None, nomes: list[str] | None = None
) -> tuple[Apresentador, ...]:
    """Constrói a dupla a partir dos nomes de voz escolhidos no estúdio.

    Sem argumentos devolve o padrão. Os tons do motor de demonstração são
    espalhados na faixa da voz humana para que 'tom' continue distinguindo os
    apresentadores mesmo com uma configuração customizada.
    """
    if not vozes:
        return APRESENTADORES_PADRAO
    desconhecidas = [v for v in vozes if v not in VOZES_PT_BR]
    if desconhecidas:
        raise ErroPodcast(
            f"Voz desconhecida: {', '.join(desconhecidas)}. Disponíveis: "
            + ", ".join(sorted(VOZES_PT_BR))
        )
    def nome_de(i: int) -> str:
        if nomes and i < len(nomes) and nomes[i].strip():
            return nomes[i].strip()
        if i < len(NOMES_PADRAO):
            return NOMES_PADRAO[i]
        return f"Convidado{i + 1 - len(NOMES_PADRAO)}"

    return tuple(
        Apresentador(nome=nome_de(i), voz=voz, tom_hz=170.0 + 60.0 * i)
        for i, voz in enumerate(vozes)
    )

# Rótulos que parecem locutor ("Rótulo: texto") mas são METADADO do roteiro.
# Sem esta lista, "Tema: energia solar" viraria um apresentador chamado
# "Tema". É uma heurística de palavras-chave, não um parser de linguagem
# natural: um apresentador que por azar se chame "Nota" cai aqui.
_ROTULOS_NAO_LOCUTOR = frozenset(
    {
        "nota", "notas", "obs", "observacao", "observacoes", "fonte", "fontes",
        "resumo", "titulo", "subtitulo", "tema", "assunto", "objetivo",
        "introducao", "conclusao", "exemplo", "exemplos", "atencao", "dica",
        "aviso", "duracao", "roteiro", "trilha", "musica", "vinheta", "efeito",
        "sfx", "bg", "cena", "bloco", "parte", "capitulo", "secao", "formato",
        "publico", "referencia", "referencias", "link", "links", "url",
        "final answer", "resposta final", "thought", "action", "observation",
    }
)

# "Locutor:" / "**Locutor:**" / "[Locutor]" / "- Locutor —". O rótulo é curto
# de propósito (<= 32 caracteres, sem pontuação final): quanto mais frouxo,
# mais frase comum de prosa seria confundida com fala.
_RE_LOCUTOR = re.compile(
    r"""^\s*
        (?:[-*>•]\s+)?                 # marcador de lista opcional
        (?:\*\*|__|\[|\()?             # decoração de abertura opcional
        \s*
        (?P<locutor>[^\s:\[\]()*_][^:\[\]()*\n]{0,31}?)
        \s*
        (?:\*\*|__|\]|\))?             # decoração de fechamento opcional
        \s*
        (?:[:：]|\s[—–]\s)             # separador: ":" ou travessão ESPAÇADO
        \s*
        (?P<texto>.*)$
    """,
    re.VERBOSE,
)

# Linha que é só rubrica de estúdio: "(risos)", "[a música sobe]", "*pausa*".
_RE_RUBRICA = re.compile(r"^\s*[\[(*_]{1,2}[^\])*_]*[\])*_]{1,2}\s*$")
# Vazamentos do laço ReAct do CrewAI (os mesmos que main.limpar_saida tira).
# São DOIS casos diferentes, e tratá-los igual corrompe o roteiro:
#  - "Final Answer:" costuma vir COLADO no conteúdo bom ("Final Answer: Ana:
#    oi"), então só o prefixo sai e o resto da linha fica.
#  - "Thought:/Action:/Observation:" são raciocínio interno do agente; a linha
#    INTEIRA vai fora. Se apenas tirássemos o rótulo, o texto restante viraria
#    continuação da fala anterior e entraria no episódio.
_RE_PREFIXO_FINAL = re.compile(
    r"^\s*(?:final answer|resposta final)\s*:?\s*", re.IGNORECASE
)
_RE_ANDAIME_REACT = re.compile(
    r"^\s*(?:thought|action(?:\s+input)?|observation)\s*:", re.IGNORECASE
)


def _sem_acento(texto: str) -> str:
    """'Observação' -> 'observacao'. Usado só para bater com a denylist."""
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c)).lower()


def _limpar_para_fala(texto: str) -> str:
    """Tira do texto tudo que é marcação visual e não se pronuncia.

    O TTS lê literalmente o que recebe: deixar "**IA**" faz a voz dizer
    "asterisco asterisco". Links viram só o rótulo — ninguém quer ouvir uma
    URL sendo soletrada no meio de um podcast.
    """
    texto = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", texto)  # [rótulo](url)
    texto = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", texto)      # `código`
    texto = re.sub(r"(\*\*|__|\*|_)", "", texto)              # ênfase markdown
    texto = re.sub(r"\[[^\]]*\]", " ", texto)                 # rubrica [pausa]
    texto = re.sub(r"<[^>]+>", " ", texto)                    # tags soltas
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip(" \t-–—")


def _tentar_json(bruto: str) -> list[Fala] | None:
    """Se o modelo devolveu JSON assim mesmo, aproveita em vez de brigar.

    Aceita tanto `[{...}]` quanto `{"roteiro": [{...}]}` e as duas grafias de
    chave (speaker/text e locutor/texto). Retorna None se não for JSON — aí o
    parser de linha assume.
    """
    trecho = bruto.strip()
    inicio = min(
        (p for p in (trecho.find("["), trecho.find("{")) if p != -1), default=-1
    )
    if inicio == -1:
        return None
    fim = max(trecho.rfind("]"), trecho.rfind("}"))
    if fim <= inicio:
        return None
    try:
        dados = json.loads(trecho[inicio : fim + 1])
    except (ValueError, TypeError):
        return None

    if isinstance(dados, dict):
        for chave in ("roteiro", "script", "falas", "dialogo", "lines"):
            if isinstance(dados.get(chave), list):
                dados = dados[chave]
                break
        else:
            return None
    if not isinstance(dados, list):
        return None

    falas: list[Fala] = []
    for item in dados:
        if not isinstance(item, dict):
            continue
        locutor = item.get("speaker") or item.get("locutor") or item.get("host")
        texto = item.get("text") or item.get("texto") or item.get("fala")
        if not isinstance(locutor, str) or not isinstance(texto, str):
            continue
        limpo = _limpar_para_fala(texto)
        if limpo:
            falas.append(Fala(locutor.strip(), limpo))
    return falas or None


def parse_roteiro(bruto: str) -> list[Fala]:
    """Converte o roteiro cru do LLM numa lista de `Fala`.

    Tolerante de propósito — aceita `Host1: ...`, `**Ana:** ...`,
    `[Bruno] ...`, `- ANA — ...` e JSON. Uma linha SEM rótulo de locutor é
    tratada como continuação da fala anterior (é o que o modelo faz quando
    quebra um parágrafo longo em duas linhas), e não como uma fala órfã.

    Nunca levanta exceção: um roteiro que não casa com nada devolve lista
    vazia, e quem chamou decide o que fazer.
    """
    if not bruto or not bruto.strip():
        return []

    via_json = _tentar_json(bruto)
    if via_json:
        return via_json

    falas: list[Fala] = []
    for linha_bruta in bruto.splitlines():
        if _RE_ANDAIME_REACT.match(linha_bruta):
            continue
        linha = _RE_PREFIXO_FINAL.sub("", linha_bruta).strip()
        if not linha:
            continue
        # Cabeçalhos markdown, réguas e cercas de código não são diálogo.
        if linha.startswith(("#", "```", "~~~")) or re.fullmatch(r"[-=_*]{3,}", linha):
            continue
        if _RE_RUBRICA.match(linha):
            continue

        # "[Ana] texto" -> "Ana: texto". Normalizar aqui é mais simples do que
        # aceitar "]" como separador no regex (o "]" já é decoração lá). A
        # rubrica pura "[música sobe]" não casa: o lookahead exige texto depois.
        linha = re.sub(r"^\s*\[([^\]\n]{1,32})\]\s*(?=[^:\s])", r"\1: ", linha)

        casou = _RE_LOCUTOR.match(linha)
        rotulo = casou.group("locutor").strip() if casou else ""
        # Um rótulo com pontuação de fim de frase é quase sempre prosa que caiu
        # no regex ("Isso é ótimo. Mas: veja" não tem locutor "Isso é ótimo.").
        # O teto de 3 palavras é o que separa "Dr. Silva:" (locutor de verdade)
        # de "Aqui está o roteiro:" (preâmbulo do modelo).
        eh_locutor = bool(
            casou
            and rotulo
            and _sem_acento(rotulo) not in _ROTULOS_NAO_LOCUTOR
            and not re.search(r"[.!?,;]$", rotulo)
            and len(rotulo.split()) <= 3
        )

        if eh_locutor:
            texto = _limpar_para_fala(casou.group("texto"))
            if texto:
                falas.append(Fala(rotulo, texto))
            elif falas:
                # "Ana:" sozinha na linha, com o texto na linha seguinte. Só
                # vale DEPOIS da primeira fala: no topo do arquivo, um rótulo
                # sem texto é cabeçalho, não locutor.
                falas.append(Fala(rotulo, ""))
        elif falas:
            continuacao = _limpar_para_fala(linha)
            if continuacao:
                anterior = falas[-1]
                juntado = f"{anterior.texto} {continuacao}".strip()
                falas[-1] = replace(anterior, texto=juntado)

    return _remover_repeticoes(
        [f for f in falas if f.texto and _em_alfabeto_latino(f.texto)]
    )


# Faixas de escrita que o pt-BR não usa: CJK, hangul, cirílico, árabe, hebraico,
# devanágari, tailandês. Três caracteres bastam para caracterizar deriva de
# idioma — um símbolo solto não.
_RE_FORA_DO_LATINO = re.compile(
    r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ"
    r"฀-๿　-ヿ㐀-鿿가-힯＀-￯]"
)
_MAX_CARACTERES_ESTRANHOS = 3


def _em_alfabeto_latino(texto: str) -> bool:
    """False quando a fala trocou de idioma no meio do episódio.

    Defeito OBSERVADO: no fim de um episódio sobre teoria dos jogos, o qwen2.5
    (modelo de origem chinesa) escreveu as duas últimas falas em MANDARIM. Se
    isso chegasse ao Piper, o fonemizador pt-BR receberia caracteres que não
    existem no mapa da voz e devolveria ruído ou silêncio — e o episódio
    terminaria com dois trechos inúteis.

    É deriva de idioma de modelo pequeno; nenhuma instrução de prompt garante
    que não aconteça. Detectar é trivial, então a barreira mora aqui.
    """
    return len(_RE_FORA_DO_LATINO.findall(texto)) < _MAX_CARACTERES_ESTRANHOS


# Falas curtas iguais são conversa normal ("Isso mesmo!" dito pelos dois);
# um parágrafo inteiro repetido palavra por palavra é o modelo travando.
# 60 caracteres é o corte entre as duas coisas.
_MIN_CARACTERES_REPETICAO = 60


def _remover_repeticoes(falas: list[Fala]) -> list[Fala]:
    """Descarta uma fala longa idêntica à imediatamente anterior.

    Defeito OBSERVADO num teste real com o tema "teoria dos jogos": o qwen 7B
    emitiu o mesmo parágrafo de 230 caracteres duas vezes seguidas, trocando
    só o locutor — no áudio, os dois apresentadores dizem exatamente a mesma
    frase, um atrás do outro. É um sintoma clássico de repetição de modelo
    pequeno, e nenhuma instrução de prompt o elimina de forma confiável. Como
    detectar é trivial e determinístico, a correção mora aqui, no mesmo
    espírito das outras redes de segurança do projeto.

    Compara só com a fala ANTERIOR, não com todas: um podcast pode legitimamente
    retomar uma frase mais adiante ("como eu disse no começo..."), e cortar isso
    seria pior do que o problema.
    """
    limpas: list[Fala] = []
    for fala in falas:
        if limpas and len(fala.texto) >= _MIN_CARACTERES_REPETICAO:
            if fala.texto.casefold() == limpas[-1].texto.casefold():
                continue
        limpas.append(fala)
    return limpas


def roteiro_para_json(falas: list[Fala], *, indent: int = 2) -> str:
    """Serializa as falas JÁ VALIDADAS. Este JSON é sempre bem formado —
    ele nasce das dataclasses, não da caneta do modelo."""
    return json.dumps(
        [f.como_dict() for f in falas], ensure_ascii=False, indent=indent
    )


def mapear_locutores(
    falas: list[Fala], apresentadores: tuple[Apresentador, ...] = APRESENTADORES_PADRAO
) -> list[tuple[Fala, Apresentador]]:
    """Casa cada rótulo do roteiro com um apresentador, por ordem de entrada.

    O modelo pode escrever "Host1/Host2", "Ana/Bruno" ou "ANFITRIÃO/CONVIDADA"
    — tanto faz. O primeiro rótulo que aparece recebe o primeiro apresentador,
    o segundo rótulo o segundo, e assim por diante (dando a volta se houver
    mais rótulos que vozes). Assim o prompt não precisa impor nomes fixos, que
    é justamente o tipo de instrução que um 7B esquece no meio do roteiro.
    """
    if not apresentadores:
        raise ErroPodcast("Nenhum apresentador configurado.")

    ordem: dict[str, Apresentador] = {}
    saida: list[tuple[Fala, Apresentador]] = []
    for fala in falas:
        chave = _sem_acento(fala.locutor).strip()
        if chave not in ordem:
            ordem[chave] = apresentadores[len(ordem) % len(apresentadores)]
        saida.append((fala, ordem[chave]))
    return saida


# Ritmo de podcast falado em pt-BR, MEDIDO nas vozes padrão do Piper: um
# episódio de 891 palavras rendeu 372,99 s de áudio, ou seja ~143 palavras por
# minuto. Um turno de fala a cada ~7 segundos.
PALAVRAS_POR_MINUTO = 145
FALAS_POR_MINUTO = 8

# O modelo entrega MAIS do que se pede. Medido em três episódios de alvo 5 min
# (~39 falas pedidas): 51, 44 e 45 falas — em média +15%. Pedir 0,85 do alvo
# faz a entrega cair em cima dele. É calibração empírica de poucas amostras,
# não uma constante universal: se você trocar de modelo, meça de novo.
# Deliberadamente conservadora — sobrar áudio é melhor que faltar.
CALIBRACAO_FALAS = 0.85


def prompt_roteiro(
    tema: str,
    *,
    minutos: int = 5,
    apresentadores: tuple[Apresentador, ...] = APRESENTADORES_PADRAO,
    tem_contexto: bool = False,
) -> str:
    """Monta o pedido que vai para a crew do main.py.

    `tem_contexto` diz se o usuário anexou material de base. Ele existe por
    causa de um problema OBSERVADO, não teórico: sem essa instrução, o
    Pesquisador (que tem ferramentas de disco) interpretava um tema genérico
    como motivo para vasculhar a pasta do projeto e disparava a
    `DirectorySearchTool` na raiz — que indexa recursivamente e engole o
    `.venv` inteiro. O job ficava mais de dez minutos indexando `.pyc` do
    litellm antes de escrever a primeira linha do roteiro. Um podcast sobre um
    tema não tem nada a ver com os arquivos deste computador, então o prompt
    proíbe explicitamente o disco quando não há material anexado.

    Duas outras sutilezas que parecem detalhe e não são:

    1. O texto começa com "PRODUZA", nunca com "crie/escreva/gere". Esses
       verbos estão em `main._VERBOS_CRIACAO`, e usá-los faria
       `main.escolher_saida` classificar o pedido como Ação de Criação — a
       rede de segurança sairia caçando um "rascunho de código" entre as
       tarefas anteriores. Como um roteiro nunca passa no `_parece_codigo`,
       o resultado prático seria inofensivo, mas o caminho certo é não
       acionar a rede: "produzir" não está na lista.

    2. Não usamos a palavra "script": em português ela significa tanto
       roteiro quanto programa, e é exatamente a ambiguidade que faria o
       Escritor devolver código Python.

    Ainda assim, o prompt é só a PRIMEIRA linha de defesa. A garantia real é
    o `parse_roteiro`, que sobrevive a cerca de markdown, "Final Answer:" e
    rubricas — porque um modelo 7B vai desobedecer o formato de vez em quando.
    """
    nomes = [a.nome for a in apresentadores] or ["Host1", "Host2"]
    palavras = max(120, int(minutos * PALAVRAS_POR_MINUTO))
    turnos = max(4, int(minutos * FALAS_POR_MINUTO))
    exemplo = "\n".join(
        f"{nome}: (uma frase falada, em português, sem marcação)" for nome in nomes[:2]
    )
    fontes = (
        "- Baseie-se no CONTEXTO FORNECIDO pelo usuário; ele é a fonte "
        "primária. NÃO tente reler esse conteúdo do disco.\n"
        if tem_contexto
        else "- NÃO use ferramentas de DISCO ('List files in directory', "
        "'Read a file's content', 'Busca Semântica na Pasta'): este pedido é "
        "sobre um TEMA, não sobre os arquivos deste computador.\n"
        "- OBRIGATÓRIO pesquisar antes: use 'Busca na Web' e depois 'Ler "
        "Página Web' na fonte mais confiável (enciclopédia, documentação, "
        "artigo) para LER o texto. Definições, nomes e exemplos do roteiro "
        "precisam sair da página lida, não da sua memória — é assim que se "
        "evita definir um conceito ao contrário ou citar um exemplo que não "
        "existe.\n"
    )
    return (
        f"PRODUZA O ROTEIRO DE UM PODCAST EM PORTUGUÊS DO BRASIL sobre: {tema}\n\n"
        f"FORMATO OBRIGATÓRIO — uma linha por fala, no padrão 'Nome: texto':\n"
        f"{exemplo}\n\n"
        f"REGRAS:\n"
        f"{fontes}"
        f"- Exatamente {len(nomes)} apresentadores, chamados {' e '.join(nomes)}, "
        f"alternando os turnos.\n"
        f"- Cerca de {turnos} falas, somando aproximadamente {palavras} palavras "
        f"(~{minutos} minutos de áudio).\n"
        f"- Conversa de verdade: um pergunta, o outro responde, discordam, dão "
        f"exemplos concretos. Nada de dois monólogos alternados.\n"
        f"- NUNCA repita uma fala já dita. Cada linha traz informação nova.\n"
        f"- Use o nome do colega com parcimônia (uma ou duas vezes no "
        f"episódio), como gente conversando — não a cada fala.\n"
        f"- Comece com uma abertura curta apresentando o tema e termine com um "
        f"fechamento de uma frase.\n"
        f"- Só texto FALADO. É proibido: títulos, markdown (**, ##, listas), "
        f"rubricas como (risos) ou [trilha sobe], URLs, números de seção e "
        f"qualquer comentário fora das falas.\n"
        f"- Números sempre por extenso ('oito gigabytes', nunca '8 GB') — o "
        f"sintetizador de voz lê literalmente o que estiver na linha.\n"
        f"- Nada de introdução antes da primeira fala."
    )


# ---------------------------------------------------------------------------
# 1b. ESCRITA EM BLOCOS — como arrancar um roteiro do tamanho pedido de um 7B
# ---------------------------------------------------------------------------
# O PROBLEMA MEDIDO: pedir "cerca de 40 falas, ~750 palavras" num passe único
# rendeu 11 falas (tema "teoria dos jogos", alvo de 5 minutos). Modelo pequeno
# não sustenta um pedido longo, e insistir em prompt não conserta; pedir em
# loop traz de volta a repetição literal que o `_remover_repeticoes` teve que
# cortar.
#
# A SOLUÇÃO: trocar UM pedido grande por N pedidos pequenos, orquestrados em
# Python. Primeiro a crew pesquisa e devolve uma PAUTA (blocos + os fatos
# verificados de cada um, com fonte). Depois, um passe curto POR BLOCO pede só
# ~13 falas sobre aquele tópico — tamanho que o 7B cumpre. A costura é
# determinística: concatenar, `parse_roteiro`, `_remover_repeticoes`.
#
# Bônus: a pesquisa roda UMA vez (na pauta) e alimenta todos os blocos, e o
# estúdio consegue mostrar "bloco 3 de 5" em vez de uma barra parada.
def blocos_para(minutos: int) -> int:
    """Quantos blocos para o alvo de duração: um a cada ~2 minutos.

    Piso de 2 (com um bloco só, voltaríamos ao pedido único que falha) e teto
    de 6 (cada bloco é uma chamada ao LLM; passar disso, no 7B, custa mais
    tempo do que o ganho de qualidade)."""
    return max(2, min(6, -(-max(1, minutos) // 2)))


def falas_por_bloco(minutos: int, blocos: int) -> int:
    """Quantas falas pedir em cada bloco, já com a calibração de excesso.

    Fica aqui e não na api.py porque é regra de dimensionamento do roteiro —
    o mesmo motivo de `blocos_para` morar neste módulo — e porque assim dá
    para testá-la sem subir a API.
    """
    alvo = max(1, minutos) * FALAS_POR_MINUTO * CALIBRACAO_FALAS
    return max(4, round(alvo / max(1, blocos)))


def prompt_pauta(
    tema: str,
    *,
    minutos: int = 5,
    blocos: int | None = None,
    tem_contexto: bool = False,
) -> str:
    """Pedido da PAUTA para a crew: os blocos e os fatos de cada um, com fonte.

    Note que aqui pedimos FATOS, não falas. É o único ponto do fluxo em que a
    pesquisa acontece, então o que interessa é a matéria-prima verificada; a
    conversa vem depois, bloco por bloco.
    """
    total = blocos or blocos_para(minutos)
    fontes = (
        "Use o CONTEXTO FORNECIDO pelo usuário como fonte primária.\n"
        if tem_contexto
        else (
            "OBRIGATÓRIO: use 'Busca na Web' e depois 'Ler Página Web' na fonte "
            "mais confiável para LER o texto. Cada fato precisa vir de uma "
            "página que você abriu — não da sua memória. NÃO use ferramentas "
            "de disco: o pedido é sobre um tema, não sobre este computador.\n"
        )
    )
    return (
        f"PRODUZA A PAUTA DE UM EPISÓDIO DE PODCAST sobre: {tema}\n\n"
        f"{fontes}\n"
        f"Divida o assunto em EXATAMENTE {total} blocos, na ordem em que serão "
        f"apresentados (o primeiro abre o tema, o último fecha o episódio).\n\n"
        # Exemplo CONCRETO, não rótulo genérico: com "Título curto do bloco"
        # como modelo, o qwen copiou o placeholder literalmente nas três linhas
        # da pauta em vez de escrever um título.
        f"FORMATO OBRIGATÓRIO — uma linha por bloco, começando com o número. "
        f"Siga o padrão, mas com títulos SEUS, sobre o tema pedido:\n"
        f"1. O que é e de onde veio | definição exata, quem formulou (fonte)\n"
        f"2. Como funciona na prática | mecanismo, um exemplo real (fonte)\n"
        f"3. Onde isso aparece | dois casos concretos (fonte)\n\n"
        f"REGRAS:\n"
        f"- Cada bloco traz de dois a quatro FATOS concretos: definições "
        f"exatas, nomes próprios, números, exemplos reais.\n"
        f"- Definição de conceito: copie o sentido da página lida, não "
        f"parafraseie de cabeça.\n"
        f"- Um exemplo só entra se você o LEU em alguma fonte.\n"
        f"- Nada de falas nem de diálogo aqui — isto é a pauta.\n"
        f"- Nada de introdução antes da linha 1."
    )


# "1. Título | fatos", "- Bloco 2: ...", "2) ...", "## 3 - ..."
_RE_ITEM_PAUTA = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:[-*•]\s*)?(?:bloco\s*)?(\d{1,2})\s*[.)\-:]\s+(?P<corpo>\S.*)$",
    re.IGNORECASE,
)


def dividir_pauta(bruto: str, esperados: int | None = None) -> list[str]:
    """Extrai os blocos da pauta. Tolerante, como o `parse_roteiro`.

    LINHA SEM NÚMERO ANEXA AO BLOCO ANTERIOR — e este detalhe já custou um
    episódio inteiro. Num teste real o modelo devolveu o título numa linha e os
    fatos pesquisados na LINHA SEGUINTE:

        1. O que é e de onde veio | definição exata, quem formulou (fonte)
        A teoria dos jogos é o estudo matemático de... teorema minimax de John
        von Neumann... Theory of Games and Economic Behavior, de 1944.

    A primeira versão desta função só coletava as linhas numeradas, então
    jogava fora exatamente a pesquisa que a crew tinha acabado de fazer, e os
    blocos eram escritos a partir do rótulo do formato. O roteiro saiu falando
    de Pitágoras e de "capturar a rainha do adversário" no xadrez — com a
    pauta correta guardada uma linha abaixo, sem ser usada.

    Se não achar itens numerados, cai para os parágrafos separados por linha
    em branco. Se nem isso, devolve a pauta inteira como um bloco único — aí a
    escrita degrada para o comportamento antigo, o que é pior mas não quebra.
    """
    itens: list[str] = []
    for linha_bruta in (bruto or "").splitlines():
        linha = linha_bruta.strip()
        if not linha:
            continue
        if casou := _RE_ITEM_PAUTA.match(linha):
            itens.append(casou.group("corpo").strip())
        elif itens:
            itens[-1] = f"{itens[-1]} {linha}".strip()

    if len(itens) < 2:
        itens = [
            " ".join(p.split())
            for p in re.split(r"\n\s*\n", (bruto or "").strip())
            if len(p.split()) >= 4
        ]
    if not itens:
        return [(bruto or "").strip() or "o tema do episódio"]
    return itens[:esperados] if esperados else itens


def prompt_bloco(
    tema: str,
    bloco: str,
    *,
    indice: int,
    total: int,
    apresentadores: tuple[Apresentador, ...] = APRESENTADORES_PADRAO,
    falas: int = 12,
    ultimas_falas: str = "",
) -> str:
    """Pedido de UM bloco: curto, focado, com o contexto mínimo de continuidade.

    `ultimas_falas` são as duas últimas falas já escritas. Sem elas, cada bloco
    recomeçaria com uma saudação de abertura ("Olá, bem-vindos!") no meio do
    episódio — e o modelo tenderia a repetir o que acabou de ser dito.
    """
    nomes = [a.nome for a in apresentadores] or list(NOMES_PADRAO)
    palavras = max(60, int(falas * PALAVRAS_POR_MINUTO / FALAS_POR_MINUTO))

    # A frase de gancho é o defeito de emenda mais visível: cada bloco tendia a
    # fechar com "ainda temos muito mais para explorar", e isso reaparecia a
    # cada costura. Não é repetição literal, então o `_remover_repeticoes` não
    # pega — tem que ser proibido na origem, e só o último bloco fecha.
    sem_gancho = (
        " NÃO encerre o bloco: nada de frase de gancho, de anunciar o que vem "
        "a seguir ('ainda temos muito a explorar', 'no próximo bloco...') nem "
        "de resumir o que foi dito. Termine no meio do assunto, numa fala "
        "comum, como se a conversa seguisse."
    )
    if indice == 1:
        posicao = (
            "Este é o PRIMEIRO bloco: abra o episódio em uma fala curta "
            "(cumprimento + o tema) e entre no assunto." + sem_gancho
        )
    elif indice == total:
        posicao = (
            "Este é o ÚLTIMO bloco: desenvolva o tópico e encerre com UMA fala "
            "curta de despedida. Só este bloco fecha o episódio."
        )
    else:
        posicao = (
            "Este é um bloco do MEIO do episódio: entre direto no assunto. "
            "É proibido cumprimentar, se apresentar ou se despedir." + sem_gancho
        )

    continuidade = (
        f"\nO episódio vinha assim (NÃO repita estas falas, continue de onde "
        f"pararam):\n{ultimas_falas.strip()}\n"
        if ultimas_falas.strip()
        else ""
    )

    return (
        f"Você escreve o bloco {indice} de {total} de um podcast em português "
        f"do Brasil sobre: {tema}\n\n"
        f"ASSUNTO DESTE BLOCO (use estes fatos, não invente outros):\n"
        f"{bloco.strip()}\n"
        f"{continuidade}\n"
        f"{posicao}\n\n"
        f"FORMATO OBRIGATÓRIO — uma linha por fala, no padrão 'Nome: texto':\n"
        f"{nomes[0]}: (fala)\n"
        f"{nomes[1 % len(nomes)]}: (fala)\n\n"
        f"REGRAS:\n"
        f"- Exatamente {len(nomes)} apresentadores, {' e '.join(nomes)}, "
        f"alternando os turnos a cada linha.\n"
        f"- {falas} falas, somando cerca de {palavras} palavras.\n"
        f"- Cada fala tem uma ou duas frases. Fala longa vira monólogo e "
        f"soa a texto lido em voz alta.\n"
        f"- Conversa de verdade: um pergunta, o outro responde e complementa.\n"
        f"- NUNCA repita uma fala. Cada linha traz informação nova.\n"
        f"- Só texto FALADO: nada de títulos, markdown, rubricas como (risos), "
        f"URLs ou números de bloco.\n"
        f"- Números por extenso ('oito gigabytes', nunca '8 GB').\n"
        f"- Responda SOMENTE com as linhas de fala, nada antes nem depois."
    )


def ultimas_falas_como_texto(falas: list[Fala], quantas: int = 2) -> str:
    """Serializa o fim do roteiro para alimentar a continuidade do bloco seguinte."""
    return "\n".join(f"{f.locutor}: {f.texto}" for f in falas[-quantas:])


# ---------------------------------------------------------------------------
# 2. AS VOZES — adapter de TTS (Piper hoje; XTTS/ElevenLabs depois sem tocar
#    no resto do pipeline, exatamente como criar_llm() isola o LLM no main.py)
# ---------------------------------------------------------------------------
class MotorTTS:
    """Interface mínima: transformar um texto num .wav mono."""

    nome = "base"

    def sintetizar(self, texto: str, apresentador: Apresentador, destino: Path) -> Path:
        raise NotImplementedError

    def verificar(self, apresentadores: tuple[Apresentador, ...]) -> None:
        """Falha CEDO (antes de gerar a primeira fala) se faltar algo.

        Descobrir que a voz não está baixada depois de 80 sínteses seria a
        pior hora possível — por isso o pipeline chama isto no começo.
        """
        return None


class MotorTom(MotorTTS):
    """Motor de DEMONSTRAÇÃO: sintetiza um tom, não voz.

    Existe por um motivo prático — enquanto o SSD novo não chega e nenhuma voz
    está baixada, ele permite exercitar o pipeline inteiro (parser, pausas,
    crossfade, trilha, equalização, MP3, player do estúdio) e OUVIR se a
    montagem ficou certa: cada apresentador vira um tom de frequência
    diferente, com a duração proporcional ao tamanho da frase. Dá para
    escutar a alternância, os respiros e a música abaixando sob a fala.
    """

    nome = "tom"

    def __init__(self, taxa: int = TAXA_PADRAO) -> None:
        self.taxa = taxa

    def sintetizar(self, texto: str, apresentador: Apresentador, destino: Path) -> Path:
        duracao = max(0.35, len(texto) / CARACTERES_POR_SEGUNDO)
        t = np.arange(int(duracao * self.taxa), dtype=np.float32) / self.taxa
        base = apresentador.tom_hz * apresentador.length_scale
        # Duas harmônicas + um leve vibrato: fica claramente audível como
        # "alguém falando" sem soar como um alarme.
        onda = (
            0.60 * np.sin(2 * np.pi * base * t + 0.8 * np.sin(2 * np.pi * 4.5 * t))
            + 0.22 * np.sin(2 * np.pi * 2 * base * t)
            + 0.08 * np.sin(2 * np.pi * 3 * base * t)
        )
        # Envelope de sílabas (~4,2 por segundo) para não virar um tom contínuo.
        silabas = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 4.2 * t))
        sinal = (onda * silabas * 0.42 * apresentador.volume).astype(np.float32)
        escrever_wav(destino, aplicar_fade(sinal, 0.02, 0.05, self.taxa), self.taxa)
        return destino


class MotorPiper(MotorTTS):
    """Piper (ONNX, CPU) — o motor de produção.

    As vozes são carregadas UMA vez e ficam em cache: um roteiro de 15 minutos
    tem ~80 falas, e recarregar o .onnx a cada fala dominaria completamente o
    tempo de execução.
    """

    nome = "piper"

    def __init__(
        self,
        dir_vozes: Path | None = None,
        *,
        taxa: int = TAXA_PADRAO,
        usar_cuda: bool = False,
    ) -> None:
        self.dir_vozes = Path(dir_vozes) if dir_vozes else VOZES_DIR
        self.taxa = taxa
        # usar_cuda exige onnxruntime-gpu E disputa VRAM com o Ollama — só
        # ligue se o modelo de chat estiver descarregado (ver docstring do módulo).
        self.usar_cuda = usar_cuda
        self._cache: dict[str, object] = {}

    def caminho_voz(self, voz: str) -> Path:
        return self.dir_vozes / f"{voz}.onnx"

    def verificar(self, apresentadores: tuple[Apresentador, ...]) -> None:
        try:
            import piper  # noqa: F401
        except ImportError as erro:
            raise ErroPodcast(
                "O motor 'piper' não está instalado. Rode:  pip install piper-tts\n"
                "(ou use o motor 'tom' para testar a montagem sem baixar nada)."
            ) from erro

        faltando = sorted(
            {a.voz for a in apresentadores if not self.caminho_voz(a.voz).exists()}
        )
        if faltando:
            comandos = "\n".join(
                f"  python -m piper.download_voices {v} --download-dir "
                f'"{self.dir_vozes}"'
                for v in faltando
            )
            raise ErroPodcast(
                "Voz(es) do Piper não encontrada(s) em "
                f"'{self.dir_vozes}': {', '.join(faltando)}.\n"
                f"Baixe com:\n{comandos}"
            )

    def _voz(self, apresentador: Apresentador):
        if apresentador.voz not in self._cache:
            from piper import PiperVoice

            self._cache[apresentador.voz] = PiperVoice.load(
                str(self.caminho_voz(apresentador.voz)), use_cuda=self.usar_cuda
            )
        return self._cache[apresentador.voz]

    def sintetizar(self, texto: str, apresentador: Apresentador, destino: Path) -> Path:
        voz = self._voz(apresentador)
        config = None
        try:  # SynthesisConfig existe do piper-tts 1.3 em diante
            from piper import SynthesisConfig

            config = SynthesisConfig(
                length_scale=apresentador.length_scale,
                volume=apresentador.volume,
                normalize_audio=True,
            )
        except ImportError:
            pass

        destino.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destino), "wb") as arquivo:
            # A assinatura mudou entre versões do piper-tts; a instalada pode
            # não aceitar syn_config. Cair para a chamada simples é melhor do
            # que exigir uma versão exata num projeto que roda em dois SOs.
            try:
                voz.synthesize_wav(texto, arquivo, syn_config=config)
            except TypeError:
                voz.synthesize_wav(texto, arquivo)
        return destino


MOTORES: dict[str, type[MotorTTS]] = {"piper": MotorPiper, "tom": MotorTom}


def criar_motor(nome: str = "piper", **kwargs) -> MotorTTS:
    """Fábrica única do motor de voz — o único ponto a mexer para trocar de
    TTS, espelhando o papel de `criar_llm()` no main.py."""
    classe = MOTORES.get((nome or "").strip().lower())
    if classe is None:
        raise ErroPodcast(
            f"Motor de TTS desconhecido: '{nome}'. Disponíveis: "
            + ", ".join(sorted(MOTORES))
        )
    return classe(**kwargs)


# ---------------------------------------------------------------------------
# 3. O ESTÚDIO — leitura/escrita de WAV e pós-produção (numpy puro)
# ---------------------------------------------------------------------------
# Convenção interna: todo áudio circula como np.float32 mono em [-1, 1]. A
# conversão de/para int16 acontece só nas bordas (ler_wav / escrever_wav).
def ler_wav(caminho: Path) -> tuple[np.ndarray, int]:
    """Lê um WAV PCM e devolve (sinal float32 mono, taxa).

    Aceita 8/16/32 bits e estéreo (que é rebaixado para mono pela média dos
    canais) porque a trilha de fundo costuma vir de fora, em qualquer formato.
    """
    caminho = Path(caminho)
    try:
        with wave.open(str(caminho), "rb") as arquivo:
            canais = arquivo.getnchannels()
            largura = arquivo.getsampwidth()
            taxa = arquivo.getframerate()
            cru = arquivo.readframes(arquivo.getnframes())
    except (wave.Error, EOFError, OSError) as erro:
        raise ErroPodcast(f"Não consegui ler o WAV '{caminho.name}': {erro}") from erro

    if largura == 1:  # PCM de 8 bits é SEM sinal (0..255), centrado em 128
        dados = (np.frombuffer(cru, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif largura == 2:
        dados = np.frombuffer(cru, dtype="<i2").astype(np.float32) / 32768.0
    elif largura == 4:
        dados = np.frombuffer(cru, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ErroPodcast(
            f"WAV de {largura * 8} bits não suportado ('{caminho.name}'). "
            "Converta para PCM 16 bits."
        )

    if canais > 1:
        # Descarta um frame incompleto no fim antes de remodelar (arquivo
        # truncado é comum em áudio baixado pela metade).
        util = (dados.size // canais) * canais
        dados = dados[:util].reshape(-1, canais).mean(axis=1)
    return np.ascontiguousarray(dados, dtype=np.float32), taxa


def escrever_wav(caminho: Path, sinal: np.ndarray, taxa: int = TAXA_PADRAO) -> Path:
    """Grava float32 mono como WAV PCM 16 bits (o formato que o Piper produz
    e que todo player — inclusive o <audio> do HTML — toca sem plugin)."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    # O clip é obrigatório: somar fala + trilha pode passar de 1.0, e o
    # int16 daria a volta (wrap) virando estalo em vez de saturação.
    limitado = np.clip(np.asarray(sinal, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(caminho), "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(int(taxa))
        arquivo.writeframes((limitado * 32767.0).astype("<i2").tobytes())
    return caminho


def reamostrar(sinal: np.ndarray, origem: int, destino: int) -> np.ndarray:
    """Reamostragem por interpolação linear.

    Não é um reamostrador com filtro anti-aliasing de verdade, e para voz
    falada indo de 16 kHz para 22,05 kHz (o caso real: a voz `edresson-low`
    entrando num programa a 22050) a diferença é inaudível. Trocar isso por
    uma solução "correta" custaria uma dependência nova (scipy/soxr) para
    ganhar nada perceptível aqui.
    """
    if origem == destino or sinal.size == 0:
        return np.asarray(sinal, dtype=np.float32)
    novo_tamanho = int(round(sinal.size * destino / float(origem)))
    if novo_tamanho <= 1:
        return np.zeros(0, dtype=np.float32)
    posicoes = np.linspace(0.0, sinal.size - 1.0, novo_tamanho)
    return np.interp(posicoes, np.arange(sinal.size), sinal).astype(np.float32)


def silencio(duracao: float, taxa: int = TAXA_PADRAO) -> np.ndarray:
    return np.zeros(max(0, int(round(duracao * taxa))), dtype=np.float32)


def aplicar_fade(
    sinal: np.ndarray, entrada: float, saida: float, taxa: int = TAXA_PADRAO
) -> np.ndarray:
    """Rampa linear no início e no fim (em segundos), sem alterar o miolo."""
    resultado = np.array(sinal, dtype=np.float32, copy=True)
    n = resultado.size
    n_entrada = min(int(entrada * taxa), n)
    # A saída nunca invade a região da entrada (num trecho curtíssimo as duas
    # rampas se sobreporiam e o miolo sumiria).
    n_saida = min(int(saida * taxa), n - n_entrada)
    if n_entrada > 0:
        resultado[:n_entrada] *= np.linspace(0.0, 1.0, n_entrada, dtype=np.float32)
    if n_saida > 0:
        resultado[-n_saida:] *= np.linspace(1.0, 0.0, n_saida, dtype=np.float32)
    return resultado


def normalizar(sinal: np.ndarray, pico: float = 0.89) -> np.ndarray:
    """Normalização por PICO (não por loudness/LUFS).

    Escolha consciente: medir LUFS de verdade (ITU-R BS.1770) exigiria uma
    dependência a mais, e para um programa de voz + trilha o pico em -1 dBFS
    já entrega um arquivo com volume consistente e sem clipping.
    """
    sinal = np.asarray(sinal, dtype=np.float32)
    if sinal.size == 0:
        return sinal
    maximo = float(np.max(np.abs(sinal)))
    if maximo <= 1e-9:
        return sinal
    return (sinal * (pico / maximo)).astype(np.float32)


def equalizar(
    sinal: np.ndarray,
    taxa: int = TAXA_PADRAO,
    *,
    corte_grave: float = 85.0,
    presenca_db: float = 2.5,
) -> np.ndarray:
    """Equalizador de voz: corta o grave e realça a presença.

    Feito no domínio da frequência (`rfft`) porque um biquad em Python puro
    seria um laço sobre milhões de amostras — inviável — e trazer scipy só
    para `lfilter` seria uma dependência pesada para duas curvas simples.
    A EQ é aplicada POR FALA (trechos de poucos segundos), então cada FFT é
    barata e o padding evita o vazamento circular nas bordas.

    - Passa-alta ~12 dB/oitava abaixo de 85 Hz: tira ronco de fundo e o
      "plosivo" do P/B, que em voz falada só suja a mixagem.
    - Sino largo em 3 kHz: é a faixa da inteligibilidade das consoantes — é o
      que faz a fala "sair" de baixo da trilha sem precisar subir o volume.
    """
    sinal = np.asarray(sinal, dtype=np.float32)
    if sinal.size < 64:
        return sinal

    preenchimento = min(int(taxa * 0.05), sinal.size)
    corpo = np.concatenate(
        [
            np.zeros(preenchimento, dtype=np.float32),
            sinal,
            np.zeros(preenchimento, dtype=np.float32),
        ]
    )
    espectro = np.fft.rfft(corpo)
    frequencias = np.fft.rfftfreq(corpo.size, 1.0 / taxa)
    seguras = np.maximum(frequencias, 1e-6)  # evita divisão por zero no DC

    ganho = 1.0 / np.sqrt(1.0 + (corte_grave / seguras) ** 4)
    if presenca_db:
        sino = np.exp(-0.5 * (np.log2(seguras / 3000.0) / 0.9) ** 2)
        ganho = ganho * 10.0 ** (presenca_db * sino / 20.0)

    filtrado = np.fft.irfft(espectro * ganho, n=corpo.size)
    return filtrado[preenchimento : preenchimento + sinal.size].astype(np.float32)


def _preparar_trilha(
    caminho: Path, amostras: int, taxa: int
) -> np.ndarray:
    """Lê a trilha, reamostra e repete (loop) até cobrir o programa inteiro."""
    trilha, taxa_trilha = ler_wav(caminho)
    trilha = reamostrar(trilha, taxa_trilha, taxa)
    if trilha.size == 0:
        return np.zeros(amostras, dtype=np.float32)
    if trilha.size < amostras:
        repeticoes = int(np.ceil(amostras / trilha.size))
        trilha = np.tile(trilha, repeticoes)
    return trilha[:amostras].astype(np.float32)


def montar_programa(
    segmentos: list[Path],
    destino: Path,
    *,
    locutores: list[str] | None = None,
    taxa: int = TAXA_PADRAO,
    pausa: float = PAUSA_MESMA_VOZ,
    pausa_troca: float = PAUSA_TROCA_VOZ,
    trilha: Path | None = None,
    volume_trilha: float = 0.55,
    volume_trilha_fundo: float = 0.11,
    intro: float = 4.0,
    outro: float = 4.0,
    crossfade: float = 1.5,
    equalizar_vozes: bool = True,
) -> dict:
    """Junta os .wav das falas num programa único, com acabamento.

    O que "acabamento" quer dizer aqui, na ordem em que acontece:
      1. tudo reamostrado para a mesma taxa (as vozes do Piper divergem);
      2. EQ de voz por fala (`equalizar`);
      3. fade de 8 ms em cada emenda, para não estalar;
      4. pausa entre falas — MAIOR quando troca de apresentador;
      5. trilha em loop, entrando alta na abertura e CAINDO por crossfade
         quando a fala começa (o "duck" clássico), subindo de novo no fim;
      6. normalização de pico do programa inteiro.

    Devolve {"arquivo", "duracao", "taxa"}.
    """
    if not segmentos:
        raise ErroPodcast("Nenhum trecho de áudio para montar o programa.")

    partes: list[np.ndarray] = []
    anterior: str | None = None
    for indice, caminho in enumerate(segmentos):
        sinal, taxa_origem = ler_wav(caminho)
        sinal = reamostrar(sinal, taxa_origem, taxa)
        if sinal.size == 0:
            continue
        if equalizar_vozes:
            sinal = equalizar(sinal, taxa)
        sinal = aplicar_fade(sinal, FADE_EMENDA, FADE_EMENDA, taxa)

        if partes:
            atual = locutores[indice] if locutores and indice < len(locutores) else None
            trocou = atual is not None and anterior is not None and atual != anterior
            partes.append(silencio(pausa_troca if trocou else pausa, taxa))
        partes.append(sinal)
        anterior = locutores[indice] if locutores and indice < len(locutores) else None

    if not partes:
        raise ErroPodcast("Os trechos de áudio estavam todos vazios.")

    fala = normalizar(np.concatenate(partes), pico=0.80)

    if trilha is None:
        programa = normalizar(fala, pico=0.89)
    else:
        inicio_fala = max(0.0, intro)
        n_intro = int(inicio_fala * taxa)
        n_outro = int(max(0.0, outro) * taxa)
        total = n_intro + fala.size + n_outro

        # Envelope da trilha: alta na abertura, ducked sob a fala, alta de
        # novo no encerramento. As rampas COMEÇAM antes da fala entrar, então
        # a música já está baixa quando a primeira palavra soa — é isso que
        # faz a transição soar profissional em vez de abrupta.
        envelope = np.full(total, volume_trilha_fundo, dtype=np.float32)
        n_cross = max(1, int(max(0.05, crossfade) * taxa))
        abertura = max(0, n_intro - n_cross)
        envelope[:abertura] = volume_trilha
        descida = min(n_cross, max(0, n_intro - abertura))
        if descida > 0:
            envelope[abertura : abertura + descida] = np.linspace(
                volume_trilha, volume_trilha_fundo, descida, dtype=np.float32
            )
        if n_outro > 0:
            fim_fala = n_intro + fala.size
            subida = min(n_cross, n_outro)
            envelope[fim_fala : fim_fala + subida] = np.linspace(
                volume_trilha_fundo, volume_trilha, subida, dtype=np.float32
            )
            envelope[fim_fala + subida :] = volume_trilha

        cama = _preparar_trilha(trilha, total, taxa) * envelope
        cama = aplicar_fade(cama, min(1.5, intro or 1.5), min(2.5, outro or 2.5), taxa)

        mistura = np.zeros(total, dtype=np.float32)
        mistura[n_intro : n_intro + fala.size] = fala
        programa = normalizar(mistura + cama, pico=0.89)

    escrever_wav(destino, programa, taxa)
    return {
        "arquivo": Path(destino),
        "duracao": round(programa.size / float(taxa), 2),
        "taxa": taxa,
    }


def exportar_mp3(
    wav: Path, destino: Path | None = None, *, bitrate: str = "192k"
) -> Path | None:
    """Converte para MP3 via FFmpeg, se ele existir no PATH.

    Devolve None quando o FFmpeg não está instalado — de propósito: o WAV já
    é um entregável válido, e o projeto não quebra por causa de uma
    dependência de sistema opcional. Instalação, se quiser o MP3:
      Windows:  winget install Gyan.FFmpeg
      Mint:     sudo apt install ffmpeg
    """
    binario = shutil.which("ffmpeg")
    if not binario:
        return None
    wav = Path(wav)
    destino = Path(destino) if destino else wav.with_suffix(".mp3")
    try:
        resultado = subprocess.run(
            [
                binario, "-y", "-loglevel", "error",
                "-i", str(wav),
                "-codec:a", "libmp3lame", "-b:a", bitrate, "-ac", "1",
                str(destino),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            shell=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return destino if resultado.returncode == 0 and destino.exists() else None


# ---------------------------------------------------------------------------
# 4. ORQUESTRAÇÃO — as três etapas numa chamada só
# ---------------------------------------------------------------------------
def _identificador(titulo: str | None) -> str:
    """Nome de pasta seguro nos dois SOs, com carimbo de tempo na frente para
    os episódios ficarem em ordem cronológica no explorador de arquivos."""
    carimbo = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not titulo:
        return f"episodio_{carimbo}"
    base = _sem_acento(titulo)
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")[:40] or "episodio"
    return f"{carimbo}_{base}"


def gerar_podcast(
    roteiro: str,
    *,
    titulo: str | None = None,
    motor: str | MotorTTS = "piper",
    apresentadores: tuple[Apresentador, ...] = APRESENTADORES_PADRAO,
    trilha: Path | None = None,
    destino: Path | None = None,
    manter_falas: bool = False,
    **opcoes_estudio,
) -> dict:
    """Roteiro cru -> episódio pronto. É o ponto de entrada do estúdio.

    `manter_falas=False` apaga os .wav individuais no fim (um episódio de 15
    minutos deixa ~80 arquivos de rascunho; guardá-los só faz sentido para
    depurar uma fala específica).

    Devolve um dicionário com os caminhos gerados, as falas e a duração —
    tudo serializável em JSON, para a API repassar direto ao estúdio HTML.
    """
    falas = parse_roteiro(roteiro)
    if not falas:
        raise ErroPodcast(
            "Não encontrei nenhuma fala no roteiro. O formato esperado é uma "
            "linha por fala, no padrão 'Locutor: texto'."
        )

    dupla = mapear_locutores(falas, apresentadores)
    usados = tuple(dict.fromkeys(a for _, a in dupla))

    motor_tts = criar_motor(motor) if isinstance(motor, str) else motor
    motor_tts.verificar(usados)  # falha ANTES de sintetizar a primeira fala

    pasta = Path(destino) if destino else PODCAST_DIR / _identificador(titulo)
    pasta_falas = pasta / "falas"
    pasta_falas.mkdir(parents=True, exist_ok=True)

    segmentos: list[Path] = []
    for indice, (fala, apresentador) in enumerate(dupla, start=1):
        arquivo = pasta_falas / f"{indice:03d}_{apresentador.nome}.wav"
        motor_tts.sintetizar(fala.texto, apresentador, arquivo)
        segmentos.append(arquivo)

    programa = montar_programa(
        segmentos,
        pasta / "programa.wav",
        locutores=[a.nome for _, a in dupla],
        trilha=Path(trilha) if trilha else None,
        **opcoes_estudio,
    )

    roteiro_json = pasta / "roteiro.json"
    roteiro_json.write_text(roteiro_para_json(falas), encoding="utf-8")

    mp3 = exportar_mp3(programa["arquivo"])

    if not manter_falas:
        for arquivo in segmentos:
            arquivo.unlink(missing_ok=True)
        if not any(pasta_falas.iterdir()):
            pasta_falas.rmdir()

    return {
        "titulo": titulo or "Episódio Cortexus",
        "pasta": str(pasta),
        "arquivo_wav": str(programa["arquivo"]),
        "arquivo_mp3": str(mp3) if mp3 else None,
        "arquivo_roteiro": str(roteiro_json),
        "duracao": programa["duracao"],
        "motor": motor_tts.nome,
        "falas": [
            {"speaker": f.locutor, "voz": a.nome, "text": f.texto} for f, a in dupla
        ],
        "total_falas": len(dupla),
        # Aviso, não erro: sem FFmpeg o episódio existe em WAV do mesmo jeito.
        "aviso_mp3": None if mp3 else "FFmpeg não encontrado — episódio só em WAV.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cortexus — transforma um roteiro em episódio de podcast."
    )
    parser.add_argument("roteiro", help="arquivo de texto com o roteiro (ou '-' p/ stdin)")
    parser.add_argument("--motor", default="tom", choices=sorted(MOTORES),
                        help="motor de voz ('tom' = demonstração, sem baixar nada)")
    parser.add_argument("--titulo", default=None)
    parser.add_argument("--trilha", default=None, help="WAV de trilha de fundo")
    parser.add_argument("--manter-falas", action="store_true",
                        help="não apaga os .wav de cada fala")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    bruto = (
        sys.stdin.read()
        if args.roteiro == "-"
        else Path(args.roteiro).read_text(encoding="utf-8")
    )

    try:
        resultado = gerar_podcast(
            bruto,
            titulo=args.titulo,
            motor=args.motor,
            trilha=Path(args.trilha) if args.trilha else None,
            manter_falas=args.manter_falas,
        )
    except ErroPodcast as erro:
        print(f"\n[X] {erro}\n")
        return 1

    print(f"\n[OK] {resultado['total_falas']} falas | {resultado['duracao']}s")
    print(f"     WAV: {resultado['arquivo_wav']}")
    print(f"     MP3: {resultado['arquivo_mp3'] or '(FFmpeg ausente)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
