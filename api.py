"""Cortexus — API local ASSÍNCRONA (Nível 4) + fachada OpenAI-compatível.

Duas faces para o mesmo cérebro:
  - Face "fire-and-forget" (n8n): POST /agencia responde na hora com um
    job_id (a equipe roda em background) e dispara um webhook ao terminar;
    GET /status/{id} acompanha por polling.
  - Face "editor de código": /v1/chat/completions imita a API da OpenAI,
    então Continue.dev / Cline se conectam apontando para localhost:8000/v1
    e conversam com a agência pela barra lateral do VS Code.

Não altera nada em main.py — só importa o que já existe lá.

Uso:
  python api.py                         # http://127.0.0.1:8000, sem reload
  python api.py --reload                # recarrega ao editar o código (dev)
  python api.py --host 0.0.0.0          # expõe na rede local (ver aviso abaixo)
  (equivalente) uvicorn api:app --host 127.0.0.1 --port 8000

Configuração do servidor (CLI vence variável de ambiente, que vence o padrão):
  --host   / NCX_HOST    padrão 127.0.0.1 (só esta máquina)
  --port   / NCX_PORT    padrão 8000
  --reload / NCX_RELOAD  padrão desligado (1/true/sim/on liga)

SEGURANÇA: não há autenticação, e POST /agencia aciona um agente com acesso
ao terminal. Por isso o padrão é loopback; um host fora dele (0.0.0.0, IP da
LAN) entrega esse terminal a qualquer um na rede e o servidor avisa no boot.

Endpoints:
  POST /agencia                 {"tema": "...", "webhook_url": "http://... (opcional)"}
  GET  /status/{job_id}         situação e resultado do job (agência OU podcast)
  POST /podcast                 {"tema": ...} ou {"roteiro": ...} -> episódio
  GET  /podcast/vozes           catálogo pt-BR e quais vozes já estão em disco
  GET  /podcast/{job_id}/audio  o episódio pronto (MP3, ou WAV se faltar FFmpeg)
  GET  /estudio                 o painel HTML do Estúdio de Podcast
  POST /v1/chat/completions     fachada OpenAI (streaming SSE) — para editores
  GET  /v1/models               lista o modelo "cortexus-crew"
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import podcast as estudio
from main import (
    OUTPUT_DIR,
    TEMA_PADRAO,
    classificar_saida,
    conteudo_para_exibicao,
    criar_llm,
    diagnostico_ollama,
    escolher_saida,
    limpar_saida,
    montar_equipe,
    salvar_relatorio,
)

app = FastAPI(title="API Cortexus")

# Registro de jobs em memória: suficiente para um servidor local de usuário
# único. Se o servidor reiniciar (ou o reload disparar), os jobs em andamento
# se perdem — por isso o reload_excludes lá no __main__.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS_GUARDADOS = 100


class PedidoAgencia(BaseModel):
    tema: str
    webhook_url: str | None = None
    usar_memoria: bool = False  # memória de longo prazo (default off; ver montar_equipe)
    contexto_extra: str = ""     # texto de arquivo(s) anexado(s) no painel (upload)
    usar_gerente: bool = False   # processo hierárquico com Gerente (default off)


class PedidoPodcast(BaseModel):
    """Pedido do Estúdio de Podcast.

    Dois caminhos de entrada, e é o que torna o estúdio utilizável HOJE:
      - `roteiro` preenchido: pula a crew e vai direto para as vozes. Não
        exige Ollama nem modelo baixado.
      - `tema` preenchido: a crew do main.py escreve o roteiro antes (aí sim
        exige Ollama no ar).
    Se os dois vierem, `roteiro` vence — o texto que o usuário escreveu à mão
    nunca deve ser descartado em favor de uma geração.
    """

    tema: str = ""
    roteiro: str = ""
    titulo: str | None = None
    minutos: int = 5
    # "blocos" (padrão): a crew pesquisa e devolve uma pauta, depois um passe
    # curto por bloco. "unico": um só pedido para a crew — mais rápido, mas
    # entrega bem menos falas do que o alvo (medido: 11 de ~40).
    modo: str = "blocos"
    motor: str = "piper"            # "piper" (voz) | "tom" (demonstração)
    vozes: list[str] | None = None  # nomes do catálogo VOZES_PT_BR
    trilha: str | None = None       # WAV de fundo, caminho DENTRO do projeto
    contexto_extra: str = ""
    usar_memoria: bool = False
    webhook_url: str | None = None


class NexusAssist(BaseModel):
    """Pedido do 'microsserviço' Nexus (consumido por outro orquestrador via
    nexus_bridge.py). Síncrono e sempre no modo SEQUENCIAL — nunca hierárquico,
    para não criar delegação-de-agente-para-agente."""
    tarefa: str
    tipo: str = "pesquisa"  # "pesquisa" | "documentacao"


def _atualizar_job(job_id: str, **campos) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(campos)


def _disparar_webhook(url: str, payload: dict) -> str | None:
    """Entrega o resultado no webhook do chamador. Retorna o erro (str) ou None.

    Ordem dos excepts importa: HTTPError (4xx/5xx — o urllib LEVANTA para
    esses códigos, não devolve resposta) é subclasse de URLError (DNS,
    conexão recusada, timeout), e o Exception final é o para-quedas — a
    entrega do webhook nunca pode matar a thread do job.
    """
    try:
        dados = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        requisicao = urllib.request.Request(
            url,
            data=dados,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(requisicao, timeout=15):
            return None  # urlopen só retorna sem exceção em 2xx/3xx
    except urllib.error.HTTPError as erro:
        return f"webhook respondeu HTTP {erro.code} ({erro.reason})"
    except urllib.error.URLError as erro:
        return f"não alcancei o webhook: {erro.reason}"
    except Exception as erro:
        return f"falha ao entregar o webhook: {erro}"


def _gerar_resultado(
    tema: str,
    usar_memoria: bool = False,
    contexto_extra: str = "",
    usar_gerente: bool = False,
) -> dict:
    """Executa a agência completa e devolve um dict com DUAS versões do texto:

      - conteudo:       COM as crases de markdown (```lang) — é o que vai para
                        o front-end/editor, para o Streamlit colorir o código.
                        Já vem sem os artefatos internos do modelo.
      - conteudo_limpo: CRU, sem crases — idêntico ao arquivo salvo em disco,
                        usado pelo botão de download (arquivo executável).

    Também devolve arquivo/tipo/linguagem. Fonte única usada pelo job
    assíncrono e pelo endpoint OpenAI-compatível. escolher_saida recupera o
    código caso o Revisor o tenha apagado (Bug 1). contexto_extra é o texto
    de arquivo(s) anexado(s) no painel, injetado no prompt do Pesquisador."""
    llm = criar_llm()  # mesma config/fix de tool-calling do main.py
    equipe = montar_equipe(
        tema,
        llm,
        usar_memoria=usar_memoria,
        contexto_extra=contexto_extra,
        usar_gerente=usar_gerente,
    )
    bruto = escolher_saida(equipe.kickoff(), tema)
    arquivo = salvar_relatorio(bruto, tema)  # arquivo cru (salvar_relatorio limpa)
    tipo, linguagem = classificar_saida(str(arquivo))
    return {
        "conteudo": conteudo_para_exibicao(bruto, tipo, linguagem),
        "conteudo_limpo": limpar_saida(bruto),
        "arquivo": str(arquivo),
        "tipo": tipo,
        "linguagem": linguagem,
    }


def _rodar_agencia(
    job_id: str,
    tema: str,
    webhook_url: str | None,
    usar_memoria: bool = False,
    contexto_extra: str = "",
    usar_gerente: bool = False,
) -> None:
    """Roda em background (threadpool do Starlette): a resposta HTTP já foi
    enviada quando isto começa, então demorar minutos aqui não trava nada."""
    _atualizar_job(job_id, status="executando")
    try:
        res = _gerar_resultado(tema, usar_memoria, contexto_extra, usar_gerente)
        campos = {
            "arquivo_salvo": res["arquivo"],
            "conteudo": res["conteudo"],              # COM crases (p/ o painel colorir)
            "conteudo_limpo": res["conteudo_limpo"],  # cru (p/ o download)
            "tipo": res["tipo"],
            "linguagem": res["linguagem"],
        }
        _atualizar_job(
            job_id,
            status="concluido",
            terminado_em=datetime.now().isoformat(timespec="seconds"),
            **campos,
        )
        payload = {"job_id": job_id, "status": "concluido", **campos}
    except Exception as erro:
        _atualizar_job(
            job_id,
            status="erro",
            detalhe=str(erro),
            terminado_em=datetime.now().isoformat(timespec="seconds"),
        )
        payload = {"job_id": job_id, "status": "erro", "detalhe": str(erro)}

    if webhook_url:
        erro_webhook = _disparar_webhook(webhook_url, payload)
        _atualizar_job(
            job_id,
            webhook_entregue=(erro_webhook is None),
            webhook_detalhe=erro_webhook,
        )


def _registrar_job(tema: str, **extras) -> str:
    """Cria a entrada do job no registro em memória e devolve o job_id.

    Extraído para ser compartilhado por /agencia e /podcast: são dois tipos de
    trabalho, mas a poda do registro e o teto de jobs precisam valer para os
    dois — se cada endpoint tivesse a sua cópia, o teto deixaria de ser uma
    garantia real no momento em que um deles esquecesse de podar.
    """
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        # Poda os jobs finalizados mais antigos para a memória não crescer sem fim
        if len(_jobs) >= _MAX_JOBS_GUARDADOS:
            finalizados = [
                jid for jid, j in _jobs.items() if j["status"] in ("concluido", "erro")
            ]
            for jid in finalizados[: len(_jobs) - _MAX_JOBS_GUARDADOS + 1]:
                del _jobs[jid]
        # Se depois de podar o teto ainda estourou, é porque há jobs demais
        # AINDA em andamento (rajada de POSTs com o Ollama lento). Recusa em
        # vez de crescer sem limite — o teto passa a ser uma garantia real.
        if len(_jobs) >= _MAX_JOBS_GUARDADOS:
            raise HTTPException(
                status_code=503,
                detail="Muitos jobs em andamento agora. Tente de novo em instantes.",
            )
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "na_fila",
            "tema": tema_curto(tema),
            "criado_em": datetime.now().isoformat(timespec="seconds"),
            **extras,
        }
    return job_id


def _validar_webhook(url: str | None) -> None:
    if url and not url.lower().startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="webhook_url deve começar com http:// ou https://"
        )


@app.post("/agencia")
def executar_agencia(pedido: PedidoAgencia, tarefas: BackgroundTasks):
    """Aceita o pedido, valida o ambiente e agenda a equipe em background.
    Responde imediatamente — o n8n nunca mais vê timeout."""
    problema = diagnostico_ollama()
    if problema:
        raise HTTPException(status_code=503, detail=problema)

    _validar_webhook(pedido.webhook_url)
    job_id = _registrar_job(pedido.tema)

    tarefas.add_task(
        _rodar_agencia,
        job_id,
        pedido.tema,
        pedido.webhook_url,
        pedido.usar_memoria,
        pedido.contexto_extra,
        pedido.usar_gerente,
    )
    return {
        "status": "Recebido! Pensando...",
        "job_id": job_id,
        "acompanhe_em": f"/status/{job_id}",
    }


def tema_curto(tema: str, limite: int = 200) -> str:
    return tema if len(tema) <= limite else tema[:limite] + "…"


@app.get("/status/{job_id}")
def status_job(job_id: str):
    """Consulta a situação de um job (polling). Quando status == 'concluido',
    a resposta inclui arquivo_salvo e conteudo."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        job = dict(job) if job else None
    if job is None:
        raise HTTPException(status_code=404, detail="job_id desconhecido")
    return job


# ---------------------------------------------------------------------------
# ESTÚDIO DE PODCAST — roteiro (crew ou colado) -> vozes -> episódio
# ---------------------------------------------------------------------------
# Por que um endpoint próprio em vez de reaproveitar /agencia: o resultado
# aqui é BINÁRIO. O registro de jobs guarda `conteudo` como string e o
# /status devolve JSON — áudio não cabe nesse contrato. Então o job guarda os
# CAMINHOS dos arquivos e o áudio sai por uma rota dedicada, que faz stream do
# arquivo. O resto da infraestrutura (registro, poda, polling, webhook) é
# reaproveitado como está.
WEB_DIR = estudio.BASE_DIR / "web"


def _caminho_de_projeto(bruto: str, sufixos: tuple[str, ...]) -> Path:
    """Resolve um caminho vindo do cliente e recusa qualquer um fora do projeto.

    Mesmo princípio do sandbox de terminal do main.py: a API não tem
    autenticação (e pode ter sido aberta para a rede com NCX_HOST), então um
    caminho arbitrário no corpo do POST não pode virar leitura de qualquer
    arquivo do disco. `resolve()` fecha o buraco de "..\\" e de symlink.
    """
    alvo = Path(bruto)
    alvo = (estudio.BASE_DIR / alvo).resolve() if not alvo.is_absolute() else alvo.resolve()
    if estudio.BASE_DIR not in alvo.parents:
        raise HTTPException(
            status_code=400, detail="Caminho fora da pasta do projeto — bloqueado."
        )
    if alvo.suffix.lower() not in sufixos:
        raise HTTPException(
            status_code=400,
            detail=f"Extensão não aceita ({alvo.suffix or 'sem extensão'}). "
            f"Use: {', '.join(sufixos)}",
        )
    if not alvo.is_file():
        raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {bruto}")
    return alvo


def _rodar_podcast(job_id: str, pedido: PedidoPodcast, trilha: Path | None) -> None:
    """Roda em background: (1) roteiro, (2) vozes, (3) montagem.

    O `status` avança em etapas nomeadas porque o estúdio HTML mostra em qual
    delas o job está — num episódio de 5 minutos a escrita pode levar minutos
    e a síntese mais um tanto, e uma barra parada sem rótulo parece travamento.
    """
    try:
        apresentadores = estudio.montar_apresentadores(pedido.vozes)

        roteiro = pedido.roteiro.strip()
        if roteiro:
            _atualizar_job(job_id, status="sintetizando", origem_roteiro="colado")
        else:
            _atualizar_job(job_id, origem_roteiro="crew")
            escrever = (
                _escrever_em_passe_unico
                if pedido.modo.strip().lower() == "unico"
                else _escrever_em_blocos
            )
            saida = escrever(job_id, pedido, apresentadores)
            roteiro = saida["bruto"]
            _atualizar_job(
                job_id,
                status="sintetizando",
                etapa="",
                blocos=saida["blocos"],
                arquivo_roteiro_bruto=saida["arquivo_roteiro"],
            )

        resultado = estudio.gerar_podcast(
            roteiro,
            titulo=pedido.titulo or tema_curto(pedido.tema or "Episódio", 60),
            motor=pedido.motor,
            apresentadores=apresentadores,
            trilha=trilha,
        )
        campos = {
            "arquivo_wav": resultado["arquivo_wav"],
            "arquivo_mp3": resultado["arquivo_mp3"],
            "arquivo_roteiro": resultado["arquivo_roteiro"],
            "duracao": resultado["duracao"],
            "total_falas": resultado["total_falas"],
            "falas": resultado["falas"],
            "motor": resultado["motor"],
            "aviso_mp3": resultado["aviso_mp3"],
            "audio_em": f"/podcast/{job_id}/audio",
        }
        _atualizar_job(
            job_id,
            status="concluido",
            terminado_em=datetime.now().isoformat(timespec="seconds"),
            **campos,
        )
        payload = {"job_id": job_id, "status": "concluido", **campos}
    except Exception as erro:
        _atualizar_job(
            job_id,
            status="erro",
            detalhe=str(erro),
            terminado_em=datetime.now().isoformat(timespec="seconds"),
        )
        payload = {"job_id": job_id, "status": "erro", "detalhe": str(erro)}

    if pedido.webhook_url:
        erro_webhook = _disparar_webhook(pedido.webhook_url, payload)
        _atualizar_job(
            job_id,
            webhook_entregue=(erro_webhook is None),
            webhook_detalhe=erro_webhook,
        )


@app.post("/podcast")
def criar_podcast(pedido: PedidoPodcast, tarefas: BackgroundTasks):
    """Agenda a produção de um episódio e responde na hora com o job_id.

    Toda validação cara acontece AQUI, no caminho síncrono: motor conhecido,
    vozes existentes, modelo baixado, trilha dentro do projeto. Descobrir
    qualquer uma dessas coisas lá no background significaria o usuário
    esperar minutos para receber um erro que era visível no primeiro segundo.
    """
    _validar_webhook(pedido.webhook_url)

    if not pedido.roteiro.strip() and not pedido.tema.strip():
        raise HTTPException(
            status_code=400,
            detail="Informe 'roteiro' (texto pronto) ou 'tema' (a crew escreve).",
        )

    # Só o caminho que passa pela crew depende do Ollama; um roteiro colado
    # gera áudio com o Ollama desligado, e é isso que faz o estúdio ser útil
    # antes de os modelos estarem instalados.
    if not pedido.roteiro.strip():
        problema = diagnostico_ollama()
        if problema:
            raise HTTPException(status_code=503, detail=problema)

    if not 1 <= pedido.minutos <= 60:
        raise HTTPException(status_code=400, detail="minutos deve ficar entre 1 e 60.")

    try:
        apresentadores = estudio.montar_apresentadores(pedido.vozes)
        motor = estudio.criar_motor(pedido.motor)
        motor.verificar(apresentadores)
    except estudio.ErroPodcast as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro

    trilha = (
        _caminho_de_projeto(pedido.trilha, (".wav",)) if pedido.trilha else None
    )

    job_id = _registrar_job(
        pedido.tema or (pedido.titulo or "roteiro colado"), tipo="podcast"
    )
    tarefas.add_task(_rodar_podcast, job_id, pedido, trilha)
    return {
        "status": "Recebido! Produzindo o episódio...",
        "job_id": job_id,
        "acompanhe_em": f"/status/{job_id}",
        "audio_em": f"/podcast/{job_id}/audio",
    }


def _escrever_em_blocos(job_id: str, pedido: PedidoPodcast, apresentadores) -> dict:
    """Pauta pesquisada pela crew + um passe curto por bloco.

    A pesquisa (com busca e leitura de página) roda UMA vez, na pauta. Os
    blocos são chamadas DIRETAS ao LLM (`llm.call`), sem crew: eles não
    precisam de ferramenta nenhuma, só dos fatos que a pauta já trouxe, e
    rodar quatro agentes por bloco multiplicaria o tempo sem ganho.
    """
    total = estudio.blocos_para(pedido.minutos)
    _atualizar_job(job_id, status="pesquisando", etapa="pauta")

    res = _gerar_resultado(
        estudio.prompt_pauta(
            pedido.tema,
            minutos=pedido.minutos,
            blocos=total,
            tem_contexto=bool(pedido.contexto_extra.strip()),
        ),
        usar_memoria=pedido.usar_memoria,
        contexto_extra=pedido.contexto_extra,
    )
    pauta_bruta = res["conteudo_limpo"]
    blocos = estudio.dividir_pauta(pauta_bruta, total)

    llm = criar_llm()
    alvo_falas = estudio.falas_por_bloco(pedido.minutos, len(blocos))
    falas: list[estudio.Fala] = []
    partes: list[str] = []

    for indice, bloco in enumerate(blocos, start=1):
        _atualizar_job(
            job_id,
            status="escrevendo_roteiro",
            etapa=f"bloco {indice} de {len(blocos)}",
        )
        try:
            texto = llm.call(
                estudio.prompt_bloco(
                    pedido.tema,
                    bloco,
                    indice=indice,
                    total=len(blocos),
                    apresentadores=apresentadores,
                    falas=alvo_falas,
                    ultimas_falas=estudio.ultimas_falas_como_texto(falas),
                )
            )
        except Exception as erro:
            # Um bloco que falha não derruba o episódio: os outros já escritos
            # continuam valendo, e o usuário completa o buraco na mão no editor.
            partes.append(f"[bloco {indice} falhou: {erro}]")
            continue
        partes.append(texto or "")
        # Reparseia o acumulado a cada bloco para (a) alimentar a continuidade
        # do próximo com falas JÁ limpas e (b) deduplicar nas emendas.
        falas = estudio.parse_roteiro("\n".join(partes))

    return {
        "falas": falas,
        "bruto": "\n\n".join(partes),
        "pauta": pauta_bruta,
        "blocos": len(blocos),
        "arquivo_roteiro": res["arquivo"],
    }


def _escrever_em_passe_unico(job_id: str, pedido: PedidoPodcast, apresentadores) -> dict:
    """Modo antigo: um único pedido para a crew. Mais rápido, entrega menos."""
    _atualizar_job(job_id, status="escrevendo_roteiro", etapa="passe único")
    res = _gerar_resultado(
        estudio.prompt_roteiro(
            pedido.tema,
            minutos=pedido.minutos,
            apresentadores=apresentadores,
            tem_contexto=bool(pedido.contexto_extra.strip()),
        ),
        usar_memoria=pedido.usar_memoria,
        contexto_extra=pedido.contexto_extra,
    )
    bruto = res["conteudo_limpo"]
    return {
        "falas": estudio.parse_roteiro(bruto),
        "bruto": bruto,
        "pauta": None,
        "blocos": 1,
        "arquivo_roteiro": res["arquivo"],
    }


def _rodar_roteiro(job_id: str, pedido: PedidoPodcast) -> None:
    """Só a etapa 1: a equipe escreve o roteiro e para por aí."""
    _atualizar_job(job_id, status="escrevendo_roteiro")
    try:
        apresentadores = estudio.montar_apresentadores(pedido.vozes)
        escrever = (
            _escrever_em_passe_unico
            if pedido.modo.strip().lower() == "unico"
            else _escrever_em_blocos
        )
        saida = escrever(job_id, pedido, apresentadores)

        falas = saida["falas"]
        # Devolve o roteiro JÁ NORMALIZADO pelo parser (sem markdown, sem
        # rubricas, sem preâmbulo). Assim o que o usuário edita na tela é
        # exatamente o que vai virar áudio — sem surpresa entre o que ele leu
        # e o que o sintetizador recebeu.
        campos = {
            "roteiro": "\n".join(f"{f.locutor}: {f.texto}" for f in falas),
            "roteiro_bruto": saida["bruto"],
            "pauta": saida["pauta"],
            "blocos": saida["blocos"],
            "total_falas": len(falas),
            "arquivo_roteiro": saida["arquivo_roteiro"],
        }
        _atualizar_job(
            job_id,
            status="concluido",
            terminado_em=datetime.now().isoformat(timespec="seconds"),
            **campos,
        )
        payload = {"job_id": job_id, "status": "concluido", **campos}
    except Exception as erro:
        _atualizar_job(
            job_id,
            status="erro",
            detalhe=str(erro),
            terminado_em=datetime.now().isoformat(timespec="seconds"),
        )
        payload = {"job_id": job_id, "status": "erro", "detalhe": str(erro)}

    if pedido.webhook_url:
        _disparar_webhook(pedido.webhook_url, payload)


@app.post("/podcast/roteiro")
def escrever_roteiro(pedido: PedidoPodcast, tarefas: BackgroundTasks):
    """Escreve o roteiro SEM sintetizar — o passo que faz o estúdio ser usável.

    A crew leva minutos e o 7B erra o tom de vez em quando; a síntese leva
    mais minutos. Sintetizar direto significaria jogar fora as duas etapas
    por causa de um roteiro ruim. Separando, o usuário lê, corrige à mão e só
    então paga o custo do áudio — e corrigir texto é grátis.
    """
    _validar_webhook(pedido.webhook_url)
    if not pedido.tema.strip():
        raise HTTPException(status_code=400, detail="Informe o 'tema' do episódio.")
    if not 1 <= pedido.minutos <= 60:
        raise HTTPException(status_code=400, detail="minutos deve ficar entre 1 e 60.")

    problema = diagnostico_ollama()
    if problema:
        raise HTTPException(status_code=503, detail=problema)

    job_id = _registrar_job(pedido.tema, tipo="roteiro")
    tarefas.add_task(_rodar_roteiro, job_id, pedido)
    return {
        "status": "Recebido! Escrevendo o roteiro...",
        "job_id": job_id,
        "acompanhe_em": f"/status/{job_id}",
    }


@app.get("/podcast/vozes")
def listar_vozes():
    """Catálogo pt-BR + quais já estão baixadas — popula o seletor do estúdio.

    `baixada` importa: sem ela o usuário escolheria uma voz que não existe em
    disco e só descobriria no POST. Com ela, o estúdio já mostra desabilitada.
    """
    motor = estudio.MotorPiper()
    return {
        "pasta": str(motor.dir_vozes),
        "vozes": [
            {
                "id": nome,
                "baixada": motor.caminho_voz(nome).exists(),
                **ficha,
            }
            for nome, ficha in estudio.VOZES_PT_BR.items()
        ],
        "motores": sorted(estudio.MOTORES),
        "padrao": [a.voz for a in estudio.APRESENTADORES_PADRAO],
    }


@app.get("/podcast/ambiente")
def checar_ambiente():
    """Diagnóstico que o estúdio consulta ao abrir.

    Fica separado de /podcast/vozes de propósito: `diagnostico_ollama` faz uma
    chamada de rede que só retorna no timeout quando o Ollama está desligado,
    e a lista de vozes (leitura de disco, instantânea) não pode ficar refém
    dela. Assim o navegador dispara as duas em paralelo e a tela monta na hora.
    """
    problema = diagnostico_ollama()
    return {
        "ollama_ok": problema is None,
        "ollama_detalhe": problema,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


@app.get("/podcast/{job_id}/audio")
def baixar_audio(job_id: str):
    """Entrega o episódio: MP3 quando houver, WAV como alternativa.

    É a rota que a tag <audio> do estúdio consome. O FileResponse faz stream
    do arquivo do disco, então um episódio de uma hora não é carregado na
    memória do servidor para ser servido.
    """
    with _jobs_lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        raise HTTPException(status_code=404, detail="job_id desconhecido")
    if job.get("status") != "concluido":
        raise HTTPException(
            status_code=409,
            detail=f"O episódio ainda não está pronto (status: {job.get('status')}).",
        )

    caminho = job.get("arquivo_mp3") or job.get("arquivo_wav")
    if not caminho or not Path(caminho).is_file():
        raise HTTPException(status_code=404, detail="Arquivo de áudio não encontrado.")

    arquivo = Path(caminho)
    return FileResponse(
        arquivo,
        media_type="audio/mpeg" if arquivo.suffix == ".mp3" else "audio/wav",
        filename=arquivo.name,
    )


@app.get("/estudio")
def abrir_estudio():
    """Serve o estúdio HTML pela PRÓPRIA API, e não como arquivo solto.

    Motivo técnico, não estético: aberto via file://, o navegador trataria a
    página e a API como origens diferentes e o fetch morreria no CORS. Servido
    daqui, tudo é a mesma origem (localhost:8000) e não existe configuração de
    CORS para acertar.
    """
    pagina = WEB_DIR / "estudio.html"
    if not pagina.is_file():
        raise HTTPException(status_code=404, detail=f"Página não encontrada: {pagina}")
    return FileResponse(pagina, media_type="text/html; charset=utf-8")


# ---------------------------------------------------------------------------
# FACHADA OPENAI-COMPATÍVEL (/v1) — para Continue.dev / Cline no VS Code
# ---------------------------------------------------------------------------
# Notas de compatibilidade (verificadas na pesquisa):
#  - Continue/Cline mandam stream=true por padrão e NÃO renderizam uma
#    resposta JSON simples — por isso o caminho SSE é o principal.
#  - A crew leva minutos; se ficarmos calados até o fim, o editor dá timeout.
#    Solução: abrir o stream JÁ, mandar keepalives invisíveis (comentários
#    SSE ": ...") enquanto a crew roda numa thread, e só então o conteúdo.
#  - Aceitar-e-ignorar campos extras (temperature, tools, etc.): recusar
#    qualquer um quebra os editores. Por isso o corpo é um dict solto.
#  - A chave de API é ignorada (uso local), mas os clientes sempre mandam
#    uma — aceitamos qualquer header Authorization.
#  - Cursor NÃO funciona com localhost (faz proxy pela nuvem dele); alvo
#    são Continue.dev e Cline.
MODELO_API = "cortexus-crew"  # id estável exposto; esconde o ollama/qwen interno


@app.get("/v1/models")
def listar_modelos():
    """Cline consulta este endpoint para popular a lista de modelos."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODELO_API,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cortexus",
            }
        ],
    }


def _extrair_tema(body: dict) -> str:
    """Mapeia messages[] da OpenAI para o 'tema' único da crew: pega a
    ÚLTIMA mensagem de role 'user'. Não concatena histórico nem inclui o
    system prompt do editor (poluiria a Regra de Intencionalidade). O
    'content' pode ser string OU lista de partes (formato de visão)."""
    mensagens = body.get("messages") or []
    conteudo_user = ""
    for msg in mensagens:
        if isinstance(msg, dict) and msg.get("role") == "user":
            c = msg.get("content", "")
            if isinstance(c, list):  # partes: [{type:text, text:...}, ...]
                c = " ".join(
                    p.get("text", "")
                    for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            conteudo_user = c if isinstance(c, str) else ""
    return conteudo_user.strip()


def _frame_chunk(cid: str, modelo: str, criado: int, delta: dict, finish=None) -> str:
    """Monta um frame SSE 'chat.completion.chunk' já no formato 'data: ...'."""
    obj = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": criado,
        "model": modelo,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    """Fachada OpenAI: roda a agência completa e devolve o resultado como
    uma resposta de chat. Streaming por padrão (com keepalive), pois é o
    que Continue/Cline esperam."""
    tema = _extrair_tema(body) or TEMA_PADRAO
    modelo_echo = body.get("model") or MODELO_API
    stream = body.get("stream", True)
    cid = "chatcmpl-" + uuid.uuid4().hex
    criado = int(time.time())

    problema = diagnostico_ollama()

    # --- Caminho não-streaming (curl/SDK; os editores raramente usam) ---
    if not stream:
        if problema:
            raise HTTPException(status_code=503, detail=problema)
        try:
            # conteudo COM crases — editores renderizam markdown/código colorido
            conteudo = _gerar_resultado(tema)["conteudo"]
        except Exception as erro:
            raise HTTPException(status_code=500, detail=str(erro)) from erro
        return {
            "id": cid,
            "object": "chat.completion",
            "created": criado,
            "model": modelo_echo,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": conteudo},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # --- Caminho streaming (SSE) — principal para os editores ---
    def gerar() -> Iterator[str]:
        # Frame inicial imediato: o cliente vê bytes e não dá timeout.
        yield _frame_chunk(cid, modelo_echo, criado, {"role": "assistant"})

        if problema:  # Ollama fora do ar: entrega o aviso como conteúdo e encerra
            yield _frame_chunk(cid, modelo_echo, criado, {"content": f"⚠️ {problema}"})
            yield _frame_chunk(cid, modelo_echo, criado, {}, finish="stop")
            yield "data: [DONE]\n\n"
            return

        # A crew (minutos) roda numa thread; o gerador emite keepalives
        # invisíveis (comentários SSE) enquanto espera, mantendo o socket vivo.
        holder: dict[str, str] = {}

        def worker():
            try:
                holder["res"] = _gerar_resultado(tema)["conteudo"]
            except Exception as erro:  # nunca deixa a thread morrer sem registro
                holder["err"] = str(erro)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=12)
            if t.is_alive():
                yield ": keepalive\n\n"  # comentário SSE: mantém vivo, não renderiza

        conteudo = holder.get("res") or f"⚠️ {holder.get('err', 'falha desconhecida')}"
        yield _frame_chunk(cid, modelo_echo, criado, {"content": conteudo})
        yield _frame_chunk(cid, modelo_echo, criado, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gerar(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# PONTE NEXUS (microsserviço para outro orquestrador — ex.: Roo Code)
# ---------------------------------------------------------------------------
@app.post("/v1/nexus-assist")
def nexus_assist(pedido: NexusAssist):
    """Executa a agência de forma SÍNCRONA e devolve o relatório num único
    JSON — pensado para uma chamada pontual de outro agente (via
    nexus_bridge.py). SEMPRE modo sequencial (usar_gerente=False): assim o
    Nexus faz o trabalho pesado nos bastidores sem virar um Gerente que
    delega a mais agentes (evita o loop 'agente delegando para agente').

    A chamada BLOQUEIA por minutos (a crew roda inteira); o cliente CLI
    espera. FastAPI roda esta função def numa threadpool, então o event loop
    não trava para outras requisições."""
    problema = diagnostico_ollama()
    if problema:
        raise HTTPException(status_code=503, detail=problema)

    tarefa = pedido.tarefa.strip()
    if not tarefa:
        raise HTTPException(status_code=400, detail="O campo 'tarefa' está vazio.")

    # Enquadra o pedido conforme o tipo, para orientar a Regra de Intencionalidade.
    tipo = (pedido.tipo or "pesquisa").strip().lower()
    if tipo == "documentacao":
        tema = (
            f"Documente de forma clara, técnica e estruturada (em prosa, com "
            f"seções): {tarefa}"
        )
    else:  # "pesquisa" (padrão) — qualquer valor desconhecido cai aqui
        tipo = "pesquisa"
        tema = (
            f"Pesquise a fundo e produza um dossiê estruturado, com fontes e "
            f"links quando houver, sobre: {tarefa}"
        )

    try:
        res = _gerar_resultado(tema, usar_memoria=False, usar_gerente=False)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=str(erro)) from erro

    return {
        "status": "concluido",
        "tipo": tipo,
        # conteudo_limpo (sem crases) é o melhor para ler no terminal do cliente
        "relatorio": res["conteudo_limpo"],
        "arquivo_salvo": res["arquivo"],
        "tipo_saida": res["tipo"],      # 'codigo' | 'analise'
        "linguagem": res["linguagem"],
    }


# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DO SERVIDOR (host / porta / reload)
# ---------------------------------------------------------------------------
# O padrão é loopback e sem reload. Antes era host="0.0.0.0" + reload=True
# fixos: qualquer aparelho da rede local alcançava /agencia — um agente com
# terminal — sem senha nenhuma. Abrir para a rede agora é escolha explícita.
HOST_PADRAO = "127.0.0.1"
PORTA_PADRAO = 8000
_VERDADEIROS = frozenset({"1", "true", "sim", "s", "yes", "y", "on"})


def _env_booleana(valor: str | None) -> bool:
    return (valor or "").strip().lower() in _VERDADEIROS


def host_e_loopback(host: str) -> bool:
    """True se o host só aceita conexões desta máquina."""
    host = host.strip().strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # nome de host qualquer: na dúvida, trata como exposto


def configuracao_servidor(
    argv: Sequence[str] | None = None, ambiente: Mapping[str, str] | None = None
) -> argparse.Namespace:
    """Resolve host/porta/reload: argumento de linha de comando > variável de
    ambiente (NCX_HOST, NCX_PORT, NCX_RELOAD) > padrão seguro."""
    ambiente = os.environ if ambiente is None else ambiente
    parser = argparse.ArgumentParser(
        prog="python api.py", description="Sobe a API do Cortexus."
    )
    parser.add_argument(
        "--host",
        default=ambiente.get("NCX_HOST") or HOST_PADRAO,
        help=f"interface de escuta (env NCX_HOST; padrão {HOST_PADRAO}). "
        "0.0.0.0 expõe na rede local SEM autenticação.",
    )
    # argparse aplica o `type` também ao default em string: um NCX_PORT
    # inválido cai no mesmo erro amigável que um --port inválido.
    parser.add_argument(
        "--port",
        type=int,
        default=ambiente.get("NCX_PORT") or str(PORTA_PADRAO),
        help=f"porta (env NCX_PORT; padrão {PORTA_PADRAO})",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=_env_booleana(ambiente.get("NCX_RELOAD")),
        help="recarrega ao editar o código — só para desenvolvimento "
        "(env NCX_RELOAD; padrão desligado)",
    )
    return parser.parse_args(argv)


def aviso_de_exposicao(host: str, porta: int) -> str | None:
    """Mensagem de alerta quando o host sai do loopback; None se é seguro."""
    if host_e_loopback(host):
        return None
    return (
        "\n"
        "!!! ATENÇÃO: a API do Cortexus vai escutar em "
        f"{host}:{porta}, FORA do loopback.\n"
        "!!! Não há autenticação: qualquer aparelho que alcance esta máquina\n"
        "!!! pode chamar POST /agencia — um agente com acesso ao TERMINAL.\n"
        "!!! Use só numa rede confiável e com firewall. Para voltar ao modo\n"
        f"!!! seguro, remova NCX_HOST / --host (padrão {HOST_PADRAO}).\n"
    )


if __name__ == "__main__":
    import uvicorn

    # Reload é opt-in (--reload / NCX_RELOAD). Ligado, reload_excludes é
    # essencial: a agência GRAVA arquivos em output/
    # (inclusive .py) no meio do job — sem a exclusão, o watcher do reload
    # reiniciaria o servidor e MATARIA o próprio job que gerou o arquivo.
    # Detalhes que importam (verificados no código do uvicorn):
    #   - "output" (nome de diretório EXISTENTE) exclui recursivamente;
    #     "output/*" sozinho só cobriria filhos diretos. Usamos os dois.
    #   - a exclusão só funciona com watchfiles instalado (uvicorn[standard]);
    #     sem ele o fallback StatReload ignora excludes.
    #   - NUNCA rode com workers>1: o registro de jobs é em memória, por
    #     processo. E todo reload/restart perde os jobs em andamento.
    config = configuracao_servidor()
    aviso = aviso_de_exposicao(config.host, config.port)
    if aviso:
        print(aviso, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # o diretório precisa existir no boot
    opcoes_reload = (
        {"reload": True, "reload_excludes": ["output", "output/*"]}
        if config.reload
        else {}
    )
    uvicorn.run("api:app", host=config.host, port=config.port, **opcoes_reload)
