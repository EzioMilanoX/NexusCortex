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
  python api.py
  (ou, sem reload)  uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /agencia                 {"tema": "...", "webhook_url": "http://... (opcional)"}
  GET  /status/{job_id}         situação e resultado do job
  POST /v1/chat/completions     fachada OpenAI (streaming SSE) — para editores
  GET  /v1/models               lista o modelo "cortexus-crew"
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from main import (
    OUTPUT_DIR,
    TEMA_PADRAO,
    criar_llm,
    diagnostico_ollama,
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


# Extensões que representam CÓDIGO (vão para a aba "Código" com highlighting)
# e o mapa para a linguagem do bloco markdown. .md/.txt são prosa (aba Análise).
_LINGUAGEM_POR_EXT = {".py": "python", ".json": "json", ".csv": "csv"}


def _classificar(arquivo: str) -> tuple[str, str]:
    """A partir do nome do arquivo salvo, devolve (tipo, linguagem):
    tipo = 'codigo' ou 'analise'; linguagem = ex. 'python' (só p/ código).
    A extensão é a autoridade — quem a decidiu foi nome_arquivo_pedido()."""
    ext = ("." + arquivo.rsplit(".", 1)[-1].lower()) if "." in arquivo else ""
    if ext in _LINGUAGEM_POR_EXT:
        return "codigo", _LINGUAGEM_POR_EXT[ext]
    return "analise", ""


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


def _gerar_resultado(tema: str, usar_memoria: bool = False) -> tuple[str, str]:
    """Executa a agência completa para um tema e devolve (conteudo_limpo,
    caminho_do_arquivo). Fonte única de execução, usada pelo job assíncrono
    e pelo endpoint OpenAI-compatível. O conteúdo devolvido é sempre LIMPO
    (sem crases); quem exibe (ui.py) reembrulha em bloco de código se for o
    caso — o arquivo em disco continua limpo, pronto para rodar."""
    llm = criar_llm()  # mesma config/fix de tool-calling do main.py
    equipe = montar_equipe(tema, llm, usar_memoria=usar_memoria)
    resultado = limpar_saida(str(equipe.kickoff()))
    arquivo = salvar_relatorio(resultado, tema)
    return resultado, str(arquivo)


def _rodar_agencia(
    job_id: str, tema: str, webhook_url: str | None, usar_memoria: bool = False
) -> None:
    """Roda em background (threadpool do Starlette): a resposta HTTP já foi
    enviada quando isto começa, então demorar minutos aqui não trava nada."""
    _atualizar_job(job_id, status="executando")
    try:
        resultado, arquivo = _gerar_resultado(tema, usar_memoria)
        tipo, linguagem = _classificar(arquivo)
        _atualizar_job(
            job_id,
            status="concluido",
            arquivo_salvo=arquivo,
            conteudo=resultado,
            tipo=tipo,
            linguagem=linguagem,
            terminado_em=datetime.now().isoformat(timespec="seconds"),
        )
        payload = {
            "job_id": job_id,
            "status": "concluido",
            "arquivo_salvo": arquivo,
            "conteudo": resultado,
            "tipo": tipo,
            "linguagem": linguagem,
        }
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


@app.post("/agencia")
def executar_agencia(pedido: PedidoAgencia, tarefas: BackgroundTasks):
    """Aceita o pedido, valida o ambiente e agenda a equipe em background.
    Responde imediatamente — o n8n nunca mais vê timeout."""
    problema = diagnostico_ollama()
    if problema:
        raise HTTPException(status_code=503, detail=problema)

    if pedido.webhook_url and not pedido.webhook_url.lower().startswith(
        ("http://", "https://")
    ):
        raise HTTPException(
            status_code=400, detail="webhook_url deve começar com http:// ou https://"
        )

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
            "tema": tema_curto(pedido.tema),
            "criado_em": datetime.now().isoformat(timespec="seconds"),
        }

    tarefas.add_task(
        _rodar_agencia, job_id, pedido.tema, pedido.webhook_url, pedido.usar_memoria
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
            conteudo, _ = _gerar_resultado(tema)
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
                holder["res"], _ = _gerar_resultado(tema)
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


if __name__ == "__main__":
    import uvicorn

    # reload_excludes é essencial: a agência GRAVA arquivos em output/
    # (inclusive .py) no meio do job — sem a exclusão, o watcher do reload
    # reiniciaria o servidor e MATARIA o próprio job que gerou o arquivo.
    # Detalhes que importam (verificados no código do uvicorn):
    #   - "output" (nome de diretório EXISTENTE) exclui recursivamente;
    #     "output/*" sozinho só cobriria filhos diretos. Usamos os dois.
    #   - a exclusão só funciona com watchfiles instalado (uvicorn[standard]);
    #     sem ele o fallback StatReload ignora excludes.
    #   - NUNCA rode com workers>1: o registro de jobs é em memória, por
    #     processo. E todo reload/restart perde os jobs em andamento.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # o diretório precisa existir no boot
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["output", "output/*"],
    )
