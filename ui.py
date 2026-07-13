"""Cortexus — painel visual (Streamlit).

Chat visual para acionar a agência sem terminal. Conversa com a API
assíncrona (api.py) via POST /agencia + polling em /status — o MESMO
caminho do n8n, então o comportamento é idêntico.

Como rodar (na pasta do projeto, com o venv ativado):
  1. Suba a API:     python api.py
  2. Suba o painel:  streamlit run ui.py
  (ou use o atalho de um clique: cortexus.bat / cortexus.sh)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
INTERVALO_POLL_S = 2.0       # de quanto em quanto tempo consulta /status
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# Teto do contexto anexado: o modelo roda com num_ctx 8192, e o prompt já
# carrega backstory + esquemas de ferramentas. ~8000 caracteres (~2-2,5k
# tokens) deixa folga; acima disso, truncamos para não estourar o contexto.
MAX_CONTEXTO_CHARS = 8000
EXTS_UPLOAD = ["txt", "py", "md", "csv"]

# linguagem por extensão (highlighting do preview / download)
_LINGUAGEM = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".sh": "bash",
    ".ps1": "powershell", ".sql": "sql", ".html": "html", ".css": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
}
_MIME = {".py": "text/x-python", ".json": "application/json", ".csv": "text/csv",
         ".md": "text/markdown", ".sh": "text/x-shellscript", ".txt": "text/plain"}

st.set_page_config(page_title="Cortexus", page_icon="🧠", layout="wide")
st.title("🧠 Cortexus")
st.caption("Agência local de agentes de IA (CrewAI + Ollama) — 100% offline.")


def montar_contexto_extra(arquivos) -> str:
    """Extrai o texto dos arquivos anexados e monta um único bloco rotulado.
    Trunca no total para não estourar o contexto do modelo (num_ctx 8192)."""
    if not arquivos:
        return ""
    partes = []
    for arq in arquivos:
        try:
            texto = arq.getvalue().decode("utf-8", errors="replace")
        except Exception as erro:  # nunca deixa um arquivo ruim derrubar a UI
            texto = f"(não consegui ler este arquivo: {erro})"
        partes.append(f"### Arquivo: {arq.name} ###\n{texto}")
    contexto = "\n\n".join(partes)
    if len(contexto) > MAX_CONTEXTO_CHARS:
        contexto = (
            contexto[:MAX_CONTEXTO_CHARS]
            + "\n\n[... contexto truncado para caber na janela do modelo ...]"
        )
    return contexto


def listar_gerados() -> list[Path]:
    """Arquivos em output/ (mais recentes primeiro), ignorando o .gitkeep."""
    if not OUTPUT_DIR.exists():
        return []
    arquivos = [
        p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.name != ".gitkeep"
    ]
    return sorted(arquivos, key=lambda p: p.stat().st_mtime, reverse=True)


def api_no_ar() -> bool:
    """True se a API responde — evita erros crípticos se o api.py não subiu."""
    try:
        requests.get(f"{API_BASE}/docs", timeout=3)
        return True
    except requests.RequestException:
        return False


def rodar_agencia(payload: dict, status, timeout_job: int) -> dict:
    """POST /agencia + polling em /status, atualizando o st.status com as
    etapas. Devolve o dict final do job (concluído) ou levanta RuntimeError."""
    try:
        resp = requests.post(f"{API_BASE}/agencia", json=payload, timeout=15)
    except requests.RequestException as erro:
        raise RuntimeError(f"Não consegui falar com a API: {erro}") from erro
    if resp.status_code == 503:
        raise RuntimeError(resp.json().get("detail", "Ollama indisponível (503)."))
    if resp.status_code != 200:
        raise RuntimeError(f"A API respondeu HTTP {resp.status_code}: {resp.text[:200]}")

    job_id = resp.json()["job_id"]
    # A API só reporta na_fila/executando (não por-agente), mas o rótulo já
    # reflete o pipeline atual de 4 etapas para dar noção do que acontece.
    rotulos = {
        "na_fila": "⏳ Na fila…",
        "executando": (
            "🧠 Analisando requisitos → pesquisando (curadoria + RAG) → "
            "redigindo → revisando…"
        ),
    }
    ultimo_estado = None
    limite = time.monotonic() + timeout_job
    while time.monotonic() < limite:
        time.sleep(INTERVALO_POLL_S)
        try:
            job = requests.get(f"{API_BASE}/status/{job_id}", timeout=10).json()
        except requests.RequestException:
            continue  # falha transitória — tenta de novo no próximo ciclo
        estado = job.get("status")
        if estado != ultimo_estado:
            status.write(rotulos.get(estado, estado))
            ultimo_estado = estado
        if estado == "concluido":
            return job
        if estado == "erro":
            raise RuntimeError(job.get("detalhe", "A equipe falhou sem detalhe."))
    raise RuntimeError(f"O job passou de {timeout_job}s sem concluir.")


def render_resposta(idx: int, entry: dict) -> None:
    """Desenha a resposta do assistente: abas Análise/Código (markdown com
    highlighting), botão de download e o expander do payload para n8n.
    Chamado ao vivo e ao redesenhar o histórico — por isso a 'key' do download
    é indexada. 'conteudo' já vem do back-end COM as crases quando é código."""
    tipo = entry.get("tipo", "analise")
    conteudo = entry.get("conteudo", "")

    aba_chat, aba_codigo = st.tabs(["💬 Análise", "💻 Código Gerado"])
    with aba_chat:
        if tipo == "analise":
            st.markdown(conteudo)
        else:
            st.markdown("_A resposta é código — veja a aba **💻 Código Gerado**._")
    with aba_codigo:
        if tipo == "codigo":
            st.markdown(conteudo)  # já vem cercado com ```lang → colorido
        else:
            st.markdown("_Nenhum código gerado nesta resposta._")

    nome = entry.get("arquivo_nome")
    if nome:
        ext = os.path.splitext(nome)[1].lower()
        st.download_button(
            f"⬇️ Baixar {nome}",
            data=entry.get("conteudo_limpo", conteudo),
            file_name=nome,
            mime=_MIME.get(ext, "text/plain"),
            key=f"dl_{idx}",
        )

    with st.expander("🛠️ Ver Payload da API (Para n8n)"):
        st.json(entry.get("payload", {}))


# ===========================================================================
# BARRA LATERAL — controles, upload e painel de arquivos gerados
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Controles")

    modo = st.radio(
        "Modo de execução",
        ["Sequencial (rápido, confiável)", "Gerente (hierárquico, flexível)"],
        help=(
            "SEQUENCIAL: Analista → Pesquisador → Escritor → Revisor, em cascata "
            "(mais rápido; mantém a rede de segurança que recupera o código).\n\n"
            "GERENTE: um LLM-gerente convoca os especialistas dinamicamente "
            "(inclui Arquiteto e Especialista em Integrações) — mais flexível, "
            "porém bem mais lento no modelo local."
        ),
    )
    usar_gerente = modo.startswith("Gerente")

    usar_memoria = st.toggle(
        "Usar Memória de Longo Prazo",
        value=False,
        help=(
            "Liga a memória (ChromaDB) entre execuções. Desligada por padrão: "
            "com o modelo local, a memória às vezes faz a agência repetir a "
            "saída de uma tarefa anterior parecida (contaminação)."
        ),
    )

    timeout_job = int(
        st.number_input(
            "Timeout do job (segundos)",
            min_value=60, max_value=3600, value=600, step=60,
            help="Quanto o painel espera a agência terminar antes de desistir.",
        )
    )

    rag_pasta = st.text_input(
        "Pasta p/ Busca Semântica (opcional)",
        value="",
        help=(
            "Subpasta do projeto onde o Pesquisador deve fazer a busca "
            "semântica (RAG), ex.: 'output' ou 'src/audio'. Deixe vazio para "
            "a agência decidir. NUNCA aponte para a raiz (indexaria o .venv)."
        ),
    ).strip()

    st.divider()
    st.subheader("📎 Anexar contexto")
    arquivos = st.file_uploader(
        "Arraste arquivos (.txt .py .md .csv)",
        type=EXTS_UPLOAD,
        accept_multiple_files=True,
        help=(
            "O texto é extraído aqui no painel e enviado junto do pedido, "
            "então a agência já lê o conteúdo sem usar a ferramenta de disco."
        ),
    )
    contexto_extra = montar_contexto_extra(arquivos)
    if contexto_extra:
        st.caption(
            f"📎 {len(arquivos)} arquivo(s), {len(contexto_extra)} caracteres "
            "no contexto."
        )

    st.divider()
    st.subheader("📂 Arquivos gerados")
    gerados = listar_gerados()
    if not gerados:
        st.caption("Nada gerado ainda — os resultados aparecem aqui.")
    else:
        escolhido = st.selectbox(
            "Ver arquivo (output/)", [p.name for p in gerados], key="sel_gerado"
        )
        caminho = OUTPUT_DIR / escolhido
        try:
            texto_arq = caminho.read_text(encoding="utf-8", errors="replace")
        except OSError as erro:
            texto_arq = f"(não consegui ler: {erro})"
        ext = caminho.suffix.lower()
        st.download_button(
            f"⬇️ Baixar {escolhido}",
            data=texto_arq,
            file_name=escolhido,
            mime=_MIME.get(ext, "text/plain"),
            key="dl_gerado",
        )
        st.code(texto_arq[:2000], language=_LINGUAGEM.get(ext))

    st.divider()
    st.caption(
        "**Equipe (sequencial):** Analista de Requisitos → Pesquisador-Curador "
        "(busca web + leitura + RAG) → Escritor → Revisor. No modo Gerente, o "
        "LLM-gerente também pode convocar Arquiteto e Especialista em Integrações."
    )


# ===========================================================================
# ÁREA PRINCIPAL — chat
# ===========================================================================
if "historico" not in st.session_state:
    st.session_state.historico = []  # dicts {papel, texto?, conteudo?, tipo?, ...}

if not api_no_ar():
    st.error(
        "A API da Cortexus não está no ar. Em um terminal, na pasta do "
        "projeto e com o venv ativado, rode: **python api.py**"
    )

# Redesenha o histórico
for i, msg in enumerate(st.session_state.historico):
    with st.chat_message(msg["papel"]):
        if msg["papel"] == "user":
            st.markdown(msg["texto"])
        else:
            render_resposta(i, msg)

# Entrada do usuário
pedido = st.chat_input("O que a Cortexus deve fazer? (ex.: 'Explique o README.md')")
if pedido:
    st.session_state.historico.append({"papel": "user", "texto": pedido})
    with st.chat_message("user"):
        st.markdown(pedido)

    # Se o usuário indicou uma subpasta p/ RAG, injeta uma dica no pedido
    # (sem verbo de criação nem extensão — não afeta o roteamento de arquivo).
    tema = pedido
    if rag_pasta:
        tema += (
            f"\n\n(Para buscar em arquivos, use a 'Busca Semântica na Pasta' "
            f"apontando para: {rag_pasta})"
        )

    payload = {
        "tema": tema,
        "usar_memoria": usar_memoria,
        "contexto_extra": contexto_extra,
        "usar_gerente": usar_gerente,
    }

    with st.chat_message("assistant"):
        with st.status("Agência trabalhando…", expanded=True) as status:
            try:
                job = rodar_agencia(payload, status, timeout_job)
                entry = {
                    "papel": "assistant",
                    "conteudo": job.get("conteudo", "(sem conteúdo)"),  # COM crases
                    "conteudo_limpo": job.get("conteudo_limpo", ""),    # cru (download)
                    "tipo": job.get("tipo", "analise"),
                    "linguagem": job.get("linguagem", ""),
                    "arquivo_nome": os.path.basename(job.get("arquivo_salvo", "")) or None,
                    "payload": payload,
                }
                status.update(label="✅ Concluído!", state="complete", expanded=False)
            except RuntimeError as erro:
                entry = {
                    "papel": "assistant",
                    "conteudo": f"⚠️ {erro}",
                    "tipo": "analise",
                    "linguagem": "",
                    "arquivo_nome": None,
                    "payload": payload,
                }
                status.update(label="❌ Falhou", state="error", expanded=True)

        render_resposta(len(st.session_state.historico), entry)

    st.session_state.historico.append(entry)
