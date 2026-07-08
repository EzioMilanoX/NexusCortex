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

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
TIMEOUT_JOB_S = 600          # teto de espera por um job (10 min)
INTERVALO_POLL_S = 2.0       # de quanto em quanto tempo consulta /status
# Teto do contexto anexado: o modelo roda com num_ctx 8192, e o prompt já
# carrega backstory + esquemas de ferramentas. ~8000 caracteres (~2-2,5k
# tokens) deixa folga; acima disso, truncamos para não estourar o contexto.
MAX_CONTEXTO_CHARS = 8000
EXTS_UPLOAD = ["txt", "py", "md", "csv"]

# mime por extensão só para o download ficar "certinho" no navegador
_MIME = {".py": "text/x-python", ".json": "application/json", ".csv": "text/csv",
         ".md": "text/markdown", ".txt": "text/plain"}

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


# --- (1) Barra lateral: memória + (2) upload de arquivos para contexto ---
with st.sidebar:
    st.header("⚙️ Controles")
    usar_memoria = st.toggle(
        "Usar Memória de Longo Prazo",
        value=False,
        help=(
            "Liga a memória (ChromaDB) entre execuções. Desligada por padrão: "
            "com o modelo local, a memória às vezes faz a agência repetir a "
            "saída de uma tarefa anterior parecida (contaminação)."
        ),
    )
    st.caption("🧠 Memória: **ligada**" if usar_memoria else "💨 Memória: **desligada**")

    usar_gerente = st.toggle(
        "Modo Gerente (Hierárquico)",
        value=False,
        help=(
            "Um Gerente (LLM) convoca dinamicamente os especialistas (incl. "
            "Arquiteto e Especialista em Integrações). Mais flexível, porém "
            "BEM mais lento no modelo local e sem a rede de segurança que "
            "recupera o código. Desligado = pipeline sequencial confiável."
        ),
    )
    st.caption("👔 Gerente: **ativo**" if usar_gerente else "➡️ Modo: **sequencial**")

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


def api_no_ar() -> bool:
    """True se a API responde — evita erros crípticos se o api.py não subiu."""
    try:
        requests.get(f"{API_BASE}/docs", timeout=3)
        return True
    except requests.RequestException:
        return False


def rodar_agencia(payload: dict, status) -> dict:
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
    rotulos = {
        "na_fila": "⏳ Na fila…",
        "executando": "🧠 Pesquisando → redigindo → revisando…",
    }
    ultimo_estado = None
    limite = time.monotonic() + TIMEOUT_JOB_S
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
    raise RuntimeError(f"O job passou de {TIMEOUT_JOB_S}s sem concluir.")


def render_resposta(idx: int, entry: dict) -> None:
    """Desenha a resposta do assistente: (2) abas Análise/Código, (5) markdown
    com highlighting, (3) botão de download e (6) expander do payload n8n.
    Chamado tanto ao vivo quanto ao redesenhar o histórico — por isso a
    'key' do download é indexada (widgets repetidos precisam de key única).

    'conteudo' já vem do back-end COM as crases de markdown (```lang) quando é
    código, então basta injetá-lo via st.markdown — o Streamlit colore sozinho.
    O download usa 'conteudo_limpo' (cru, sem crases = arquivo executável)."""
    tipo = entry.get("tipo", "analise")
    conteudo = entry.get("conteudo", "")

    aba_chat, aba_codigo = st.tabs(["💬 Análise", "💻 Código Gerado"])
    with aba_chat:
        if tipo == "analise":
            st.markdown(conteudo)  # (5) prosa/markdown renderizada
        else:
            st.markdown("_A resposta é código — veja a aba **💻 Código Gerado**._")
    with aba_codigo:
        if tipo == "codigo":
            st.markdown(conteudo)  # (5) já vem cercado com ```lang → colorido
        else:
            st.markdown("_Nenhum código gerado nesta resposta._")

    # (3) Download em vez do caminho cru: baixa o conteúdo LIMPO (== arquivo)
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

    # (6) Inspecionador "n8n Ready": o payload exato que o ui.py enviou
    with st.expander("🛠️ Ver Payload da API (Para n8n)"):
        st.json(entry.get("payload", {}))


# --- Estado do chat (persiste entre interações da mesma sessão) ---
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

    # (1) memória + (upload) contexto + modo gerente (hierárquico)
    payload = {
        "tema": pedido,
        "usar_memoria": usar_memoria,
        "contexto_extra": contexto_extra,
        "usar_gerente": usar_gerente,
    }

    with st.chat_message("assistant"):
        # (4) st.status no lugar do spinner, com etapas
        with st.status("Agência trabalhando…", expanded=True) as status:
            try:
                job = rodar_agencia(payload, status)
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
