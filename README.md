# NexusCortex(NCX)

Servidor local de múltiplos agentes de IA — 100% gratuito, 100% offline —
construído com [CrewAI](https://docs.crewai.com) (orquestração) e
[Ollama](https://ollama.com) (inferência local do LLM).

**Equipe atual:** `Pesquisador e Coletor de Dados` (busca web via `ddgs` +
leitura de disco) → `Especialista em Resolução` (escreve código/prosa e tem
terminal sandboxed) → `Engenheiro de Qualidade`, gerando o arquivo final em
`output/`. Roda pelo terminal (`python main.py`) ou como **API HTTP** para o
n8n (`python api.py`).

**Capacidades por nível:** (1) lê pastas/arquivos locais; (2) memória de
longo prazo via ChromaDB + embeddings locais; (3) API FastAPI; (4) API
assíncrona (não trava o n8n) + agente com terminal sandboxed.

## Arquitetura e decisões de projeto

- **Cross-platform (Windows 11 + Linux Mint):** todo caminho de arquivo usa
  `pathlib` — nunca strings com `/` ou `\`. O código é idêntico nos dois
  sistemas; só o *venv* é separado (um por SO, ver abaixo).
- **Modelo (`qwen2.5` 7B):** trocado do `llama3.1` 8B no Nível 4 — é o
  modelo local ~7B que a comunidade CrewAI reporta como o tool-caller mais
  confiável, com melhor pt-BR. O **32B pedido não cabe** na RTX 4060 (~20 GB
  de pesos → transborda para a CPU a ~3-6 tok/s = horas por job); o 14B
  forçaria split GPU/CPU. O 7B a 8k usa ~6 GB e roda 100% na GPU.
- **GPU (RTX 4060, 8 GB VRAM):** o offload para CUDA **não é configurado no
  Python** — é responsabilidade do servidor Ollama, que detecta o driver
  NVIDIA automaticamente. O `qwen2.5` 7B quantizado (~4,7 GB) cabe inteiro
  na VRAM, então roda 100% na GPU e deixa a CPU livre.
- **Memória de longo prazo (Nível 2), agora opt-in:** `Crew(memory=...)` com
  embeddings **locais** (`nomic-embed-text` via Ollama) — nunca bate na API
  da OpenAI. O armazenamento (ChromaDB) fica em `%LOCALAPPDATA%\crewai\`
  (fora do repo). **Desligada por padrão:** o toggle "Usar Memória de Longo
  Prazo" no painel a liga por execução. O motivo do default off: com o
  modelo 7B, a memória às vezes faz a agência recuperar e repetir a saída de
  uma tarefa anterior parecida (contaminação) — ligue só quando o histórico
  entre execuções agregar valor.
- **Tool-calling forçado a ReAct (fix crítico do Nível 4):** com memória
  ligada, o CrewAI usaria o executor de function-calling NATIVO, e nesse
  caminho o qwen2.5 devolve resposta vazia quando o prompt cresce
  (`Invalid response from LLM call - None or empty`) — reproduzido 2/2. A
  fábrica `criar_llm()` sobrescreve `supports_function_calling` na instância
  do LLM para False, mantendo o laço ReAct (texto puro, higienizado por
  `limpar_saida`). Não dá para subclassar: `LLM.__new__` é uma fábrica que
  ignora subclasses.
- **Agente com terminal (Nível 4):** o Escritor tem `executar_comando`, um
  sandbox rígido (`shell=False`, allowlist de executáveis, denylist de
  flags perigosas do git, confinado ao projeto, timeout 120s, `stdin`
  fechado). Só `python <script no projeto>`, `git status|diff|log|add|
  commit|show|branch` e `pytest`.
- **Encoding:** toda escrita de arquivo usa `encoding="utf-8"` explícito,
  senão o Windows grava em `cp1252` e os acentos quebram no Linux.
- **Busca web gratuita:** ferramenta customizada com o pacote `ddgs`
  (metabusca sem chave de API). As ferramentas de busca **na web** do
  `crewai-tools` foram descartadas de propósito: todas exigem chave de
  API (a única busca gratuita embutida, `ArxivPaperTool`, cobre apenas
  artigos do arXiv).
- **Busca semântica local (RAG):** o Pesquisador tem a `DirectorySearchTool`
  para achar trechos relevantes em arquivos grandes/base de código sem ler
  tudo (o que estouraria o contexto de 8k). Usa os embeddings **locais**
  `nomic-embed-text` (via `CONFIG_EMBEDDER`, o mesmo da memória) — nunca a
  OpenAI. Indexa a pasta recursivamente, então aponte para uma **subpasta
  específica**, nunca a raiz do projeto (que contém `.venv`).
- **Contexto do modelo:** o projeto usa a variante local `llama3.1-8k`
  (criada do [Modelfile](Modelfile) com `num_ctx 8192`) — o padrão do
  Ollama (4096) trunca silenciosamente o começo do prompt, quebrando as
  instruções de ferramenta do CrewAI. Passar `num_ctx` pelo `LLM(...)` do
  CrewAI 1.15.1 **não funciona** (erro `Completions.create() ... 'num_ctx'`,
  verificado empiricamente), por isso o parâmetro é assado no modelo.
  8192 tokens cabem nos 8 GB de VRAM (~5,8–6 GB no total, medido); acima
  de ~12288 o modelo tende a transbordar para a CPU.

## Estrutura

```
NexusCortex(NCX)/
├── main.py            # a agência: agentes, tarefas, ferramentas, orquestração
├── api.py             # API HTTP: /agencia (n8n) + /v1 OpenAI-compat (editores)
├── ui.py              # painel visual em Streamlit (conversa com a api.py)
├── cortexus.bat       # launcher de um clique (Windows)
├── cortexus.sh        # launcher de um clique (Linux Mint)
├── .streamlit/        # config do painel (desliga telemetria/prompt de e-mail)
├── Modelfile          # variante llama3.1-8k (legado)
├── Modelfile.qwen     # variante qwen2.5-8k (num_ctx 8192) — modelo atual
├── requirements.txt   # dependências Python
├── output/            # arquivos gerados (ignorado pelo git)
├── .gitignore
└── README.md
```

## Setup do ambiente (hoje — sem baixar modelos)

> **Versão do Python:** o CrewAI (1.15.1, verificado em jul/2026) exige
> Python **>=3.10 e <3.14** — o Python 3.14 (que está instalado no Windows
> desta máquina) **não é suportado**: o pip se recusa a instalar o crewai
> nele. Instale o 3.12 ao lado (não precisa remover o 3.14).
>
> **Espaço em disco:** a instalação das dependências ocupa **~700 MB**
> no venv (por SO). Os modelos do Ollama (a parte pesada, ~5 GB+) ficam
> para quando o novo SSD chegar.

### Windows 11 (PowerShell)

```powershell
# 0. (uma vez) Instalar Python 3.12 ao lado do 3.14
winget install Python.Python.3.12

# 1. Entrar na pasta do projeto

# 2. Criar o venv com o Python 3.12 (o launcher 'py' escolhe a versão)
py -3.12 -m venv .venv

# 3. Ativar o venv
#    (se der erro de execução de script, rode uma única vez:
#     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)
.\.venv\Scripts\Activate.ps1

# 4. Instalar as dependências
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux Mint (Bash)

```bash
# 0. (uma vez) Garantir venv/pip — o Mint não traz por padrão
sudo apt update && sudo apt install -y python3-venv python3-pip

# 1. Entrar na pasta do projeto

# 2. Criar o venv (separado do venv do Windows — eles NÃO são portáveis
#    entre SOs; o .gitignore já exclui os dois)
python3 -m venv .venv

# 3. Ativar o venv
source .venv/bin/activate

# 4. Instalar as dependências
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Dica dual-boot:** se a pasta do projeto vive numa partição NTFS
> compartilhada, mantenha os venvs fora dela ou com nomes distintos
> (ex.: `.venv-win` e `.venv-linux`) — binários de um SO não rodam no outro.

## Guia futuro — quando o SSD chegar

### 1. Instalar o Ollama

- **Windows:** instalador em <https://ollama.com/download> (já embute o
  runtime CUDA; basta ter o driver NVIDIA atualizado).
- **Linux Mint:** primeiro ative o driver proprietário NVIDIA no
  *Gerenciador de Drivers* (teste com `nvidia-smi`), depois:

  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

### 2. Baixar os modelos e criar a variante (~5 GB)

```bash
ollama pull qwen2.5:7b                        # modelo de chat (~4,7 GB)
ollama create qwen2.5-8k -f Modelfile.qwen    # variante contexto 8192
                                              # (custo zero de disco)
ollama pull nomic-embed-text                  # embeddings da memória (~274 MB)
```

### 3. Rodar o Cortexus

Pelo terminal:
```bash
# com o venv ativado, na pasta do projeto:
python main.py
python main.py Energia solar no Brasil                  # tema customizado
python main.py Crie um script da hora e salve como relogio.py   # gera código
```

Ou como API HTTP (para o n8n / Postman):
```bash
python api.py                    # sobe em http://0.0.0.0:8000
```
- `POST /agencia` com `{"tema": "...", "webhook_url": "http://... (opcional)"}`
  responde **na hora** com um `job_id` (não trava o n8n); a equipe roda em
  background e, ao terminar, dispara um POST no `webhook_url`.
- `GET /status/{job_id}` acompanha por polling; quando `status` é
  `concluido`, traz `arquivo_salvo` e `conteudo`.
- **Segurança/limites:** `host="0.0.0.0"` expõe na rede local sem
  autenticação — se o n8n roda na mesma máquina, prefira `127.0.0.1`. Rode
  com **um único worker** (o registro de jobs é em memória); reiniciar o
  servidor perde os jobs em andamento.

### Atalho de um clique (o jeito mais fácil)

Há um atalho **Cortexus** na Área de Trabalho (Windows) apontando para
`cortexus.bat`. Um duplo-clique sobe a API e o painel visual em janelas
próprias e abre o navegador quando tudo estiver pronto — feche as duas
janelas para desligar. No **Linux Mint**, o equivalente é `./cortexus.sh`
(rode `chmod +x cortexus.sh` uma vez). Ambos exigem o Ollama já rodando e
o `.venv` criado.

> Recriar o atalho do zero (PowerShell), se precisar:
> ```powershell
> $ws = New-Object -ComObject WScript.Shell
> $sc = $ws.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Cortexus.lnk")
> $sc.TargetPath = "$PWD\cortexus.bat"; $sc.WorkingDirectory = "$PWD"
> $sc.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"; $sc.Save()
> ```

### Painel visual (Streamlit) — manualmente

Se preferir subir na mão, dois processos com o venv ativado:
```bash
python api.py            # 1) sobe a API (o painel conversa com ela)
streamlit run ui.py      # 2) sobe o painel (abre no navegador)
```
No VS Code, `Ctrl+Shift+P` → **Simple Browser: Show** → cole a URL do
Streamlit (`http://localhost:8501`) para ficar lado a lado com o código.
O painel oferece: toggle de **memória** na barra lateral; **abas** que
separam a análise (prosa) do código gerado (com syntax highlighting);
**botão de download** do arquivo; feedback de progresso via `st.status`; e
um expander **"Ver Payload da API (Para n8n)"** com o JSON exato enviado —
pronto para colar num nó do n8n.

### Dentro do VS Code (Continue.dev / Cline) — endpoint OpenAI-compatível

A API expõe `/v1/chat/completions` e `/v1/models` imitando a OpenAI, então
extensões de IA do VS Code conversam com a agência inteira pela barra
lateral. Com o `python api.py` no ar, configure a extensão apontando para
**`http://localhost:8000/v1`** (com `/v1`, sem barra no fim):

```yaml
# Continue.dev (config.yaml) — provider "openai" + apiBase; a chave é fake
models:
  - name: Cortexus
    provider: openai
    model: cortexus-crew
    apiBase: http://localhost:8000/v1
    apiKey: sk-local
```

- Cada mensagem roda a **crew inteira** (Pesquisador → Escritor → Revisor),
  então a resposta leva minutos. A API abre o stream na hora e manda
  keepalives para o editor não dar timeout — mas **aumente o timeout** da
  extensão acima do pior caso da crew.
- É uma fachada **de chat/análise**: `tools`/`tool_choice` são ignorados, ou
  seja, o *modo agente* do Cline (que edita arquivos via tool-use nativo)
  não é suportado por aqui — para automação, use `POST /agencia`.
- **Cursor não funciona** com localhost (faz proxy pela nuvem dele); o alvo
  são Continue.dev e Cline.

### 4. Confirmar que a inferência está na VRAM da RTX 4060

Com o modelo respondendo (durante ou logo após uma execução):

```bash
ollama ps
```

A coluna **PROCESSOR** deve mostrar **`100% GPU`**. Se aparecer algo como
`52%/48% CPU/GPU`, o modelo não coube na VRAM (ver dicas abaixo).

Confirmação cruzada:

- **Nos dois SOs:** `nvidia-smi` → o processo `ollama` deve aparecer
  usando ~5 GB de VRAM, e `GPU-Util` sobe durante a geração.
- **Windows:** Gerenciador de Tarefas → aba *Desempenho* → GPU →
  "Memória de GPU dedicada" ocupada enquanto o modelo está carregado.
- **Sintoma de CPU:** se os 28 núcleos baterem 100% durante a geração e a
  GPU ficar parada, o offload falhou — verifique o driver com `nvidia-smi`.

**Se o offload for parcial (modelo não coube nos 8 GB):**

- Reduza a janela de contexto (o cache KV também mora na VRAM). No
  `main.py`, a chamada `LLM(...)` do CrewAI repassa parâmetros extras ao
  Ollama — comece com o padrão e só ajuste se necessário.
- Ou use uma quantização menor: `ollama pull qwen2.5:7b-instruct-q4_0`.

## Roadmap

- [x] Busca web real para o Pesquisador (via `ddgs`, sem chave de API).
- [x] Terceiro agente revisor (QA) antes da gravação do arquivo.
- [x] Leitura de pastas/arquivos locais (Nível 1).
- [x] Memória de longo prazo com embeddings locais (Nível 2).
- [x] API local FastAPI, assíncrona com webhook de retorno (Níveis 3-4).
- [x] Painel visual Streamlit (`ui.py`) e endpoint OpenAI-compatível `/v1`
      para Continue.dev / Cline dentro do VS Code (Nível 5).
- [x] Agente com terminal sandboxed (Nível 4).
- [x] Troca do modelo para `qwen2.5` (Nível 4).
- [ ] Autenticação (token) na API se for exposta além do `127.0.0.1`.
- [ ] O agente Escritor nem sempre aciona o terminal para auto-testar o
      código (limitação do modelo 7B) — avaliar um passo de verificação
      determinístico pós-geração para scripts `.py`.
- [ ] Integrações do ecossistema (GitHub etc.): o `GithubSearchTool` é RAG
      e exige token + embedder local — avaliar quando houver necessidade.
