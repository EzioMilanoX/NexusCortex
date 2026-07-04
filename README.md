# Cortexus

Servidor local de múltiplos agentes de IA — 100% gratuito, 100% offline —
construído com [CrewAI](https://docs.crewai.com) (orquestração) e
[Ollama](https://ollama.com) (inferência local do LLM).

**Equipe atual:** `Pesquisador Sênior` (com busca web real e gratuita via
`ddgs`) → `Escritor Técnico` → `Revisor de Qualidade`, gerando um
relatório `.txt` em `output/`.

## Arquitetura e decisões de projeto

- **Cross-platform (Windows 11 + Linux Mint):** todo caminho de arquivo usa
  `pathlib` — nunca strings com `/` ou `\`. O código é idêntico nos dois
  sistemas; só o *venv* é separado (um por SO, ver abaixo).
- **GPU (RTX 4060, 8 GB VRAM):** o offload para CUDA **não é configurado no
  Python** — é responsabilidade do servidor Ollama, que detecta o driver
  NVIDIA automaticamente. O `llama3.1` 8B quantizado (~4,9 GB) cabe inteiro
  na VRAM, então roda 100% na GPU e deixa a CPU livre.
- **Encoding:** toda escrita de arquivo usa `encoding="utf-8"` explícito,
  senão o Windows grava em `cp1252` e os acentos quebram no Linux.
- **Busca web gratuita:** ferramenta customizada com o pacote `ddgs`
  (metabusca sem chave de API). As ferramentas de busca **na web** do
  `crewai-tools` foram descartadas de propósito: todas exigem chave de
  API (a única busca gratuita embutida, `ArxivPaperTool`, cobre apenas
  artigos do arXiv).
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
Cortexus/
├── main.py            # a agência: agentes, tarefas e orquestração
├── Modelfile          # variante llama3.1-8k (num_ctx 8192)
├── requirements.txt   # dependências Python
├── output/            # relatórios gerados (ignorado pelo git)
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
cd "C:\Users\Enzo Lai\Documents\PersonalProjects\NexusCortex(NCX)"

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
cd ~/Documentos/PersonalProjects/NexusCortex\(NCX\)   # ajuste ao seu caminho

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

### 2. Baixar o modelo e criar a variante (~4,9 GB)

```bash
ollama pull llama3.1                      # baixa o modelo base
ollama create llama3.1-8k -f Modelfile    # variante com contexto 8192
                                          # (na pasta do projeto; custo zero
                                          #  de disco — reutiliza as camadas)
```

### 3. Rodar o Cortexus

```bash
# com o venv ativado, na pasta do projeto:
python main.py
# ou com tema customizado:
python main.py Energia solar no Brasil
```

### 4. Confirmar que a inferência está na VRAM da RTX 4060

Com o modelo respondendo (durante ou logo após uma execução):

```bash
ollama ps
```

A coluna **PROCESSOR** deve mostrar **`100% GPU`**. Se aparecer algo como
`52%/48% CPU/GPU`, o modelo não coube na VRAM (ver dicas abaixo).

Confirmação cruzada:

- **Nos dois SOs:** `nvidia-smi` → o processo `ollama` deve aparecer
  usando ~5–6 GB de VRAM, e `GPU-Util` sobe durante a geração.
- **Windows:** Gerenciador de Tarefas → aba *Desempenho* → GPU →
  "Memória de GPU dedicada" ocupada enquanto o modelo está carregado.
- **Sintoma de CPU:** se os 28 núcleos baterem 100% durante a geração e a
  GPU ficar parada, o offload falhou — verifique o driver com `nvidia-smi`.

**Se o offload for parcial (modelo não coube nos 8 GB):**

- Reduza a janela de contexto (o cache KV também mora na VRAM). No
  `main.py`, a chamada `LLM(...)` do CrewAI repassa parâmetros extras ao
  Ollama — comece com o padrão e só ajuste se necessário.
- Ou use uma quantização menor: `ollama pull llama3.1:8b-instruct-q4_0`.

## Roadmap

- [x] Busca web real para o Pesquisador (via `ddgs`, sem chave de API —
      o `crewai[tools]` foi descartado: nenhuma busca na web embutida é
      gratuita).
- [x] Terceiro agente revisor (QA) antes da gravação do relatório.
- [ ] Expor a agência como API local (FastAPI) para virar um "servidor"
      de agentes de verdade.
- [ ] Se o formato ReAct do llama3.1 se mostrar instável com ferramentas:
      testar o prefixo `ollama_chat/llama3.1` ou trocar o Pesquisador para
      um modelo qwen2.5-instruct (reportado como mais confiável com tools).
