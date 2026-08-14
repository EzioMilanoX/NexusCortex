# NexusCortex(NCX)

Servidor local de múltiplos agentes de IA — 100% gratuito, 100% offline —
construído com [CrewAI](https://docs.crewai.com) (orquestração) e
[Ollama](https://ollama.com) (inferência local do LLM).

> 📖 **Novo por aqui?** Vá direto ao [TUTORIAL.md](TUTORIAL.md) — guia
> passo a passo de instalação e dos jeitos de usar. Rode os testes com `pytest`.

**Equipe atual:** `Pesquisador e Coletor de Dados` (busca web via `ddgs` +
leitura de disco) → `Especialista em Resolução` (escreve código/prosa e tem
terminal sandboxed) → `Engenheiro de Qualidade`, gerando o arquivo final em
`output/`. Roda pelo terminal (`python main.py`) ou como **API HTTP** para o
n8n (`python api.py`).

**Capacidades por nível:** (1) lê pastas/arquivos locais; (2) memória de
longo prazo via ChromaDB + embeddings locais; (3) API FastAPI; (4) API
assíncrona (não trava o n8n) + agente com terminal sandboxed; (5) painel
Streamlit + endpoint OpenAI-compatível; (6) **Estúdio de Podcast** — a mesma
equipe escreve o roteiro, duas vozes pt-BR o narram e o programa é montado em
MP3, tudo offline (`http://localhost:8000/estudio`).

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
- **A busca LÊ as páginas sozinha (`busca_web` + `_ler_fontes`):** antes ela
  devolvia só título, link e um resumo curto — o agente nunca via o TEXTO de
  nada. Diagnosticado num caso real: um pedido de podcast sobre "teoria dos
  jogos" recebeu da busca três vídeos do YouTube, e o modelo, sem nada em que
  se apoiar, escreveu de memória — definiu equilíbrio de Nash ao contrário e
  inventou exemplos. **Não era limite do 7B, era falta de acesso ao conteúdo.**

  A primeira tentativa foi dar ao Pesquisador uma ferramenta `ler_pagina_web`
  e instruí-lo, em maiúsculas, a usá-la depois de buscar. **Não funcionou:** o
  log de um teste completo registrou UMA execução de ferramenta
  (`busca_na_web`) e nenhuma leitura — o qwen 7B simplesmente não a chamava, a
  mesma limitação já documentada do terminal, que o Escritor também ignora.
  Enquanto a leitura era uma *decisão do modelo*, ela não acontecia.

  A correção foi **fundir busca e leitura numa ferramenta só**: a `busca_web`
  ordena os resultados por credibilidade (enciclopédia e domínio institucional
  antes de blog), descarta o que não tem prosa (vídeo, rede social, PDF), abre
  as duas melhores e devolve o texto delas junto dos resultados. A decisão
  deixa de existir — o agente pede uma busca e recebe prosa, queira ou não.
  Mesma escolha de projeto do `escolher_saida` e do `parse_roteiro`: passo
  essencial mora no código, não na obediência do modelo. A `ler_pagina_web`
  continua disponível para conferir uma página específica.
- **Invólucro do raspador (`_enxugar_pagina`):** o motor é a
  `ScrapeWebsiteTool` do `crewai-tools`, gratuita e **sem chave** (é scraper,
  não busca), mas crua ela devolve o menu de navegação e não tem teto de
  tamanho. O filtro é **pontuação de fim de frase**, não tamanho de linha — a
  primeira versão filtrava linhas curtas e deixava passar o pior lixo, porque
  o scraper junta o menu inteiro numa linha longa e a lista de idiomas da
  Wikipédia noutra. Na Wikipédia isso corta 72% e o que sobra é prosa. Efeito
  desejado: página de vídeo não tem uma frase, então a ferramenta avisa que
  não há texto legível em vez de entregar "About Press Copyright" como fonte.
- **Sandbox de URL (`_url_segura`):** só `http`/`https`, e o host é resolvido
  para recusar loopback, rede privada, link-local e multicast. Não é enfeite:
  o Pesquisador processa texto **não confiável** da web aberta, então uma
  injeção de prompt poderia (a) usar `file:///C:/...` para contornar o
  confinamento a `BASE_DIR` das ferramentas de disco, ou (b) fazer SSRF contra
  `localhost:11434` (o Ollama) e a própria API do Cortexus, que roda sem
  autenticação.
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
├── api.py             # API HTTP: /agencia (n8n), /podcast, /v1 OpenAI-compat
├── podcast.py         # Estúdio de Podcast: roteiro -> vozes -> programa final
├── ui.py              # painel visual em Streamlit (conversa com a api.py)
├── web/estudio.html   # painel do Estúdio de Podcast (servido em /estudio)
├── cortexus.bat       # launcher de um clique: API + painel (Windows)
├── cortexus.sh        # launcher de um clique: API + painel (Linux Mint)
├── estudio.bat        # launcher só do Estúdio de Podcast (Windows)
├── estudio.sh         # launcher só do Estúdio de Podcast (Linux Mint)
├── .streamlit/        # config do painel (desliga telemetria/prompt de e-mail)
├── Modelfile          # variante llama3.1-8k (legado)
├── Modelfile.qwen     # variante qwen2.5-8k (num_ctx 8192) — modelo atual
├── requirements.txt   # dependências Python
├── tests/             # suíte pytest (hermética; `pytest` não exige Ollama)
├── vozes/             # modelos .onnx do Piper (ignorado pelo git, ~63 MB cada)
├── output/            # arquivos gerados (ignorado pelo git)
│   └── podcast/       # episódios: programa.wav, programa.mp3, roteiro.json
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
>
> E um atalho só para o **Estúdio de Podcast** (mesma receita, outro alvo):
> ```powershell
> $ws = New-Object -ComObject WScript.Shell
> $sc = $ws.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Cortexus Estudio.lnk")
> $sc.TargetPath = "$PWD\estudio.bat"; $sc.WorkingDirectory = "$PWD"
> $sc.IconLocation = "$env:SystemRoot\System32\shell32.dll,138"; $sc.Save()
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

### Estúdio de Podcast (`/estudio`)

**O jeito fácil:** duplo-clique em **`estudio.bat`** (Windows) ou
`./estudio.sh` (Linux Mint) — sobe a API e abre a página sozinho. Ele não
carrega o Streamlit, então é mais leve e rápido que o `cortexus.bat`. Se o
painel da agência já estiver no ar, a página também está: é a **mesma API**
que serve as duas, e há um botão para ela na barra lateral do painel.

Manualmente: com a `api.py` rodando, abra **<http://localhost:8000/estudio>**.

A mesma equipe que escreve relatórios escreve o roteiro; duas vozes pt-BR o
narram; o programa sai montado em MP3. Tudo offline.

O fluxo é deliberadamente em **duas etapas**: “Escrever roteiro com a IA”
devolve o texto para você **editar na tela**, e só o segundo botão gasta os
minutos de síntese. A crew erra o tom de vez em quando e refazer texto é
grátis — refazer áudio não é. Você também pode pular a primeira etapa e colar
um roteiro pronto: **esse caminho não precisa do Ollama**.

```bash
# 1. Instalar as vozes pt-BR (uma vez, ~126 MB)
python -m piper.download_voices pt_BR-faber-medium pt_BR-cadu-medium \
       --download-dir vozes

# 2. Usar pelo terminal, sem servidor nenhum
python podcast.py roteiro.txt --motor piper --titulo "Episódio 1"
python podcast.py roteiro.txt --motor tom      # demonstração, sem baixar voz
```

**Decisões de projeto (medidas nesta máquina, não copiadas da documentação):**

- **Piper, e não Bark/XTTS/Tortoise.** O `qwen2.5-8k` já ocupa ~5 dos 8 GB da
  RTX 4060 (medido com `ollama ps`: `100% GPU`); os modelos de voz “pesados”
  pedem outros 2-8 GB e **não cabem junto** — seria preciso descarregar o
  Ollama a cada episódio, ou aceitar o TTS na CPU a passo de tartaruga. O
  Piper roda em ONNX **na CPU**, sem tocar na VRAM, e o `onnxruntime` já vinha
  instalado como dependência do chromadb. Medido de ponta a ponta com as vozes
  padrão (síntese + EQ + mixagem + encode MP3): **2 min 17 s de episódio em
  96 s**, ~1,4× mais rápido que tempo real. Vozes de qualidade `low` chegam a
  4×, mas veja o item seguinte antes de trocar por elas.
- **Vozes pt-BR: use `faber` + `cadu`.** O catálogo tem quatro, e a
  `pt_BR-edresson-low` foi publicada **sem o til combinante (U+0303) no
  `phoneme_id_map`** — ela emite `Missing phoneme from id map` e come a
  nasalização (“não” vira “nao”). Verificado nos quatro arquivos `.onnx.json`.
  As outras três nasalizam corretamente. A dupla padrão é `faber` (F0 ≈ 165 Hz)
  + `cadu` (F0 ≈ 138 Hz), o par de maior distância de timbre disponível — as
  quatro vozes são masculinas, não há voz feminina pt-BR no Piper hoje.
- **Escrita em blocos, não num pedido só.** Pedir "cerca de 40 falas, ~750
  palavras" rendeu **11 falas** (medido, alvo de 5 minutos). Modelo pequeno não
  sustenta pedido longo, insistir no prompt não conserta e pedir em loop traz
  de volta a repetição literal. Então trocamos UM pedido grande por N pequenos,
  orquestrados em Python: a crew pesquisa e devolve uma **pauta** (blocos +
  fatos verificados com fonte); depois **um passe curto por bloco** pede só
  ~13 falas sobre aquele tópico. A pesquisa roda uma vez e alimenta todos os
  blocos; os blocos são chamadas diretas (`llm.call`), sem crew, porque não
  precisam de ferramenta nenhuma. A costura é determinística (concatenar →
  `parse_roteiro` → `_remover_repeticoes`), cada bloco recebe as duas últimas
  falas para não recomeçar com "Olá, bem-vindos!" no meio, e um bloco que
  estoura não derruba o episódio. De brinde, o estúdio mostra "bloco 3 de 5"
  em vez de uma barra parada. O modo antigo continua disponível em
  `{"modo": "unico"}`.
- **JSON gerado por código, nunca pelo modelo.** O roteiro é pedido no formato
  `Nome: texto` (que qualquer modelo acerta) e convertido em Python por
  `parse_roteiro`. Um roteiro de 15 minutos tem ~80 falas: pedir JSON ao qwen
  7B no caminho ReAct significaria perder o job inteiro por uma vírgula. O
  `roteiro.json` continua existindo — ele nasce das dataclasses já validadas,
  então é sempre bem formado. O parser sobrevive a cerca de markdown,
  `Final Answer:`, rubricas `(risos)` e até a um JSON que o modelo mande por
  conta própria.
- **Mixagem em numpy + `wave` da stdlib, não pydub.** O pydub depende do
  módulo `audioop`, **removido no Python 3.13** (PEP 594), e arrastaria o
  FFmpeg como dependência obrigatória. Aqui o FFmpeg é **opcional** e só faz o
  encode final em MP3 — sem ele, o episódio sai em WAV normalmente.
- **Pós-produção:** EQ de voz por fala (passa-alta em 85 Hz + realce de
  presença em 3 kHz, via FFT), fade de 8 ms em cada emenda para não estalar,
  pausa maior quando **troca** de apresentador, trilha em loop que entra alta
  na abertura e **abaixa por crossfade** sob a fala, e normalização de pico.

**Endpoints** (todos reaproveitam o registro de jobs e o `GET /status/{id}`):

| Rota | O que faz |
|---|---|
| `POST /podcast/roteiro` | só escreve o roteiro (exige Ollama) |
| `POST /podcast` | `{"roteiro": ...}` ou `{"tema": ...}` → episódio |
| `GET /podcast/{job_id}/audio` | o MP3 (ou WAV) pronto |
| `GET /podcast/vozes` | catálogo pt-BR e quais estão em disco |
| `GET /podcast/ambiente` | diagnóstico de Ollama e FFmpeg |

> **Trilha de fundo:** passe o caminho de um `.wav` **dentro da pasta do
> projeto**. Caminhos fora dela são recusados de propósito — a API escuta em
> `0.0.0.0` sem autenticação, e um caminho arbitrário no corpo do POST não
> pode virar leitura de qualquer arquivo do disco.

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
- [x] Estúdio de Podcast (Nível 6): `podcast.py` + `/estudio`, com roteiro
      editável antes da síntese, vozes pt-BR do Piper na CPU e pós-produção
      (EQ, pausas, trilha com duck, MP3).
- [ ] Trilha de fundo pronta no repo: hoje é preciso apontar um `.wav` seu.
      Falta escolher uma faixa com licença livre para embarcar.
- [x] Duração e tamanho de fala: resolvido pela escrita em blocos. Medido no
      mesmo tema ("teoria dos jogos", alvo de 5 min): **11 → 45 falas** e
      **250 → 125 caracteres** por fala (não são mais monólogos de 15 s).
- [x] Alucinação factual: resolvida pela busca que lê as fontes. No mesmo
      tema, o roteiro passou a citar von Neumann, Morgenstern, o livro de 1944,
      o teorema minimax e a carta de Waldegrave de 1713 — tudo rastreável à
      pauta. Antes: equilíbrio de Nash definido ao contrário e exemplos
      inventados ("jogo do lembretista").
- [x] **Qualidade das fontes.** Agregadores sem revisão (`prezi`, `slideshare`,
      `scribd`, `quora`…) nunca são lidos; o resto é lido em ordem de
      credibilidade e **rotulado** como `[FONTE DE REFERÊNCIA]` (enciclopédia,
      `.gov`, `.edu`, documentação) ou `[FONTE SECUNDÁRIA]`, com a regra de que
      datas, nomes e definições saem da primeira. Motivo: o único erro factual
      que sobreviveu ao teste ("equilíbrio de Nash formulado no início do
      século XX"; foi 1950-51) veio de um blog de consultoria lido lado a lado
      com a Wikipédia — o modelo não tinha como saber qual das duas pesava
      mais. Sem fonte de referência entre os resultados, o bloco avisa que os
      números são incertos.
- [x] **Calibração de duração.** O modelo entrega ~15% mais falas do que se
      pede (medido: 51, 44 e 45 contra 39 pedidas), então o pedido sai
      multiplicado por `CALIBRACAO_FALAS = 0.85` e o ritmo real das vozes foi
      medido em 143 palavras/minuto (era estimado em 150). Calibração empírica
      de três amostras — trocando de modelo, meça de novo.
- [x] **Emenda entre blocos.** Só o último bloco fecha o episódio; os demais
      são proibidos de usar frase de gancho ou anunciar o que vem a seguir, e
      terminam no meio do assunto.
- [ ] O modelo copia o rótulo do exemplo da pauta como título do bloco
      ("1. O que é e de onde veio | definição exata..."). Inofensivo desde que
      `dividir_pauta` anexe os fatos da linha seguinte, mas é cosmética suja.
- [ ] Voz feminina em pt-BR: não existe no catálogo do Piper. Depende de
      avaliar o XTTS-v2 (VRAM disputada com o qwen + licença CPML
      não-comercial) ou treinar uma voz.
- [ ] Autenticação (token) na API se for exposta além do `127.0.0.1`.
- [ ] O agente Escritor nem sempre aciona o terminal para auto-testar o
      código (limitação do modelo 7B) — avaliar um passo de verificação
      determinístico pós-geração para scripts `.py`.
- [ ] Integrações do ecossistema (GitHub etc.): o `GithubSearchTool` é RAG
      e exige token + embedder local — avaliar quando houver necessidade.
