# Tutorial de uso — NexusCortex (Cortexus)

Guia prático de ponta a ponta: instalar, rodar e usar a agência local de IA
(CrewAI + Ollama, 100% offline). Para as decisões de arquitetura, veja o
[README.md](README.md).

> **Nomes:** o *repositório* é **NexusCortex(NCX)**; a *aplicação/agência*
> internamente se chama **Cortexus**.

---

## 1. Pré-requisitos (uma vez)

1. **Ollama** instalado e rodando (ícone na bandeja no Windows; `nvidia-smi`
   funcionando se quiser GPU). Baixe os modelos e crie a variante de contexto:
   ```bash
   ollama pull qwen2.5:7b
   ollama create qwen2.5-8k -f Modelfile.qwen     # variante num_ctx 8192
   ollama pull nomic-embed-text                   # embeddings (memória + RAG)
   ```
2. **Ambiente Python** (Python 3.10–3.13; NÃO 3.14). Na pasta do projeto:
   - Windows (PowerShell):
     ```powershell
     py -3.12 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     python -m pip install -r requirements.txt
     ```
   - Linux Mint (Bash):
     ```bash
     sudo apt install -y python3-venv python3-pip
     python3 -m venv .venv && source .venv/bin/activate
     python -m pip install -r requirements.txt
     ```

Detalhes e solução de problemas: seção "Setup do ambiente" do README.

---

## 2. Os cinco jeitos de usar

### 2.1. Terminal (o mais direto)

```bash
# com o venv ativado, na pasta do projeto:
python main.py Explique em 3 frases o que é um webhook
python main.py Crie um script que liste os pares de 1 a 20 e salve como pares.py
python main.py Leia o README.md e faça um resumo
```
Tudo depois de `main.py` é o **pedido**. O resultado é impresso e salvo em
`output/` — com a extensão certa (`.py`/`.sh`/… para código; `.txt` para
prosa), deduzida do que você pediu ou do conteúdo gerado.

### 2.2. Painel visual (Streamlit) — recomendado no dia a dia

Atalho de um clique: **duplo-clique em `Cortexus` na Área de Trabalho**
(Windows) ou `./cortexus.sh` (Linux). Ele sobe a API + o painel e abre o
navegador. Manualmente, em dois terminais:
```bash
python api.py            # 1) sobe a API
streamlit run ui.py      # 2) sobe o painel
```
No painel (barra lateral) você controla:
- **Modo de execução:** *Sequencial* (rápido, confiável) ou *Gerente*
  (hierárquico, mais flexível e bem mais lento).
- **Memória de Longo Prazo:** liga/desliga (padrão desligada).
- **Timeout do job** e **Pasta p/ Busca Semântica** (RAG) opcional.
- **Anexar contexto:** arraste `.txt/.py/.md/.csv` — o texto entra no pedido.
- **Arquivos gerados:** navegue, veja o preview e baixe o que a agência criou.

A resposta aparece em abas (**Análise** / **Código Gerado**, com cores), com
botão de download e um expander **"Ver Payload da API (Para n8n)"**.

### 2.3. API HTTP (para o n8n / automação)

Com `python api.py` no ar (escuta em `http://127.0.0.1:8000` — só esta
máquina; `--port`/`NCX_PORT` muda a porta e `--reload` liga o recarregamento
para desenvolvimento):
- `POST /agencia` com `{"tema": "...", "webhook_url": "http://... (opcional)"}`
  responde **na hora** com um `job_id` (não trava o n8n); a equipe roda no
  fundo e, ao terminar, dispara um POST no `webhook_url`.
- `GET /status/{job_id}` acompanha por polling; ao concluir, traz `conteudo`,
  `conteudo_limpo`, `arquivo_salvo`, `tipo` e `linguagem`.

Exemplo:
```bash
curl -X POST http://localhost:8000/agencia \
  -H "Content-Type: application/json" \
  -d '{"tema": "Pesquise novidades do FastAPI e resuma"}'
```

**Acessar de outro dispositivo (celular, outro PC, n8n em outra máquina):**
por padrão a API **não** aceita conexões de fora — ela não tem senha e o
`/agencia` comanda um agente com acesso ao terminal. Se precisar, abra de
propósito e só numa rede confiável:
```bash
python api.py --host 0.0.0.0          # ou: NCX_HOST=0.0.0.0 python api.py
```
O servidor imprime um aviso no boot; depois acesse por
`http://<IP-desta-máquina>:8000`. Os atalhos (`cortexus.bat`/`estudio.bat` e
os `.sh`) sobem sempre em `127.0.0.1`.

### 2.4. Dentro do VS Code (Continue.dev / Cline)

A API imita a OpenAI em `/v1`. Configure a extensão apontando para
**`http://localhost:8000/v1`** (com `/v1`, sem barra no fim). No Continue,
o `config.yaml` (schema v1) usa **`name:`** (não `title:`):
```yaml
models:
  - name: Cortexus (Local)
    provider: openai
    model: cortexus-crew
    apiKey: qualquer-coisa      # chave dummy; local ignora
    apiBase: http://localhost:8000/v1
    roles: [chat]
```
Cada mensagem roda a crew inteira (minutos) — aumente o timeout da extensão.
É fachada de **chat/análise**; o modo *Agente* (tool-use nativo) não passa
por aqui. **Cursor não funciona** com localhost (faz proxy pela nuvem).

### 2.5. Estúdio de Podcast (`/estudio`)

Transforma um tema (ou um roteiro seu) em um episódio narrado por duas vozes.

**Instalar as vozes, uma vez (~126 MB):**
```bash
python -m piper.download_voices pt_BR-faber-medium pt_BR-cadu-medium \
       --download-dir vozes
```

**Abrir (o jeito fácil):** duplo-clique em **`estudio.bat`** (Windows) ou
`./estudio.sh` (Linux Mint). Ele sobe a API e abre a página sozinho — não
carrega o Streamlit, então é mais leve que o `cortexus.bat`.

Se você já está com o painel da agência aberto, o estúdio **já está no ar**
(é a mesma API): use o botão **🎙️ Abrir o Estúdio de Podcast** na barra
lateral, ou vá direto em <http://localhost:8000/estudio>.

Na mão: `python api.py` e abra <http://localhost:8000/estudio>.

1. Escreva o **tema**, escolha a duração e clique em **“Escrever roteiro com
   a IA”** — a equipe leva alguns minutos.
2. O roteiro cai num editor. **Corrija à mão**: é aqui que você conserta o
   que o 7B errou. Editar texto é grátis; refazer o áudio custa minutos.
3. Clique em **“Gerar episódio”**. O player aparece com o MP3 e a transcrição.

**Modo de escrita** (na coluna da esquerda):

- **Em blocos** (padrão) — a equipe primeiro pesquisa na web, **lê** a melhor
  fonte e monta uma pauta; depois escreve um trecho por vez. É mais lento e é
  o que faz o episódio chegar perto da duração pedida: no passe único, um
  alvo de 5 minutos rendia 11 falas.
- **Passe único** — um pedido só. Mais rápido, roteiro bem mais curto.

**A dica que mais melhora o resultado:** cole um texto de referência em
**“Material de base”**. Com ele, os fatos saem *desse* texto em vez da memória
do modelo — é a defesa mais forte contra definição errada e exemplo inventado.

> **Não precisa do Ollama** se você colar um roteiro pronto no editor e pular
> direto para o passo 3. Só a escrita do roteiro depende do modelo de chat.

**Pelo terminal, sem servidor nenhum:**
```bash
python podcast.py roteiro.txt --motor piper --titulo "Episódio 1"
python podcast.py roteiro.txt --motor tom       # demonstração, sem baixar voz
python podcast.py roteiro.txt --trilha web/trilha.wav
```

**Formato do roteiro** — uma linha por fala:
```
Host1: Bem-vindo ao episódio de hoje.
Host2: E o assunto é bom.
```
O parser é tolerante: aceita `**Ana:**`, `[Bruno]`, `- ANA — `, junta linhas
quebradas na fala anterior, e descarta títulos, `Tema:`, rubricas `(risos)` e
o `Final Answer:` que o CrewAI às vezes vaza.

**Motor `tom`:** gera bipes em vez de voz, com a duração de cada fala. Serve
para conferir a montagem (pausas, trilha, crossfade) sem baixar modelo nenhum.

### 2.6. Ponte para outro orquestrador (nexus_bridge.py)

Se você tem OUTRO agente (ex.: Roo Code) e quer que ele consulte a Cortexus
pontualmente, sem apontar o modelo dele para a `:8000` (evita loop
agente→agente):
```bash
python nexus_bridge.py --pesquisar "Quais as novidades do FastAPI 0.115?"
python nexus_bridge.py --documentar "o módulo de crossfade do player"
```
O relatório sai no terminal (stdout). Endpoint por trás: `POST /v1/nexus-assist`.

---

## 3. Como a agência pensa (bom saber)

- **Regra de Intencionalidade:** pedidos de *criação* ("crie/escreva/gere um
  script…") → saída é **código cru**; pedidos de *análise*
  ("explique/resuma/analise…") → **relatório em prosa**.
- **Pipeline sequencial:** Analista de Requisitos → Pesquisador-Curador
  (busca web + leitura de disco + busca semântica/RAG) → Escritor → Revisor.
- **RAG:** para achar algo em arquivos grandes sem estourar o contexto,
  aponte a "Pasta p/ Busca Semântica" para uma **subpasta específica** (nunca
  a raiz — indexaria o `.venv`).

---

## 4. Exemplos de pedidos

| Objetivo | Pedido de exemplo |
|---|---|
| Gerar código | `Crie um script Python que baixe um arquivo via HTTP e salve como baixar.py` |
| Analisar arquivo | `Explique o que a função busca_web faz no main.py` |
| Pesquisa web | `Pesquise boas práticas de systemd services em 2026` |
| Documentar | (bridge) `--documentar "a API /agencia do projeto"` |
| Buscar em código | `Onde está a lógica de crossfade?` + Pasta RAG = `src/audio` |

---

## 5. Rodar os testes

A suíte é **hermética** (não exige Ollama nem rede — a crew e a rede são
mockadas). Cobre a camada determinística (classificação, limpeza, nomes de
arquivo, rede de segurança, sandbox) e o contrato HTTP.
```bash
pytest
```
O que a suíte **não** cobre (de propósito): a qualidade da saída do modelo
Qwen em si — isso é avaliação manual, não teste automatizado.

---

## 6. Problemas comuns

- **"A API não está no ar"** no painel → rode `python api.py` (ou o atalho).
- **`WinError 10048` / porta 8000 ocupada** → já há uma API rodando; use-a ou
  encerre a antiga antes de subir outra.
- **Job demora muito / timeout** → o modo Gerente e pedidos grandes são
  lentos no 7B; aumente o timeout no painel ou no cliente.
- **Resposta repetida/estranha** → desligue a Memória (contaminação do 7B).
- **Sem cores no código do painel** → confira que a extensão do arquivo é
  reconhecida (`.py/.sh/.js/...`); `.md`/`.txt` são tratados como prosa.

### Estúdio de Podcast

- **"Voz do Piper não encontrada"** → rode o `download_voices` da seção 2.5.
  O erro já traz o comando exato, com a pasta certa.
- **"Não encontrei nenhuma fala no roteiro"** → o texto não tem o formato
  `Nome: texto` em nenhuma linha. Acontece quando o 7B ignora o formato e
  devolve prosa corrida; peça de novo ou escreva as falas à mão.
- **`Missing phoneme from id map: ̃`** → você escolheu a voz
  **`pt_BR-edresson-low`**, que foi publicada sem o til nasal e engole o "ã"
  ("não" sai "nao"). Use `faber`, `cadu` ou `jeff`.
- **Sai WAV em vez de MP3** → falta o FFmpeg no PATH. É opcional; para ter o
  MP3: `winget install Gyan.FFmpeg` (Windows) ou `sudo apt install ffmpeg`.
- **As duas vozes soam parecidas** → o catálogo pt-BR do Piper é todo
  masculino. O par de maior contraste é `faber` (mais agudo) + `cadu` (mais
  grave); `jeff` fica no meio.
- **Quanto tempo leva?** Com a GPU liberada: a equipe escreve um roteiro de 20
  falas em ~2,5 min, e a síntese + montagem de 2 min 17 s de áudio leva
  ~1,5 min. Não tente mandar o Piper para a GPU: ela está ocupada pelo modelo
  de chat, e na CPU ele já roda mais rápido que tempo real.

- **A escrita do roteiro levou UMA HORA em vez de minutos.** Não é a agência:
  é a GPU sendo impedida de subir o clock. Aconteceu de verdade num teste —
  a mesma crew que leva 2,5 min levou 62 min. Diagnóstico em um comando:

  ```powershell
  nvidia-smi --query-gpu=utilization.gpu,clocks.current.graphics,clocks.max.graphics,power.draw,pstate --format=csv
  ```

  Se aparecer **`utilization=100%` junto de `pstate=P8` e o clock atual em
  ~210 MHz** (contra ~3105 MHz de máximo) e uns 5 W de consumo, a placa está
  trabalhando no clock de repouso — ~4% do desempenho. Medindo a geração:

  ```powershell
  $b='{"model":"qwen2.5-8k","prompt":"Escreva um paragrafo.","stream":false}'
  $r=Invoke-RestMethod http://localhost:11434/api/generate -Method Post -Body $b -ContentType application/json
  [math]::Round($r.eval_count/($r.eval_duration/1e9),1)   # tok/s
  ```

  Saudável nesta máquina é **~39 tok/s em P0** (clock 1110 MHz); o sintoma é
  **1,7 tok/s em P8** — 23× mais lento. O sinal que importa é o **pstate**,
  não a utilização: 100% de uso em P8 significa "trabalhando no clock de
  repouso".

  **O estado se resolveu sozinho** entre dois testes, sem nada ter sido
  fechado — então é uma condição transitória do driver/energia, não um app
  específico. (Roblox e o Epic Games Launcher estavam abertos nas DUAS
  medições, a lenta e a rápida; não eram a causa.) Se acontecer: espere um
  pouco e meça de novo; se persistir, verifique o Modo de Energia do Windows
  em *Melhor desempenho* e, no **Painel de Controle da NVIDIA → Gerenciar
  configurações 3D → Modo de gerenciamento de energia**, escolha "Preferir
  desempenho máximo".
