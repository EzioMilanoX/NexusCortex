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

## 2. Os quatro jeitos de usar

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

Com `python api.py` no ar:
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

### 2.5. Ponte para outro orquestrador (nexus_bridge.py)

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
