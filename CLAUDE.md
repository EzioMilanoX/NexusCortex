# CLAUDE.md — NexusCortex (Cortexus)

Guia para o Claude Code neste projeto. Idioma: PT-BR.

## O que é
Equipe local de agentes de IA (CrewAI + Ollama, qwen2.5 7B na RTX 4060),
100% offline: pesquisa, escreve, revisa e entrega arquivo — ou um podcast
narrado (Estúdio de Podcast). Blueprint **SYS-13 C.O.R.T.E.X.**; a revisão
semanal (`recap.py`) é o protótipo do **SYS-16 R.E.C.A.P.**

## Rodar / testar
```bash
python main.py <pedido>        # pipeline principal no terminal
python api.py                  # API em 127.0.0.1:8000 (/agencia, /podcast, /recap, /critica, /v1)
python recap.py --so-coletar   # dados da semana do git, sem Ollama
python critica.py arquivo.md   # Crítico de Texto
pytest                         # suíte hermética (sem Ollama/rede); `pytest -m llm` = fumaça real
ruff check .                   # lint (ruff.toml: E4/E7/E9/F)
```
Use o venv: `.venv\Scripts\python.exe -m pytest`. O ruff não está no venv;
o global (`python -m ruff`) serve.

## Mapa
- `main.py` — pipeline principal (Analista -> Pesquisador -> Escritor ->
  Revisor; modo Gerente hierárquico opcional), ferramentas com sandbox
  (`busca_web`, `ler_pagina_web`, disco, `executar_comando`), limpeza e
  salvamento da saída. `criar_llm()` força ReAct — não mexer sem motivo.
- `especialistas.py` — fábricas dos papéis de entrada fechada (Revisor de
  Projetos, Crítico de Texto, Roteirista do Podcast Semanal). Sem ferramentas.
- `recap.py` — coleta determinística (git + PROJECTS.md) + equipe da revisão
  semanal + CLI. `critica.py` — equipe do Crítico + CLI.
- `api.py` — FastAPI: jobs em memória (`_registrar_job`/`/status`), webhook,
  Estúdio de Podcast, fachada OpenAI, ponte Nexus, `/recap` e `/critica`.
- `podcast.py` — roteiro -> vozes (Piper) -> MP3. `ui.py` — painel Streamlit.

## Convenções
- **Passo essencial mora no código, não na obediência do modelo** (7B):
  fatos entram no prompt coletados por código; formato garantido por parser
  (`limpar_saida`, `parse_roteiro`, `escolher_saida`).
- Agente novo: fábrica em `especialistas.py` (ou em `montar_equipe` se for do
  pipeline principal), equipe própria com `Process.sequential`, rota com o
  contrato de job do `/agencia`, CLI `python <modulo>.py`, testes com crew
  falsa no `kickoff` e `diagnostico_ollama` mockado.
- Contexto do modelo é 8k: todo texto que entra no prompt tem teto.
- A API não tem autenticação: nunca aceitar caminho de disco arbitrário do
  cliente (ver `_caminho_de_projeto` e `recap.raiz_padrao`).
- Saídas em `output/`; o git ignora `output/*.txt`, `output/podcast/`,
  `output/recap/` e `output/critica/`.
- Atualize o `CHANGELOG.md` a cada mudança visível.
- Segue as convenções gerais de [../../CLAUDE.md](../../CLAUDE.md).
  Só commitar quando o Enzo pedir.
