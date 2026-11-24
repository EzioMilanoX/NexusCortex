# Changelog

O que mudou no NexusCortex (Cortexus), do ponto de vista de quem usa. O
histórico anterior a este arquivo está no `git log` (níveis 1 a 6, até o
Estúdio de Podcast e a API só no loopback, 2026-11-19).

## Ainda sem versão

### Equipe ampliada: três especialistas novos

- **Revisor de Projetos** — a revisão semanal dos projetos pessoais (conceito
  SYS-16 R.E.C.A.P.). `python recap.py` lê o `git log` dos últimos 7 dias de
  cada pasta em `PersonalProjects`, as alterações ainda sem commit e a linha
  do projeto no `PROJECTS.md`, e escreve um relatório: o que andou, como isso
  se compara ao próximo passo anotado, um palpite de próximo passo e quem está
  parado. Sai em `output/recap/`. `--so-coletar` mostra só os dados, sem
  Ollama; `--dias` muda a janela; `--raiz` / `NCX_PROJETOS` mudam a pasta.
- **Crítico de Texto** — confere um texto contra uma referência (fidelidade,
  clareza, língua) e devolve a lista de problemas e a versão revisada. Na
  revisão semanal, ele recebe os mesmos dados do Revisor e corta progresso que
  não está nos commits. Sozinho: `python critica.py arquivo.md [--foco ...]
  [--referencia fatos.md]`, saída em `output/critica/`.
- **Roteirista do Podcast Semanal** — com `python recap.py --roteiro`, a
  revisão vira um diálogo `Rafa: ... / Bruno: ...`, já normalizado pelo parser
  do Estúdio; `python podcast.py <roteiro> --motor piper` narra.
- **API:** `GET /recap/coleta?dias=7` (só os dados, sem Ollama), `POST /recap`
  e `POST /critica`, com o mesmo contrato do `/agencia` (job_id na hora,
  `/status`, webhook opcional). A pasta de projetos é sempre a do servidor —
  a API não aceita caminho vindo do cliente.
- Os agentes novos não têm ferramentas: os fatos entram no prompt coletados
  por código, com teto de tamanho para caber no contexto de 8k do qwen 7B.

### Manutenção

- **Ruff com config** (`ruff.toml`): regras E4/E7/E9/F, o mesmo padrão da
  OuroborosEngine e do BulletHell; `output/` (scripts gerados pela agência)
  fica fora. `ruff check .` limpo (saiu um import morto nos testes).
- Testes: 302 → 346, todos sem rede e sem Ollama.
