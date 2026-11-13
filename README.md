# 🏁 Pit wall: estratégia de F1 com dados públicos

Engenheiro de estratégia de Fórmula 1 feito com dados abertos de cronometragem
([OpenF1](https://openf1.org), [FastF1](https://github.com/theOehrly/Fast-F1)) e as
nominações de compostos da Pirelli. Mede a degradação dos pneus nas corridas anteriores,
compara estratégias de pit stop, simula a corrida inteira com todos os carros e mostra
tudo num dashboard.

Começou como um preditor do GP de Abu Dhabi de 2025 (dezembro de 2025, scripts originais
na raiz do repositório) e virou, ao longo da temporada de 2026, o pacote `f1agent`.

## O dashboard

No ar em [huggingface.co/spaces/vtaua/f1-pit-wall](https://huggingface.co/spaces/vtaua/f1-pit-wall).
Para rodar localmente:

```bash
pip install -r requirements.txt
python app.py        # http://127.0.0.1:7860
```

- **Strategy**: estratégia recomendada para a corrida escolhida, faixa de incerteza de
  cada opção, stints comparados com o que o top 10 fez de verdade, curvas de degradação e
  mapa da pista colorido por velocidade. Compostos, temperatura da pista, perda no pit e
  chance de safety car são editáveis.
- **Race simulation**: o grid inteiro volta a volta (tráfego, ultrapassagem, ar sujo,
  safety car, undercut), distribuição da posição final para duas estratégias e replay dos
  carros no mapa.
- **Scoreboard**: todos os backtests em 2026, inclusive onde o modelo não supera
  referências simples.
- **Method**: a matemática ([docs/methodology.md](docs/methodology.md)).

## Resultados (backtests walk-forward, sem vazamento de dados)

| | resultado |
|---|---|
| Ordem de chegada, Spearman médio (68 corridas de 2024 a 2026) | ensemble 0,705 vs grid 0,677 (2025–26, fora do ajuste: 0,684 vs 0,648) |
| Simulação da corrida inteira, ordem esperada (17 corridas de 2026) | 0,703 vs grid 0,672 |
| Degradação prevista, erro médio em 2026 (s/volta por volta de idade) | histórico 0,033; long runs dos treinos 0,088 |
| Estratégia com os compostos que a maioria do top 10 usou (17 corridas de 2026) | 65% (primeira versão: 12%) |
| Primeira parada, erro mediano (17 corridas de 2026, top 5) | simulação 8–9,5 voltas, referência histórica 8 |

A previsão de chegada combina um gradient boosting com um modelo de ranking Plackett-Luce
(forças de piloto e de equipe por temporada, como em van Kesteren & Bergkamp, 2023) e dá
chances de vitória, pódio e top 10, avaliadas com o Brier score. A simulação inclui abandonos
e usa também o ritmo de corrida das temporadas anteriores. O grid ainda acerta o vencedor
mais vezes que o modelo.

Os long runs dos treinos livres pioraram a estimativa de degradação e a estratégia, então o
dashboard usa só o modelo ajustado nas corridas anteriores.

O agente (Gemma 4 31B com 10 ferramentas) acertou 16 de 16 perguntas da avaliação automática,
que checa a resposta e a ferramenta usada (`results/agent_eval_gemini`). O Nemotron, que entra
na cadeia quando o Gemma gratuito está ocupado, acertou as 14 que respondeu antes do limite
diário do plano gratuito (`results/agent_eval_nemotron`).

A simulação não acerta o momento da parada melhor que a referência histórica: um quarto
das primeiras paradas reais acontece sob safety car, que nenhum plano pré-corrida prevê.
Ela serve para comparar opções, não como previsão. Detalhes e todas as execuções em
[docs/test-log.md](docs/test-log.md).

## Estrutura

```text
f1agent/
  data.py, features.py, backtest.py    resultados (FastF1), features e backtest da ordem de chegada
  openf1.py, laps.py, trackmap.py      voltas, stints, clima e telemetria do OpenF1
  tyres.py, practice.py                degradação dos pneus (corridas e long runs dos treinos)
  strategy.py, strategy_backtest.py    simulador de estratégia de um carro (Monte Carlo)
  racesim.py, racepace.py, racesim_backtest.py   simulação com todos os carros
  agent.py, tools.py, agent_eval.py    agente LLM com ferramentas sobre os dados
  planner.py                           camada de dados do dashboard
app.py                                 dashboard (Gradio + Plotly)
data/                                  dados processados (data/upcoming.csv: próximas corridas)
results/                               saídas dos backtests
```

## Atualizar a cada sessão

Depois de cada treino livre, classificação ou corrida:

```bash
python -m f1agent.update --publish
```

Baixa as sessões novas do OpenF1, os resultados novos, o grid real depois da
classificação, refaz as voltas, registra a previsão atual de cada próxima corrida em
`results/live/predictions.jsonl` (só quando ela muda) e, com `--publish`, roda os testes,
faz o commit, o push e o deploy do Space (`HF_TOKEN` no `.env`). Quando a Pirelli anunciar
os compostos de uma corrida:

```bash
python -m f1agent.update --compounds "Las Vegas Grand Prix=C3,C4,C5" --publish
```

## Reproduzir

```bash
python -m f1agent.data --years 2023 2024 2025 2026     # resultados e grids
python -m f1agent.openf1 --years 2024 2025 2026 --practice
python -m f1agent.laps                                 # data/laps_clean.csv.gz
python -m f1agent.trackmap                             # mapas das pistas
python -m f1agent.backtest
python -m f1agent.tyres
python -m f1agent.strategy_backtest
python -m f1agent.racesim_backtest
python -m pytest
```

O agente usa só planos gratuitos (`OPENROUTER_API_KEY` e/ou `GEMINI_API_KEY` num arquivo `.env`,
fora do git). Como as cotas gratuitas oscilam, ele percorre uma cadeia de modelos gratuitos que
chamam ferramentas (Gemma 4 31B e 26B e Nemotron no OpenRouter, Gemma 4 31B na API do Gemini) e
troca de modelo assim que um está ocupado. O chat mostra ao vivo o modelo e cada ferramenta
chamada, e cada resposta termina dizendo qual modelo respondeu.

---

Projeto de fã, sem ligação com a Fórmula 1 ou a Pirelli. Os dados do OpenF1 são para uso
não comercial.
