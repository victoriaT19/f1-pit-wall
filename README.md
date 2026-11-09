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
| Ordem de chegada, Spearman médio (68 corridas de 2024 a 2026) | modelo 0,691 vs grid 0,677 |
| Degradação prevista, erro médio (s/volta por volta de idade) | histórico 0,030; long runs dos treinos 0,088 |
| Primeira parada, erro mediano (17 corridas de 2026, top 5) | simulação 8–9,5 voltas, referência histórica 8 |

Os long runs dos treinos livres pioraram a estimativa de degradação e a estratégia, então o
dashboard usa só o modelo ajustado nas corridas anteriores.

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

O agente usa a API do Gemini (`GEMINI_API_KEY` num arquivo `.env`, fora do git).

---

Projeto de fã, sem ligação com a Fórmula 1 ou a Pirelli. Os dados do OpenF1 são para uso
não comercial.
