
# 🏎️ F1 Predictor - Abu Dhabi Grand Prix 2025

Este projeto utiliza Machine Learning para prever os resultados do Grande Prêmio de Abu Dhabi de Fórmula 1 na temporada de 2025. O sistema analisa estatísticas de pilotos, similaridade de pistas e dados históricos para gerar insights, tabelas de classificação e visualizações gráficas sobre o desempenho esperado.

## 📂 Estrutura do Projeto

O projeto está organizado de forma modular, separando scripts, dados e resultados visuais:

```text
F1_PRE-ABU_DHABI/
│
├── graficos/                            # Armazena os gráficos gerados pelos scripts
│   ├── feature_importance_abu_dhabi.png # Gráfico: Variáveis que mais impactaram a previsão
│   └── previsao_ganhos_perdas.png       # Gráfico: Comparativo de Posição de Largada vs. Chegada
│
├── results/                          # Armazena datasets de entrada e tabelas de saída
│   ├── driver_stats_2025.csv         # Estatísticas atualizadas dos pilotos (Input)
│   ├── f1_training_dataset.csv       # Dataset processado para treino do modelo (Input)
│   └── resultado_previsao_abu_dhabi  # Tabela final com a classificação prevista (Output)
│
├── f1_predictor_model.pkl     # Modelo de Machine Learning treinado
├── gerador_dataset.py         # Script: Processa dados brutos e gera o dataset de treino
├── preditor_Abu_Dhabi.py      # Script: Carrega o modelo e executa a previsão da corrida
├── similarity_score.py        # Script: Calcula similaridade entre Yas Marina e outras pistas
└── vizualização.py            # Script: Gera os gráficos baseados nos resultados
```

## 🛠️ Tecnologias Utilizadas

  * **Python 3.x**
  * **Pandas** (Manipulação de dados)
  * **Scikit-Learn** (Machine Learning e Random Forest)
  * **Matplotlib & Seaborn** (Visualização de dados)
  * **FastF1** (Extração de dados de telemetria e cronometragem)

## 🚀 Como Executar

Siga a ordem abaixo para garantir que os arquivos sejam gerados corretamente nas pastas.

### 1\. Instalação das Dependências

Certifique-se de ter as bibliotecas instaladas:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn fastf1
```

### 2\. Configuração Avançada (Similaridade de Pista)

O arquivo `similarity_score.py` define como o modelo compara a pista alvo (Yas Marina) com as outras do calendário.

  * **Se precisar ajustar:** Abra o arquivo `similarity_score.py` e altere os dicionários `target_track_features` (para mudar as características da pista alvo) ou `feature_weights` (para mudar o peso dado a curvas, retas, etc.).

### 3\. Geração do Dataset

Caso precise processar novos dados ou recriar a base de treino na pasta `results`:

```bash
python gerador_dataset.py
```

### 4\. Executar a Previsão

Este script carrega o modelo `f1_predictor_model.pkl` e os dados de `results/`, gerando a previsão final:

```bash
python preditor_Abu_Dhabi.py
```

*O resultado será salvo em: `results/resultado_previsao_abu_dhabi.csv`*

### 5\. Gerar Gráficos e Insights

Para visualizar a Importância das Features e a Previsão de Ganhos/Perdas:

```bash
python vizualização.py
```

*As imagens serão salvas automaticamente na pasta `graficos/`.*

## 📊 Análises Disponíveis

O projeto gera duas análises visuais principais:

1.  **Feature Importance:** Mostra quais estatísticas (ex: pontos na temporada, vitórias, grid de largada) foram decisivas para o modelo escolher o vencedor.
2.  **Ganho/Perda de Posições:** Um gráfico comparativo que ilustra a diferença prevista entre a posição de largada e a posição final prevista.

-----

*Desenvolvido para fins acadêmicos, análise de dados esportivos e por diversão😏.*
