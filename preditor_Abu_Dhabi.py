import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

df_history = pd.read_csv('f1_training_dataset.csv')

GRID_DE_LARGADA_2025 = {
    'VER': 1,
    'NOR': 2,
    'PIA': 3,
    'RUS': 4,
    'LEC': 5,
    'ALO': 6,
    'BOR': 7,
    'OCO': 8,
    'HAD': 9,
    'TSU': 10,
    'BEA': 11,
    'SAI': 12,
    'LAW': 13,
    'ANT': 14,
    'STR': 15,
    'HAM': 16,
    'ALB': 17,
    'HUL': 18,
    'GAS': 19,
    'COL': 20
}

def prepare_driver_stats(df):
    """
    Criar um perfil estatístico para cada piloto com base em seu desempenho histórico.
    """
    driver_stats = []

    drivers = df['Driver'].unique()

    for driver in drivers:
        d_data = df[df['Driver'] == driver]

        # A. força em pistas similares (2025)
        # filtra corridas de 2025 que tenham similaridade (>0.7)
        # se não tiver dados, vai usar a media geral
        similar_races = d_data[(d_data['Year'] == 2025) & (d_data['Track_Similarity'] > 0.7)]
        if len(similar_races) > 0:
            avg_similar_pos = similar_races['Finish_Pos'].mean()
        else:
            avg_similar_pos = d_data[d_data['Year'] == 2025]['Finish_Pos'].mean()
        
        # B. desempenho em Abu Dhabi
        abu_dhabi_hist = d_data[d_data['Round'] == 'Abu Dhabi']
        if len(abu_dhabi_hist) > 0:
            avg_abu_pos = abu_dhabi_hist['Finish_Pos'].mean()
        else:
            avg_abu_pos = 15.0 # valor neutro para os rookies

        # C. Momentum (ultimas 3 corridas)
        recent = d_data[d_data['Year'] == 2025].tail(3)
        if len(recent) > 0:
            momentum = recent['Finish_Pos'].mean()
        else:
            momentum = 15.0

        driver_stats.append(
            {
                'Driver': driver,
                'Avg_Pos_Similar_Tracks': avg_similar_pos,
                'Avg_Pos_Abu_Dhabi': avg_abu_pos,
                'Recent_Momentum': momentum
            }
        )
    return pd.DataFrame(driver_stats)

print("Calculando estatísticas dos pilotos...")
stats_df = prepare_driver_stats(df_history)

stats_df.to_csv('driver_stats_2025.csv', index=False)

print("Treinando o modelo...")

train_data = df_history.merge(stats_df, on='Driver', how='left')
train_data = train_data.fillna(15.0)

features = ['Grid_Pos', 'Track_Similarity',
            'Avg_Pos_Similar_Tracks', 'Avg_Pos_Abu_Dhabi', 'Recent_Momentum']
target = 'Finish_Pos'

# Modelo: Random Forest Regressor (lida bem com não linearidades e dados pequenos)
model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
model.fit(train_data[features], train_data[target])

joblib.dump(model, 'f1_predictor_model.pkl')
print("Modelo e Estatísticas salvos com sucesso!")

# Gerar previsões para Abu Dhabi 2025(07 de Dezembro de 2025)
print("Gerando previsões para Abu Dhabi 2025...")

prediction_rows = []

for driver, grid_pos in GRID_DE_LARGADA_2025.items():
    # buscar estatísticas do piloto
    drv_stat = stats_df[stats_df['Driver'] == driver]
    if not drv_stat.empty:
        avg_sim = drv_stat.iloc[0]['Avg_Pos_Similar_Tracks']
        avg_abu = drv_stat.iloc[0]['Avg_Pos_Abu_Dhabi']
        mom = drv_stat.iloc[0]['Recent_Momentum']
    else:
        avg_sim = 15.0
        avg_abu = 15.0
        mom = 15.0
    
    row = {
        'Driver': driver,
        'Grid_Pos': grid_pos,
        'Track_Similarity': 1.0,  # Abu Dhabi vs Abu Dhabi
        'Avg_Pos_Similar_Tracks': avg_sim,
        'Avg_Pos_Abu_Dhabi': avg_abu,
        'Recent_Momentum': mom
    }
    prediction_rows.append(row)

df_tomorrow = pd.DataFrame(prediction_rows)

predicted_positions = model.predict(df_tomorrow[features])
df_tomorrow['Predicted_Finish'] = predicted_positions

final_grid = df_tomorrow.sort_values('Predicted_Finish')[['Driver', 'Grid_Pos', 'Predicted_Finish']]
final_grid['Predicted_Pos'] =  range(1, len(final_grid) + 1)

final_grid['Predicted_Finish'] = final_grid['Predicted_Finish'].round(2)

print("\nPrevisão GP Abu Dhabi 2025:")
print(final_grid[['Driver', 'Grid_Pos', 'Predicted_Pos']].to_string(index=False))

final_grid.to_csv('resultado_previsao_abu_dhabi.csv', index=False)
print("\nArquivo 'resultado_previsao_abu_dhabi.csv' gerado com sucesso!")

importances = pd.Series(model.feature_importances_, index=features)
print("\n O que pesou mais na decisão do modelo?")
print(importances.sort_values(ascending=False))