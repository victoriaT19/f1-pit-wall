import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('cache')

# preenchido com o resultado do similarity_score.py (automatizar dps)
TRACK_SIMILARITY_MAP = {
    'Abu Dhabi': 1.000000,
    'Emilia-Romagna': 0.924791,
    'Spain': 0.833175,
    'Austria': 0.804746,
    'Great Britain': 0.804746,
    'Las Vegas': 0.780231,
    'Netherlands': 0.751072,
    'Belgium': 0.741562,
    'China': 0.737874,
    "Qatar": 0.713642,
    'Brazil': 0.710713,
    'Miami': 0.710065,
    'United States': 0.699448,
    'Australia': 0.692756,
    'Canada': 0.671013,
    'Saudi Arabia': 0.655450,
    'Bahrain': 0.652141,
    'Italy': 0.646390,
    'Hungary': 0.645235,
    'Azerbaijan': 0.635056,
    'Japan': 0.623925,
    'Mexico': 0.562857,
    'Singapore': 0.523445,
    'Monaco': 0.449348
}

RACES_TO_PROCESS = [
    (2021, 'Abu Dhabi'), (2022, 'Abu Dhabi'), (2023, 'Abu Dhabi'),

    #temporada de 2024

    (2024, 'Bahrain'), (2024, 'Saudi Arabia'), (2024, 'Australia'),
    (2024, 'Japan'), (2024, 'China'), (2024, 'Miami'),
    (2024, 'Emilia-Romagna'), (2024, 'Monaco'), (2024, 'Canada'),
    (2024, 'Spain'), (2024, 'Austria'), (2024, 'Great Britain'),
    (2024, 'Hungry'), (2024, 'Belgium'), (2024, 'Netherlands'),
    (2024, 'Italy'), (2024, 'Azerbaijan'), (2024, 'Singapore'),
    (2024, 'United States'), (2024, 'Mexico'), (2024, 'Brazil'),
    (2024, 'Las Vegas'), (2024, 'Qatar'), (2024, 'Abu Dhabi'),

    #temporada de 2025
    (2025, 'Bahrain'), (2025, 'Saudi Arabia'), (2025, 'Australia'),
    (2025, 'Japan'), (2025, 'China'), (2025, 'Miami'),
    (2025, 'Emilia-Romagna'), (2025, 'Monaco'), (2025, 'Canada'),
    (2025, 'Spain'), (2025, 'Austria'), (2025, 'Great Britain'),
    (2025, 'Hungry'), (2025, 'Belgium'), (2025, 'Netherlands'),
    (2025, 'Italy'), (2025, 'Azerbaijan'), (2025, 'Singapore'),
    (2025, 'United States'), (2025, 'Mexico'), (2025, 'Brazil'),
    (2025, 'Las Vegas'), (2025, 'Qatar')
]

def build_training_dataset():
    all_data = []

    print(f"Iniciando coleta de {len(RACES_TO_PROCESS)} corridas...")

    for year, gp_name in RACES_TO_PROCESS:
        try:
            print(f'Processando {gp_name} {year}...')

            # carregar sessão de corrida (Resultados finais)
            session = fastf1.get_session(year, gp_name, 'R')
            session.load(telemetry=False, weather=False, messages=False)

            results = session.results

            #Busca a similaridade
            sim_score = TRACK_SIMILARITY_MAP.get(gp_name, None)

            #iterar por cada piloto na corrida
            for driver_number in results.index:
                drv_res = results.loc[driver_number]
                driver_code = drv_res['Abbreviation']

                # tratamento DNF
                status = drv_res['Status']
                finish_pos = drv_res['Position']

                if not np.isfinite(finish_pos):
                    finish_pos = 20

                grid_pos = drv_res['GridPosition']
                if grid_pos == 0:
                    grid_pos = 20 #largando do pitlane

                row = {
                    'Year': year,
                    'Round': gp_name,
                    'Driver': driver_code,
                    'Team': drv_res['TeamName'],
                    'Grid_Pos': grid_pos,
                    'Finish_Pos': finish_pos,
                    'Points': drv_res['Points'],
                    'Status': status,
                    'Track_Similarity': sim_score,
                    'Weighted_Score': sim_score * (21 - finish_pos)
                }
                all_data.append(row)

        except Exception as e:
            print(f'Erro ao processar {gp_name} {year}: {e}')
            continue
    
    df = pd.DataFrame(all_data)
    return df

df_train = build_training_dataset()

df_train.to_csv(r'results\f1_training_dataset.csv', index=False)
print("\nDataset salvo em 'f1_training_dataset.csv'")
print(df_train.head())
        