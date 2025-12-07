import fastf1
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
import os

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir) 

TRACKS_TO_COMP = [
    ('Australia', 2025), ('China', 2025), ('Japan', 2025),
    ('Bahrain', 2025), ('Saudi Arabia', 2025), ('Miami', 2025),
    ('Emilia-Romagna', 2025), ('Monaco', 2025), ('Spain', 2025),
    ('Canada', 2025), ('Austria', 2025), ('Great Britain', 2025),
    ('Belgium', 2025), ('Hungary', 2025), ('Netherlands', 2025),
    ('Italy', 2025), ('Azerbaijan', 2025), ('Singapore', 2025),
    ('United States', 2025), ('Mexico', 2025), ('Brazil', 2025),
    ('Qatar', 2025), ('Las Vegas', 2025), ('Abu Dhabi', 2025)
]

TARGET_TRACK = ('Abu Dhabi', 2025)

def extract_track_dna(gp_name, year):
    """Extrai a assinatura técnica da pista baseada na volta da Pole."""
    print(f'Extraindo DNA da pista: {gp_name} {year}...')

    try:
        #carregar sessão de qualificação
        qual_session = fastf1.get_session(year, gp_name, 'Q')
        qual_session.load(telemetry=True, laps=True, weather=True)

        #filtrar chuva(se tiver chovido, os dados não são confiaveis para comparação)
        if qual_session.weather_data['Rainfall'].any():
            print(f'Pista {gp_name} {year} teve chuva na qualificação. Ignorando...')
            return None
        
        #pegar volta mais rapida (referencia absoluta da pista)
        lap = qual_session.laps.pick_fastest()
        tel = lap.get_telemetry()

        #Engenharia de features

        # A. Velocidade Media
        avg_speed = tel['Speed'].mean()

        # B. Power Sensivity (% Pé embaixo)
        # Quantidade de tempo com acelerador > 95%
        full_throttle = len(tel[tel['Throttle'] > 95]) / len(tel) * 100

        # C. Complexidade Técnica (Trocas de marcha por km)
        # Conta mudanças na coluna 'nGear' e normaliza pelo tamanho da pista
        gear_changes = tel['nGear'].diff().abs().sum()

        technicality_index = gear_changes / len(tel)

        # D. Exigência de Freio (% da volta com freio acionado)

        braking = len(tel[tel['Brake'] > 0]) / len(tel) * 100

        return{
            'Track': gp_name,
            'Year': year,
            'Avg_Speed': avg_speed,
            'Full_Throttle_%': full_throttle,
            'Technicality': technicality_index,
            'Braking_%': braking
        }
    except Exception as e:
        print(f'Erro ao processar {gp_name} {year}: {e}')
        return None

data = []

target_dna = extract_track_dna(TARGET_TRACK[0], TARGET_TRACK[1])
if target_dna:
    data.append(target_dna)
else:
    # Fallback para 2024 se 2025 falhar
    print(f'Tentando Abu Dhabi 2024 como fallback...')
    target_dna = extract_track_dna('Abu Dhabi', 2024)
    data.append(target_dna)

# Processar outras pistas
for track_name, year in TRACKS_TO_COMP:
    if (track_name, year) == TARGET_TRACK:
        continue
    dna = extract_track_dna(track_name, year)
    if dna:
        data.append(dna)
    else:
        # Fallback para 2024 se 2025 falhar
        print(f'Tentando {track_name} 2024 como fallback...')
        dna = extract_track_dna(track_name, 2024)
        if dna:
            data.append(dna)

# Criar DataFrame
df = pd.DataFrame(data).set_index('Track')

# Normalizar features
scaler = MinMaxScaler()
features = ['Avg_Speed', 'Full_Throttle_%', 'Technicality', 'Braking_%']
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), 
                         columns=features, 
                         index=df.index)

# Calcular similaridade (distância Euclidiana) em relação à pista alvo
target_vector = df_scaled.loc[TARGET_TRACK[0]]

distances = []
for track in df_scaled.index:
    # calcula distância entre a pista atual e Abu Dhabi
    dist = euclidean(df_scaled.loc[track], target_vector)
    #similaridade é o inverso da distância (quanto menor a distância, mais similar)
    similarity = 1 / (1 + dist)
    distances.append(similarity)

df['Similarity_Score'] = distances

print('\nPistas similares a Abu Dhabi:')
print(df[['Year', 'Similarity_Score']].sort_values(by='Similarity_Score', ascending=False))

    