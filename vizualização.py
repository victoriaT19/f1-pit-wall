import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

arquivo_csv = 'resultado_previsao_abu_dhabi.csv'

# dados obtidos do preditor_Abu_Dhabi.py
features = ['Grid_Pos', 'Avg_Pos_Similar_Tracks', 'Track_Similarity', 'Avg_Pos_Abu_Dhabi', 'Recent_Momentum']
importances = [0.804277, 0.076060, 0.050817, 0.034930, 0.033917]

df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
df_imp = df_imp.sort_values('Importance', ascending=True)

# Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(df_imp['Feature'], df_imp['Importance'], color="#FF0000") # Vermelho F1

for index, value in enumerate(df_imp['Importance']):
    plt.text(value, index, f'{value:.1%}', va='center', fontsize=10, fontweight='bold')

plt.title('O que define o vencedor em Abu Dhabi? (Peso das Variáveis)', fontsize=14)
plt.xlabel('Importância (0-1)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(r'graficos\feature_importance_abu_dhabi.png')
plt.show()

final_grid = pd.read_csv(arquivo_csv)

final_grid['Difference'] = final_grid['Grid_Pos'] - final_grid['Predicted_Pos'] 
final_grid = final_grid.sort_values('Difference', ascending=False)

plt.figure(figsize=(14, 10))
sns.set_style("whitegrid") 

colors = []
for val in final_grid['Difference']:
    if val > 0:
        colors.append('#00D2BE') 
    elif val < 0:
        colors.append('#DC0000') 
    else:
        colors.append('#B0B0B0') 

ax = sns.barplot(
    x='Difference', 
    y='Driver', 
    data=final_grid, 
    palette=colors, 
    edgecolor='black' 
)

plt.title('PREVISÃO DE PERFORMANCE:\nGP Abu Dhabi 2025', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Saldo de Posições (Verde = Ganhou | Vermelho = Perdeu)', fontsize=12, fontweight='bold')
plt.ylabel('', fontsize=12)

plt.axvline(0, color='black', linewidth=2)

for i, v in enumerate(final_grid['Difference']):
    label = f'+{int(v)}' if v > 0 else f'{int(v)}'
    
    offset = 0.2 if v >= 0 else -0.5
    
    ha_align = 'left' if v >= 0 else 'right'
    
    ax.text(v + offset, i, label, 
            va='center', ha=ha_align, 
            fontsize=12, fontweight='bold', color='black')
    
plt.yticks(
    fontweight='bold',  
    fontsize=12,        
    rotation=45       
)

plt.xticks(
    fontweight='bold',  
    fontsize=12,        
    rotation=32       
)

plt.xlim(final_grid['Difference'].min() - 1, final_grid['Difference'].max() + 1)
plt.tight_layout()

plt.savefig(r'graficos\previsao_ganhos_perdas.png', dpi=300)
print("Gráfico 'previsao_ganhos_perdas.png' salvo!")
plt.show()