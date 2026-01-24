import pandas as pd
import matplotlib.pyplot as plt

# Configuration
CSV_FILE = 'mesures_fps.csv'

def plot_timeline():
    try:
        # 1. Lecture et Nettoyage (comme avant, on gère l'en-tête cassé)
        col_names = ['Vague', 'Info_FPS', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
        df = pd.read_csv(CSV_FILE, sep=';', skiprows=1, names=col_names, on_bad_lines='skip')

        # 2. Calculer une valeur unique par ligne (Moyenne des 10 frames de l'échantillon)
        # Cela représente la performance à cet instant T
        frame_cols = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']
        df['Avg_Ms'] = df[frame_cols].mean(axis=1)

        # 3. Préparation du graphique
        plt.figure(figsize=(14, 8))
        
        # On trace la courbe principale : X = Index (Temps), Y = Ms
        plt.plot(df.index, df['Avg_Ms'], color='#1f77b4', linewidth=1.5, label='Temps de frame (ms)')
        
        # On peut aussi afficher les points individuels pour voir le "bruit" (optionnel, mais joli)
        plt.scatter(df.index, df['Avg_Ms'], s=10, color='#1f77b4', alpha=0.5)

        # 4. Gestion des Barres Verticales (Changement de Vague)
        # On cherche les index où la valeur de 'Vague' change par rapport à la ligne d'avant
        # df['Vague'].shift() décale la colonne d'un cran pour comparer
        transitions = df[df['Vague'] != df['Vague'].shift()]
        
        # Pour chaque transition, on trace une ligne et on met un texte
        for idx, row in transitions.iterrows():
            wave_num = int(row['Vague'])
            zombie_count = wave_num * 1000 + 100
            
            # Ligne verticale rouge
            plt.axvline(x=idx, color='red', linestyle='--', alpha=0.6)
            
            # Texte (Vague + Nb Zombies)
            # On place le texte un peu en haut du graphique
            max_y = df['Avg_Ms'].max()
            plt.text(idx + 0.5, max_y, 
                     f"V{wave_num}\n({zombie_count} Z)", 
                     rotation=0, 
                     color='red', 
                     fontsize=9, 
                     verticalalignment='bottom',
                     fontweight='bold')

        # 5. Seuils de FPS (Lignes horizontales)
        plt.axhline(y=16.66, color='green', linestyle=':', label='60 FPS')
        plt.axhline(y=33.33, color='orange', linestyle=':', label='30 FPS')

        # 6. Esthétique
        plt.title('Chronologie de la Performance (Évolution seconde par seconde)', fontsize=14)
        plt.xlabel('Temps écoulé (en secondes/mesures)', fontsize=12)
        plt.ylabel('Temps de calcul (ms)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Ajuster les marges pour que les textes du haut ne soient pas coupés
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Fichier {CSV_FILE} introuvable.")
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    plot_timeline()