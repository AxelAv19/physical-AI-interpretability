#!/usr/bin/env python

"""
Analyse TOUS les épisodes d'un dataset sans dépendre de la capture d'attention.
Utilise des métriques de base : durée, actions, variance, etc.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

def analyze_episode_basic(dataset: LeRobotDataset, episode_id: int, max_frames: int = 50):
    """Analyser un épisode avec des métriques de base (pas d'attention requise)."""
    
    try:
        episode_info = dataset.episode_data_index["from"][episode_id], dataset.episode_data_index["to"][episode_id]
        episode_start, episode_end = episode_info
        episode_length = episode_end - episode_start
        
        if episode_length == 0:
            return None
        
        # Limiter le nombre de frames pour la performance
        frames_to_analyze = min(episode_length, max_frames)
        episode_frame_indices = np.linspace(episode_start, episode_end - 1, frames_to_analyze, dtype=int)
        
    except (KeyError, IndexError, AttributeError):
        return None
    
    # Collections des métriques
    actions = []
    timestamps = []
    
    # Traiter les frames
    for dataset_idx in episode_frame_indices:
        try:
            dataset_idx = int(dataset_idx)
            frame = dataset[dataset_idx]
            
            # Extraire les actions
            if 'action' in frame:
                action = frame['action']
                if hasattr(action, 'numpy'):
                    action = action.numpy()
                actions.append(action.flatten())
            
            # Extraire timestamp
            if 'timestamp' in frame:
                timestamp = frame['timestamp']
                if hasattr(timestamp, 'item'):
                    timestamp = timestamp.item()
                timestamps.append(timestamp)
                
        except Exception:
            continue
    
    if len(actions) < 2:
        return None
    
    # Convertir en array numpy
    actions = np.array(actions)
    
    # Calculer les métriques
    features = {
        'episode_id': episode_id,
        'episode_length': int(episode_length),
        'frames_analyzed': len(actions),
        
        # Métriques d'actions
        'action_mean': float(np.mean(actions)),
        'action_std': float(np.std(actions)),
        'action_range': float(np.max(actions) - np.min(actions)),
        'action_velocity': float(np.mean(np.abs(np.diff(actions, axis=0)))),  # Variation entre frames
        
        # Métriques temporelles
        'duration_seconds': float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else float(episode_length),
        'fps_estimated': len(actions) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 and timestamps[-1] != timestamps[0] else 30.0,
        
        # Métriques de complexité
        'action_complexity': float(np.mean([np.std(action) for action in actions])),  # Variance interne des actions
        'trajectory_smoothness': float(1.0 / (1.0 + np.mean(np.abs(np.diff(actions, axis=0))))),  # Plus c'est lisse, plus c'est haut
    }
    
    return features

def create_basic_clustering(episode_features, n_clusters=4):
    """Faire du clustering sur les métriques de base."""
    
    df = pd.DataFrame(episode_features)
    
    # Sélectionner les features pour le clustering
    feature_columns = [
        'episode_length', 'action_std', 'action_range', 'action_velocity',
        'duration_seconds', 'action_complexity', 'trajectory_smoothness'
    ]
    
    X = df[feature_columns].values
    
    # Standardiser
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # t-SNE pour visualisation
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    df['tsne_x'] = X_tsne[:, 0]
    df['tsne_y'] = X_tsne[:, 1]
    
    # Calculer des scores composites
    df['efficiency_score'] = calculate_efficiency_score(df)
    df['quality_score'] = calculate_quality_score(df)
    df['overall_score'] = (df['efficiency_score'] + df['quality_score']) / 2
    
    return df

def calculate_efficiency_score(df):
    """Score d'efficacité basé sur durée et vitesse."""
    # Normaliser (plus court + plus fluide = mieux)
    duration_norm = 1 - (df['duration_seconds'] - df['duration_seconds'].min()) / (df['duration_seconds'].max() - df['duration_seconds'].min() + 1e-8)
    smoothness_norm = (df['trajectory_smoothness'] - df['trajectory_smoothness'].min()) / (df['trajectory_smoothness'].max() - df['trajectory_smoothness'].min() + 1e-8)
    
    return (duration_norm + smoothness_norm) / 2

def calculate_quality_score(df):
    """Score de qualité basé sur complexité et consistance."""
    # Normaliser (complexité modérée + faible variance = bien)
    complexity_norm = 1 - abs(df['action_complexity'] - df['action_complexity'].median()) / (df['action_complexity'].std() + 1e-8)
    velocity_norm = 1 - (df['action_velocity'] - df['action_velocity'].min()) / (df['action_velocity'].max() - df['action_velocity'].min() + 1e-8)
    
    return (complexity_norm + velocity_norm) / 2

def create_comprehensive_visualization(df):
    """Créer une visualisation complète de tous les épisodes."""
    
    # Configuration matplotlib non-interactive
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Clustering t-SNE
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['tsne_x'], df['tsne_y'], c=df['cluster'], 
                         cmap='viridis', s=100, alpha=0.7)
    for _, row in df.iterrows():
        ax1.annotate(f"Ep{int(row['episode_id'])}", 
                    (row['tsne_x'], row['tsne_y']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax1.set_title('Clustering t-SNE (Tous les épisodes)')
    plt.colorbar(scatter, ax=ax1)
    
    # 2. Durée vs Efficacité
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(df['duration_seconds'], df['efficiency_score'], 
                          c=df['overall_score'], cmap='RdYlGn', s=100, alpha=0.7)
    for _, row in df.iterrows():
        ax2.annotate(f"Ep{int(row['episode_id'])}", 
                    (row['duration_seconds'], row['efficiency_score']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Durée (secondes)')
    ax2.set_ylabel('Score d\'efficacité')
    ax2.set_title('Durée vs Efficacité')
    plt.colorbar(scatter2, ax=ax2)
    
    # 3. Complexité vs Qualité
    ax3 = axes[0, 2]
    scatter3 = ax3.scatter(df['action_complexity'], df['quality_score'], 
                          c=df['overall_score'], cmap='RdYlGn', s=100, alpha=0.7)
    for _, row in df.iterrows():
        ax3.annotate(f"Ep{int(row['episode_id'])}", 
                    (row['action_complexity'], row['quality_score']),
                    xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax3.set_xlabel('Complexité d\'action')
    ax3.set_ylabel('Score de qualité')
    ax3.set_title('Complexité vs Qualité')
    plt.colorbar(scatter3, ax=ax3)
    
    # 4. Distribution des scores par cluster
    ax4 = axes[1, 0]
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]['overall_score']
        ax4.hist(cluster_data, alpha=0.6, label=f'Cluster {cluster_id}', bins=8)
    ax4.set_xlabel('Score Global')
    ax4.set_ylabel('Nombre d\'épisodes')
    ax4.set_title('Distribution des scores par cluster')
    ax4.legend()
    
    # 5. Top épisodes
    ax5 = axes[1, 1]
    df_sorted = df.sort_values('overall_score', ascending=True)
    colors = ['red' if score >= 0.7 else 'orange' if score >= 0.5 else 'gray' 
              for score in df_sorted['overall_score']]
    bars = ax5.barh(range(len(df_sorted)), df_sorted['overall_score'], color=colors, alpha=0.7)
    ax5.set_yticks(range(len(df_sorted)))
    ax5.set_yticklabels([f"Ep{int(x)}" for x in df_sorted['episode_id']])
    ax5.set_xlabel('Score Global')
    ax5.set_title('Classement des épisodes')
    
    # 6. Matrice de corrélation
    ax6 = axes[1, 2]
    correlation_cols = ['episode_length', 'action_std', 'trajectory_smoothness', 
                       'efficiency_score', 'quality_score', 'overall_score']
    corr_matrix = df[correlation_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6, fmt='.2f')
    ax6.set_title('Corrélations entre métriques')
    
    plt.tight_layout()
    output_file = '/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/all_episodes_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Analyse complète sauvée : {output_file}")
    return output_file

def print_comprehensive_results(df):
    """Afficher les résultats détaillés."""
    
    print("\n🎯 ANALYSE COMPLÈTE DE TOUS LES ÉPISODES")
    print("=" * 80)
    
    print(f"\n📊 STATISTIQUES GÉNÉRALES:")
    print(f"   Épisodes analysés: {len(df)}")
    print(f"   Durée moyenne: {df['duration_seconds'].mean():.1f} secondes")
    print(f"   Durée min/max: {df['duration_seconds'].min():.1f} - {df['duration_seconds'].max():.1f} secondes")
    
    # Top 5 épisodes
    print(f"\n🏆 TOP 5 ÉPISODES (score global):")
    top_5 = df.nlargest(5, 'overall_score')
    for i, (_, row) in enumerate(top_5.iterrows()):
        medal = "🥇🥈🥉  "[i] if i < 3 else "  "
        print(f"   {medal} #{i+1} - Épisode {int(row['episode_id'])}: "
              f"Score={row['overall_score']:.3f} "
              f"(Efficacité={row['efficiency_score']:.3f}, "
              f"Qualité={row['quality_score']:.3f})")
    
    # Analyse par cluster
    print(f"\n🎯 ANALYSE PAR CLUSTER:")
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        episodes = [int(x) for x in cluster_df['episode_id'].tolist()]
        avg_score = cluster_df['overall_score'].mean()
        avg_duration = cluster_df['duration_seconds'].mean()
        
        print(f"   Cluster {cluster_id}: {len(episodes)} épisodes")
        print(f"     Épisodes: {episodes}")
        print(f"     Score moyen: {avg_score:.3f}")
        print(f"     Durée moyenne: {avg_duration:.1f}s")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    excellent = df[df['overall_score'] >= 0.7]['episode_id'].tolist()
    good = df[(df['overall_score'] >= 0.5) & (df['overall_score'] < 0.7)]['episode_id'].tolist()
    poor = df[df['overall_score'] < 0.5]['episode_id'].tolist()
    
    if excellent:
        print(f"   🏆 EXCELLENTS (≥0.7): Épisodes {[int(x) for x in excellent]}")
        print(f"      👉 Utiliser en priorité pour l'entraînement")
    
    if good:
        print(f"   👍 BONS (0.5-0.7): Épisodes {[int(x) for x in good]}")
        print(f"      👉 Bons compléments pour diversifier")
    
    if poor:
        print(f"   ⚠️  FAIBLES (<0.5): Épisodes {[int(x) for x in poor]}")
        print(f"      👉 À éviter ou analyser les problèmes")

def main():
    parser = argparse.ArgumentParser(description="Analyser TOUS les épisodes sans dépendre de l'attention")
    parser.add_argument("--dataset-repo-id", type=str, 
                       default="PLB/phospho-playground-mono",
                       help="Repository ID du dataset")
    parser.add_argument("--max-episodes", type=int, default=50,
                       help="Nombre maximum d'épisodes à analyser")
    parser.add_argument("--max-frames", type=int, default=30,
                       help="Nombre maximum de frames par épisode")
    parser.add_argument("--n-clusters", type=int, default=4,
                       help="Nombre de clusters")
    
    args = parser.parse_args()
    
    print("🎯 ANALYSE COMPLÈTE DE TOUS LES ÉPISODES")
    print("=" * 60)
    print("ℹ️  Cette analyse N'UTILISE PAS la capture d'attention")
    print("ℹ️  Métriques: durée, actions, trajectoires, complexité")
    
    # Charger le dataset
    print(f"\n📥 Chargement du dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    print(f"✅ Dataset chargé: {dataset.num_episodes} épisodes disponibles")
    
    # Analyser tous les épisodes
    num_episodes = min(args.max_episodes, dataset.num_episodes)
    episode_features = []
    
    print(f"\n🔄 Analyse de {num_episodes} épisodes...")
    for episode_id in tqdm(range(num_episodes), desc="Épisodes"):
        features = analyze_episode_basic(dataset, episode_id, args.max_frames)
        if features:
            episode_features.append(features)
    
    if not episode_features:
        print("❌ Aucun épisode analysé avec succès")
        return
    
    print(f"✅ {len(episode_features)} épisodes analysés avec succès")
    
    # Clustering et analyse
    print(f"\n📊 Clustering et calcul des scores...")
    df = create_basic_clustering(episode_features, args.n_clusters)
    
    # Créer les visualisations
    print(f"\n🎨 Génération des visualisations...")
    create_comprehensive_visualization(df)
    
    # Afficher les résultats
    print_comprehensive_results(df)
    
    # Sauver les données
    output_csv = '/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/all_episodes_data.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Données détaillées sauvées: {output_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()