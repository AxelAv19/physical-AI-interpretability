#!/usr/bin/env python

"""
Tableau de clustering simple avec efficiency_score et focus_score pour lecture immédiate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from src.attention_maps import ACTPolicyWithAttention
from tqdm import tqdm

def get_max_episodes_from_clustering():
    """Déterminer le nombre d'épisodes à analyser basé sur les données de clustering récentes."""
    import glob
    
    # Find the most recent clustering analysis file
    csv_files = glob.glob('/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/episode_analysis_*.csv')
    
    if csv_files:
        try:
            # Get the most recent file
            latest_file = max(csv_files, key=lambda x: x.split('_')[-1])
            df = pd.read_csv(latest_file)
            num_episodes = len(df)
            print(f"📊 Référence clustering récent trouvée: {num_episodes} épisodes")
            return num_episodes
        except Exception as e:
            print(f"⚠️ Erreur lecture clustering récent: {e}")
    
    print("📊 Aucun clustering récent trouvé, utilisation par défaut: 15 épisodes")
    return 15

def prepare_observation_for_policy(frame: dict, 
                                 device: torch.device, 
                                 model_dtype: torch.dtype = torch.float32) -> dict:
    """Convert dataset frame to policy observation format."""
    observation = {}
    
    for key, value in frame.items():
        if "image" in key:
            if isinstance(value, torch.Tensor):
                while value.dim() > 3:
                    value = value.squeeze(0)
                
                if value.dim() == 3:
                    h, w, c = value.shape
                    if c in [1, 3]:
                        value = value.permute(2, 0, 1)
                
                if value.dtype != model_dtype:
                    value = value.type(model_dtype)
                
                if value.max() > 1.0:
                    value = value / 255.0
            
            observation[key] = value.unsqueeze(0).to(device)
            
        elif key in ["observation.state", "robot_state", "state"]:
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value).type(model_dtype)
            observation[key] = value.unsqueeze(0).to(device)
    
    return observation

def load_policy(policy_path: str, dataset_meta):
    """Load and initialize a policy from checkpoint."""
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy = make_policy(policy_cfg, ds_meta=dataset_meta)
    policy = ACTPolicyWithAttention(policy)
    return policy

def extract_episode_scores(dataset: LeRobotDataset, policy, episode_id: int, device: torch.device):
    """Extract efficiency and focus scores for an episode."""
    
    try:
        episode_info = dataset.episode_data_index["from"][episode_id], dataset.episode_data_index["to"][episode_id]
        episode_start, episode_end = episode_info
        episode_length = episode_end - episode_start
        
        if episode_length == 0:
            return None
        
        # Sample 5 frames for quick analysis
        episode_frame_indices = np.linspace(episode_start, episode_end - 1, 5, dtype=int)
        
    except (KeyError, IndexError, AttributeError):
        return None
    
    attention_stats = []
    attention_max_values = []
    attention_entropy = []
    attention_concentration = []
    
    # Process frames
    for dataset_idx in episode_frame_indices:
        try:
            dataset_idx = int(dataset_idx)
            frame = dataset[dataset_idx]
            observation = prepare_observation_for_policy(frame, device)
            
            with torch.inference_mode():
                if hasattr(policy, 'select_action'):
                    result = policy.select_action(observation)
                    
                    if isinstance(result, tuple):
                        action, attention_maps = result
                        
                        if attention_maps:
                            valid_maps = [m for m in attention_maps if m is not None]
                            if valid_maps:
                                combined_attention = np.concatenate([m.flatten() for m in valid_maps])
                                
                                attention_max_values.append(np.max(combined_attention))
                                
                                # Attention entropy
                                normalized_attention = combined_attention / (combined_attention.sum() + 1e-10)
                                entropy = -np.sum(normalized_attention * np.log(normalized_attention + 1e-10))
                                attention_entropy.append(entropy)
                                
                                # Spatial concentration
                                concentration = np.sum(combined_attention ** 2) / (np.sum(combined_attention) ** 2 + 1e-10)
                                attention_concentration.append(concentration)
                        
        except Exception:
            continue
    
    if not attention_max_values:
        return None
    
    # Calculate scores
    max_attention = np.max(attention_max_values)
    avg_entropy = np.mean(attention_entropy)
    avg_concentration = np.mean(attention_concentration)
    
    # Efficiency score: high attention + short duration
    # Normalize attention (assuming max possible is 1.0)
    attention_norm = max_attention
    # Normalize duration (shorter is better, assume max reasonable is 500 frames)
    duration_norm = max(0, (500 - episode_length) / 500)
    efficiency_score = (attention_norm + duration_norm) / 2
    
    # Focus score: high concentration + low entropy
    # Normalize concentration (higher is better)
    concentration_norm = avg_concentration  # Already between 0-1 typically
    # Normalize entropy (lower is better, assume max reasonable entropy is 50)
    entropy_norm = max(0, (50 - avg_entropy) / 50)
    focus_score = (concentration_norm + entropy_norm) / 2
    
    return {
        'episode_id': episode_id,
        'episode_length': episode_length,
        'attention_max': max_attention,
        'entropy_avg': avg_entropy,
        'concentration_avg': avg_concentration,
        'efficiency_score': efficiency_score,
        'focus_score': focus_score
    }

def create_simple_clustering_table(max_episodes: int = None):
    """Créer un tableau de clustering simple avec efficiency vs focus."""
    
    if max_episodes is None:
        max_episodes = get_max_episodes_from_clustering()
    
    print("🎯 CRÉATION DU TABLEAU DE CLUSTERING SIMPLE")
    print("=" * 60)
    
    # Load dataset and policy
    dataset_repo_id = "PLB/phospho-playground-mono"
    policy_path = "/Users/askel/Documents/GitHub/pro2robot/resources/lerobot/src/lerobot/resources/outputs/train/act_so101_PLB_2/checkpoints/last/pretrained_model"
    device = torch.device("mps")
    
    dataset = LeRobotDataset(dataset_repo_id)
    policy = load_policy(policy_path, dataset.meta)
    
    if hasattr(policy, 'model'):
        policy.model.eval()
        policy.model.to(device)
    
    # Extract scores for all episodes
    episode_data = []
    for episode_id in tqdm(range(min(max_episodes, dataset.num_episodes)), desc="Extraction scores"):
        scores = extract_episode_scores(dataset, policy, episode_id, device)
        if scores:
            episode_data.append(scores)
    
    if not episode_data:
        print("❌ Aucune donnée extraite")
        return
    
    df = pd.DataFrame(episode_data)
    
    # Classification en catégories
    def classify_efficiency(score):
        if score >= 0.9: return "Très efficace"
        elif score >= 0.7: return "Efficace" 
        elif score >= 0.5: return "Modéré"
        else: return "Inefficace"
    
    def classify_focus(score):
        if score >= 0.9: return "Très focalisé"
        elif score >= 0.7: return "Bien focalisé"
        elif score >= 0.5: return "Focus modéré"
        else: return "Attention dispersée"
    
    df['efficiency_category'] = df['efficiency_score'].apply(classify_efficiency)
    df['focus_category'] = df['focus_score'].apply(classify_focus)
    
    # Afficher le tableau
    print("\n📊 TABLEAU DE CLUSTERING SIMPLIFIÉ")
    print("=" * 90)
    print(f"{'Épisode':>7} {'Efficacité':>12} {'Focus':>12} {'Catégorie Efficacité':>18} {'Catégorie Focus':>18}")
    print("-" * 90)
    
    # Trier par score global
    df['overall_score'] = (df['efficiency_score'] + df['focus_score']) / 2
    df_sorted = df.sort_values('overall_score', ascending=False)
    
    for _, row in df_sorted.iterrows():
        ep_id = int(row['episode_id'])
        eff_score = row['efficiency_score']
        focus_score = row['focus_score']
        eff_cat = row['efficiency_category']
        focus_cat = row['focus_category']
        
        print(f"{ep_id:>7} {eff_score:>12.3f} {focus_score:>12.3f} {eff_cat:>18} {focus_cat:>18}")
    
    # Créer la matrice de clustering visuelle
    create_clustering_matrix(df_sorted)
    create_scatter_plot(df_sorted)
    
    return df_sorted

def create_clustering_matrix(df):
    """Créer une matrice de clustering visuelle."""
    
    # Créer des bins pour la matrice
    efficiency_bins = ['Inefficace', 'Modéré', 'Efficace', 'Très efficace']
    focus_bins = ['Attention dispersée', 'Focus modéré', 'Bien focalisé', 'Très focalisé']
    
    # Créer la matrice de comptage
    matrix = pd.crosstab(df['focus_category'], df['efficiency_category'])
    
    # Réordonner les colonnes et indices
    matrix = matrix.reindex(index=focus_bins, columns=efficiency_bins, fill_value=0)
    
    # Créer la heatmap
    plt.figure(figsize=(12, 8))
    
    # Heatmap avec annotations des épisodes
    ax = sns.heatmap(matrix, annot=True, cmap='RdYlGn', center=0, 
                     square=True, linewidths=1, cbar_kws={"shrink": .8})
    
    # Ajouter les numéros d'épisodes dans chaque case
    for i, focus_cat in enumerate(focus_bins):
        for j, eff_cat in enumerate(efficiency_bins):
            episodes = df[(df['focus_category'] == focus_cat) & 
                         (df['efficiency_category'] == eff_cat)]['episode_id'].tolist()
            if episodes:
                episode_text = ', '.join([f"Ep{int(ep)}" for ep in episodes])
                ax.text(j + 0.5, i + 0.7, episode_text, ha='center', va='center',
                       fontsize=10, fontweight='bold', color='black')
    
    plt.title('🎯 MATRICE DE CLUSTERING : EFFICACITÉ vs FOCUS\n' + 
              '(Nombres = épisodes par catégorie)', fontsize=14, fontweight='bold')
    plt.xlabel('EFFICACITÉ (Attention + Vitesse)', fontsize=12)
    plt.ylabel('FOCUS (Concentration)', fontsize=12)
    plt.tight_layout()
    
    output_file = '/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/simple_clustering_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    
    print(f"\n✅ Matrice de clustering sauvée : {output_file}")

def create_scatter_plot(df):
    """Créer un scatter plot avec les scores et épisodes."""
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot coloré par score global
    scatter = plt.scatter(df['efficiency_score'], df['focus_score'], 
                         c=df['overall_score'], cmap='RdYlGn', 
                         s=150, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Ajouter les numéros d'épisodes
    for _, row in df.iterrows():
        plt.annotate(f"Ep{int(row['episode_id'])}", 
                    (row['efficiency_score'], row['focus_score']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')
    
    # Ajouter les lignes de séparation des catégories
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    
    # Labels et zones
    plt.text(0.25, 0.95, 'LENT\n+ Dispersé', ha='center', va='top', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    plt.text(0.95, 0.95, 'EXCELLENT\nRapide + Focalisé', ha='center', va='top',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    plt.text(0.25, 0.25, 'PROBLÉMATIQUE\nLent + Dispersé', ha='center', va='center',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    plt.text(0.95, 0.25, 'RAPIDE\nmais Dispersé', ha='center', va='center',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    plt.colorbar(scatter, label='Score Global')
    plt.xlabel('EFFICACITÉ SCORE (0-1)', fontsize=12)
    plt.ylabel('FOCUS SCORE (0-1)', fontsize=12)
    plt.title('🎯 CLUSTERING SIMPLE : EFFICACITÉ vs FOCUS\n' +
              'Chaque point = 1 épisode', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    output_file = '/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/simple_clustering_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    
    print(f"✅ Scatter plot sauvé : {output_file}")

def print_recommendations(df):
    """Afficher les recommandations basées sur le clustering."""
    
    print("\n🎯 RECOMMANDATIONS PAR ZONE DE CLUSTERING")
    print("=" * 60)
    
    # Zone excellente (efficace + focalisé)
    excellent = df[(df['efficiency_score'] >= 0.7) & (df['focus_score'] >= 0.7)]
    if not excellent.empty:
        print("\n🏆 ZONE EXCELLENTE (Efficace + Focalisé) :")
        episodes = excellent['episode_id'].tolist()
        print(f"   Épisodes : {[int(ep) for ep in episodes]}")
        print("   👉 À UTILISER EN PRIORITÉ pour l'entraînement")
    
    # Zone efficace mais pas focalisé
    fast_dispersed = df[(df['efficiency_score'] >= 0.7) & (df['focus_score'] < 0.7)]
    if not fast_dispersed.empty:
        print("\n⚡ ZONE RAPIDE mais DISPERSÉE :")
        episodes = fast_dispersed['episode_id'].tolist()
        print(f"   Épisodes : {[int(ep) for ep in episodes]}")
        print("   👉 ANALYSER : Rapides mais manquent de précision")
    
    # Zone focalisé mais pas efficace
    slow_focused = df[(df['efficiency_score'] < 0.7) & (df['focus_score'] >= 0.7)]
    if not slow_focused.empty:
        print("\n🐌 ZONE LENTE mais FOCALISÉE :")
        episodes = slow_focused['episode_id'].tolist()
        print(f"   Épisodes : {[int(ep) for ep in episodes]}")
        print("   👉 ÉTUDIER : Bonne précision mais trop lents")
    
    # Zone problématique
    problematic = df[(df['efficiency_score'] < 0.5) & (df['focus_score'] < 0.5)]
    if not problematic.empty:
        print("\n❌ ZONE PROBLÉMATIQUE :")
        episodes = problematic['episode_id'].tolist()
        print(f"   Épisodes : {[int(ep) for ep in episodes]}")
        print("   👉 À ÉVITER : Lents et imprécis")

def main():
    try:
        df = create_simple_clustering_table()  # Will automatically detect episode count
        if df is not None:
            print_recommendations(df)
            print(f"\n🎯 RÉSUMÉ FINAL :")
            print(f"   📊 {len(df)} épisodes analysés")
            best_episode = df.iloc[0]
            print(f"   🏆 Meilleur épisode : {int(best_episode['episode_id'])} " +
                  f"(Efficacité: {best_episode['efficiency_score']:.3f}, " +
                  f"Focus: {best_episode['focus_score']:.3f})")
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    main()