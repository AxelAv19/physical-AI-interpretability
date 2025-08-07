#!/usr/bin/env python

"""
Script pour récupérer les vraies données des épisodes et faire une analyse objective.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from src.attention_maps import ACTPolicyWithAttention
from tqdm import tqdm

def prepare_observation_for_policy(frame: dict, 
                                 device: torch.device, 
                                 model_dtype: torch.dtype = torch.float32) -> dict:
    """Convert dataset frame to policy observation format."""
    observation = {}
    
    for key, value in frame.items():
        if "image" in key:
            # Convert image to policy format
            if isinstance(value, torch.Tensor):
                # Remove any extra batch dimensions first
                while value.dim() > 3:
                    value = value.squeeze(0)
                
                # Convert to (C, H, W) format
                if value.dim() == 3:
                    h, w, c = value.shape
                    if c in [1, 3]:  # Standard (H, W, C) format
                        value = value.permute(2, 0, 1)
                
                # Ensure float and normalize
                if value.dtype != model_dtype:
                    value = value.type(model_dtype)
                
                if value.max() > 1.0:
                    value = value / 255.0
            
            observation[key] = value.unsqueeze(0).to(device)  # Add batch dimension
            
        elif key in ["observation.state", "robot_state", "state"]:
            # Proprioceptive state
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value).type(model_dtype)
            observation[key] = value.unsqueeze(0).to(device)
    
    return observation

def load_policy(policy_path: str, dataset_meta, policy_overrides: list = None):
    """Load and initialize a policy from checkpoint."""
    
    # Load regular LeRobot policy
    if policy_overrides:
        # Convert list of "key=value" strings to dict
        overrides = {}
        for override in policy_overrides:
            key, value = override.split('=', 1)
            overrides[key] = value
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, **overrides)
    else:
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path

    # NOTE: policy has to be an ACT policy for this to work
    policy = make_policy(policy_cfg, ds_meta=dataset_meta)
    policy = ACTPolicyWithAttention(policy)
        
    return policy, policy_cfg

def extract_episode_features(dataset: LeRobotDataset, 
                           policy, 
                           episode_id: int,
                           device: torch.device,
                           model_dtype: torch.dtype = torch.float32,
                           max_frames: int = 50):
    """Extract features from an episode for clustering analysis."""
    
    # Get episode info from dataset
    try:
        episode_info = dataset.episode_data_index["from"][episode_id], dataset.episode_data_index["to"][episode_id]
        episode_start, episode_end = episode_info
        episode_length = episode_end - episode_start
        
        if episode_length == 0:
            return {"episode_id": episode_id, "valid": False}
        
        # Limit frames for performance
        frames_to_analyze = min(episode_length, max_frames)
        # Get evenly spaced frame indices within the episode
        episode_frame_indices = np.linspace(episode_start, episode_end - 1, frames_to_analyze, dtype=int)
        
    except (KeyError, IndexError, AttributeError):
        return {"episode_id": episode_id, "valid": False}
    
    # Initialize feature collectors
    attention_stats = []
    attention_entropy = []
    attention_max_values = []
    attention_spatial_concentration = []
    
    # Process frames
    for dataset_idx in episode_frame_indices[:5]:  # Limit to 5 frames for speed
        try:
            # Convert to regular int to avoid numpy type issues
            dataset_idx = int(dataset_idx)
            frame = dataset[dataset_idx]
            
            # Prepare observation
            observation = prepare_observation_for_policy(frame, device, model_dtype)
            
            # Run policy inference
            with torch.inference_mode():
                if hasattr(policy, 'select_action'):
                    result = policy.select_action(observation)
                    
                    if isinstance(result, tuple):
                        action, attention_maps = result
                        
                        if attention_maps:
                            # Extract attention statistics
                            valid_maps = [m for m in attention_maps if m is not None]
                            if valid_maps:
                                # Combine all attention maps for analysis
                                combined_attention = np.concatenate([m.flatten() for m in valid_maps])
                                
                                # Compute features
                                attention_stats.append({
                                    'mean': np.mean(combined_attention),
                                    'std': np.std(combined_attention),
                                    'max': np.max(combined_attention),
                                    'min': np.min(combined_attention)
                                })
                                
                                # Attention entropy (measure of focus vs. diffusion)
                                normalized_attention = combined_attention / (combined_attention.sum() + 1e-10)
                                entropy = -np.sum(normalized_attention * np.log(normalized_attention + 1e-10))
                                attention_entropy.append(entropy)
                                
                                # Maximum attention value
                                attention_max_values.append(np.max(combined_attention))
                                
                                # Spatial concentration (how concentrated is attention)
                                # Higher values = more concentrated attention
                                concentration = np.sum(combined_attention ** 2) / (np.sum(combined_attention) ** 2 + 1e-10)
                                attention_spatial_concentration.append(concentration)
                        
        except Exception as e:
            continue
    
    # Aggregate features
    if not attention_stats:
        return {"episode_id": episode_id, "valid": False}
    
    # Compute episode-level features
    features = {
        "episode_id": episode_id,
        "valid": True,
        "episode_length": int(episode_length),  # Convert to int
        "frames_analyzed": len(attention_stats),
        
        # Basic attention statistics (convert all to float)
        "attention_mean_avg": float(np.mean([s['mean'] for s in attention_stats])),
        "attention_mean_std": float(np.std([s['mean'] for s in attention_stats])),
        "attention_max_avg": float(np.mean([s['max'] for s in attention_stats])),
        "attention_max_std": float(np.std([s['max'] for s in attention_stats])),
        "attention_max_peak": float(np.max(attention_max_values)),
        
        # Attention dynamics (convert all to float)
        "attention_entropy_avg": float(np.mean(attention_entropy)),
        "attention_entropy_std": float(np.std(attention_entropy)),
        "attention_concentration_avg": float(np.mean(attention_spatial_concentration)),
        "attention_concentration_std": float(np.std(attention_spatial_concentration)),
        
        # Episode characteristics
        "duration_category": "short" if episode_length < 100 else "medium" if episode_length < 200 else "long",
        "attention_category": "low" if np.max(attention_max_values) < 0.5 else "medium" if np.max(attention_max_values) < 0.8 else "high"
    }
    
    return features

def load_and_analyze_episodes(dataset_repo_id: str, policy_path: str, max_episodes: int = 15):
    """Charger les données d'épisodes et les analyser objectivement."""
    
    print("🔄 Chargement des données d'épisodes...")
    
    # Load dataset and policy
    device = torch.device("mps")
    dataset = LeRobotDataset(dataset_repo_id)
    policy, _ = load_policy(policy_path, dataset.meta)
    
    if hasattr(policy, 'model'):
        policy.model.eval()
        policy.model.to(device)
    
    # Extract features from episodes
    episode_features = []
    for episode_id in range(min(max_episodes, dataset.num_episodes)):
        print(f"Extraction épisode {episode_id}...")
        features = extract_episode_features(
            dataset=dataset,
            policy=policy, 
            episode_id=episode_id,
            device=device,
            max_frames=25
        )
        episode_features.append(features)
    
    # Convert to DataFrame
    valid_features = [f for f in episode_features if f.get('valid', False)]
    if not valid_features:
        print("❌ Aucun épisode valide trouvé")
        return None
        
    df = pd.DataFrame(valid_features)
    return df

def analyze_episode_rankings(df: pd.DataFrame):
    """Analyser et classer les épisodes basé sur les vraies métriques."""
    
    print("\n📊 ANALYSE DES DONNÉES RÉELLES")
    print("="*60)
    
    # Display basic statistics
    print("\n📈 Statistiques de base:")
    numerical_cols = ['episode_length', 'attention_mean_avg', 'attention_max_peak', 
                     'attention_entropy_avg', 'attention_concentration_avg']
    
    for col in numerical_cols:
        if col in df.columns:
            print(f"{col:25}: {df[col].mean():.3f} ± {df[col].std():.3f} (range: {df[col].min():.3f} - {df[col].max():.3f})")
    
    # Create composite quality scores
    df['efficiency_score'] = rank_by_efficiency(df)
    df['focus_score'] = rank_by_focus(df) 
    df['overall_score'] = (df['efficiency_score'] + df['focus_score']) / 2
    
    # Sort by overall quality
    df_ranked = df.sort_values('overall_score', ascending=False)
    
    print("\n🏆 CLASSEMENT DES ÉPISODES (données réelles):")
    print("="*60)
    
    for i, (idx, row) in enumerate(df_ranked.iterrows()):
        episode_id = int(row['episode_id'])
        length = int(row['episode_length']) if 'episode_length' in row else 'N/A'
        attention_peak = row.get('attention_max_peak', 0)
        efficiency = row.get('efficiency_score', 0)
        focus = row.get('focus_score', 0)
        overall = row.get('overall_score', 0)
        
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        
        print(f"{medal} #{i+1:2d} - Épisode {episode_id:2d}: "
              f"Durée={length}, Attention={attention_peak:.3f}, "
              f"Efficacité={efficiency:.2f}, Focus={focus:.2f}, Overall={overall:.2f}")
    
    return df_ranked

def rank_by_efficiency(df: pd.DataFrame) -> pd.Series:
    """Score d'efficacité: attention élevée avec durée raisonnable."""
    if 'attention_max_peak' not in df.columns or 'episode_length' not in df.columns:
        return pd.Series([0] * len(df))
    
    # Normalize metrics
    attention_norm = (df['attention_max_peak'] - df['attention_max_peak'].min()) / (df['attention_max_peak'].max() - df['attention_max_peak'].min() + 1e-8)
    length_norm = (df['episode_length'].max() - df['episode_length']) / (df['episode_length'].max() - df['episode_length'].min() + 1e-8)  # Shorter is better
    
    return (attention_norm + length_norm) / 2

def rank_by_focus(df: pd.DataFrame) -> pd.Series:
    """Score de focus: attention concentrée (faible entropie, haute concentration)."""
    if 'attention_concentration_avg' not in df.columns:
        return pd.Series([0] * len(df))
    
    # High concentration = good focus
    concentration_norm = (df['attention_concentration_avg'] - df['attention_concentration_avg'].min()) / (df['attention_concentration_avg'].max() - df['attention_concentration_avg'].min() + 1e-8)
    
    # Low entropy = good focus (if available)
    if 'attention_entropy_avg' in df.columns:
        entropy_norm = (df['attention_entropy_avg'].max() - df['attention_entropy_avg']) / (df['attention_entropy_avg'].max() - df['attention_entropy_avg'].min() + 1e-8)
        return (concentration_norm + entropy_norm) / 2
    else:
        return concentration_norm

def create_real_data_visualization(df: pd.DataFrame):
    """Créer des visualisations basées sur les vraies données."""
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Duration vs Attention Peak
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['episode_length'], df['attention_max_peak'], 
                         c=df['overall_score'], cmap='RdYlGn', s=100, alpha=0.7)
    ax1.set_xlabel('Durée (frames)')
    ax1.set_ylabel('Pic d\'attention max')
    ax1.set_title('Durée vs Attention (couleur = score global)')
    
    # Add episode labels
    for _, row in df.iterrows():
        ax1.annotate(f"{int(row['episode_id'])}", 
                    (row['episode_length'], row['attention_max_peak']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.colorbar(scatter, ax=ax1)
    
    # 2. Focus metrics
    ax2 = axes[0, 1] 
    if 'attention_concentration_avg' in df.columns and 'attention_entropy_avg' in df.columns:
        scatter2 = ax2.scatter(df['attention_concentration_avg'], df['attention_entropy_avg'],
                              c=df['focus_score'], cmap='RdYlGn', s=100, alpha=0.7)
        ax2.set_xlabel('Concentration moyenne')
        ax2.set_ylabel('Entropie moyenne')
        ax2.set_title('Concentration vs Entropie')
        plt.colorbar(scatter2, ax=ax2)
        
        for _, row in df.iterrows():
            ax2.annotate(f"{int(row['episode_id'])}", 
                        (row['attention_concentration_avg'], row['attention_entropy_avg']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 3. Efficiency ranking
    ax3 = axes[1, 0]
    df_sorted = df.sort_values('efficiency_score', ascending=True)
    bars1 = ax3.barh(range(len(df_sorted)), df_sorted['efficiency_score'], 
                     color='skyblue', alpha=0.7)
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels([f"Ep {int(x)}" for x in df_sorted['episode_id']])
    ax3.set_xlabel('Score d\'efficacité')
    ax3.set_title('Classement par efficacité')
    
    # 4. Overall ranking  
    ax4 = axes[1, 1]
    df_sorted2 = df.sort_values('overall_score', ascending=True)
    bars2 = ax4.barh(range(len(df_sorted2)), df_sorted2['overall_score'],
                     color='lightgreen', alpha=0.7)
    ax4.set_yticks(range(len(df_sorted2)))
    ax4.set_yticklabels([f"Ep {int(x)}" for x in df_sorted2['episode_id']])
    ax4.set_xlabel('Score global')
    ax4.set_title('Classement global')
    
    plt.tight_layout()
    plt.savefig('/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/real_data_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking

def get_max_episodes_from_clustering():
    """Déterminer le nombre d'épisodes à analyser basé sur les données de clustering récentes."""
    import glob
    import pandas as pd
    
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

def main():
    # Default parameters based on your setup
    dataset_repo_id = "PLB/phospho-playground-mono"
    policy_path = "/Users/askel/Documents/GitHub/pro2robot/resources/lerobot/src/lerobot/resources/outputs/train/act_so101_PLB_2/checkpoints/last/pretrained_model"
    
    print("🎯 ANALYSE DES ÉPISODES AVEC DONNÉES RÉELLES")
    print("="*60)
    
    # Déterminer le nombre d'épisodes à analyser
    max_episodes = get_max_episodes_from_clustering()
    
    # Load real data
    try:
        df = load_and_analyze_episodes(dataset_repo_id, policy_path, max_episodes=max_episodes)
        if df is None:
            return
            
        # Analyze and rank episodes
        df_ranked = analyze_episode_rankings(df)
        
        # Create visualizations
        print("\n🎨 Génération des visualisations...")
        create_real_data_visualization(df_ranked)
        
        # Save results
        output_file = '/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/real_episode_data.csv'
        df_ranked.to_csv(output_file, index=False)
        print(f"\n✅ Données sauvées: {output_file}")
        
        print(f"\n🎯 RECOMMANDATIONS FINALES:")
        top_3 = df_ranked.head(3)
        for i, (_, row) in enumerate(top_3.iterrows()):
            print(f"  {i+1}. Épisode {int(row['episode_id'])} - Score: {row['overall_score']:.3f}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()