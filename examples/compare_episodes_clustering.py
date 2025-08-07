#!/usr/bin/env python

"""
Script to compare and cluster episodes from a dataset based on attention patterns.
Generates clustering visualizations and analysis using matplotlib.
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig

from src.attention_maps import ACTPolicyWithAttention


def extract_episode_features(dataset: LeRobotDataset, 
                           policy, 
                           episode_id: int,
                           device: torch.device,
                           model_dtype: torch.dtype = torch.float32,
                           max_frames: int = 50) -> Dict:
    """Extract features from an episode for clustering analysis.
    
    Args:
        dataset: The dataset
        policy: The policy with attention
        episode_id: Episode to analyze
        device: Device for computation
        model_dtype: Model data type
        max_frames: Maximum frames to analyze (for speed)
        
    Returns:
        Dictionary with episode features
    """
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
        # Fallback to the filter method if episode_data_index is not available
        episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
        episode_length = len(episode_frames)
        
        if episode_length == 0:
            return {"episode_id": episode_id, "valid": False}
        
        frames_to_analyze = min(episode_length, max_frames)
        episode_frame_indices = np.linspace(0, episode_length - 1, frames_to_analyze, dtype=int)
        # Convert to actual dataset indices
        episode_frame_indices = [episode_frames[i]['index'].item() for i in episode_frame_indices]
    
    print(f"Analyzing episode {episode_id}: {episode_length} frames ({frames_to_analyze} sampled)")
    
    # Initialize feature collectors
    attention_stats = []
    attention_entropy = []
    attention_max_values = []
    attention_spatial_concentration = []
    
    # Process frames
    for dataset_idx in tqdm(episode_frame_indices, desc=f"Episode {episode_id}", leave=False):
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
            print(f"Error processing frame {dataset_idx} in episode {episode_id}: {e}")
            continue
    
    # Aggregate features
    if not attention_stats:
        return {"episode_id": episode_id, "valid": False}
    
    # Compute episode-level features
    features = {
        "episode_id": episode_id,
        "valid": True,
        "episode_length": episode_length,
        "frames_analyzed": len(attention_stats),
        
        # Basic attention statistics
        "attention_mean_avg": np.mean([s['mean'] for s in attention_stats]),
        "attention_mean_std": np.std([s['mean'] for s in attention_stats]),
        "attention_max_avg": np.mean([s['max'] for s in attention_stats]),
        "attention_max_std": np.std([s['max'] for s in attention_stats]),
        "attention_max_peak": np.max(attention_max_values),
        
        # Attention dynamics
        "attention_entropy_avg": np.mean(attention_entropy),
        "attention_entropy_std": np.std(attention_entropy),
        "attention_concentration_avg": np.mean(attention_spatial_concentration),
        "attention_concentration_std": np.std(attention_spatial_concentration),
        
        # Episode characteristics
        "duration_category": "short" if episode_length < 100 else "medium" if episode_length < 200 else "long",
        "attention_category": "low" if np.max(attention_max_values) < 0.5 else "medium" if np.max(attention_max_values) < 0.8 else "high"
    }
    
    return features


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


def load_policy(policy_path: str, dataset_meta, policy_overrides: list = None) -> Tuple[torch.nn.Module, dict]:
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


def perform_clustering_analysis(episode_features: List[Dict], n_clusters: int = 5) -> Dict:
    """Perform clustering analysis on episode features.
    
    Args:
        episode_features: List of feature dictionaries
        n_clusters: Number of clusters for K-means
        
    Returns:
        Dictionary with clustering results
    """
    # Convert to DataFrame
    valid_features = [f for f in episode_features if f.get('valid', False)]
    if not valid_features:
        return {"error": "No valid episodes"}
    
    df = pd.DataFrame(valid_features)
    
    # Convert any PyTorch tensors to regular Python numbers
    for col in df.columns:
        if df[col].dtype == object:
            # Check if column contains tensors and convert them
            df[col] = df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
    
    # Select numerical features for clustering
    numerical_features = [
        'episode_length', 'attention_mean_avg', 'attention_mean_std',
        'attention_max_avg', 'attention_max_std', 'attention_max_peak',
        'attention_entropy_avg', 'attention_entropy_std',
        'attention_concentration_avg', 'attention_concentration_std'
    ]
    
    # Prepare data for clustering
    X = df[numerical_features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Perform dimensionality reduction for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Add results to dataframe
    df['cluster'] = cluster_labels
    df['tsne_x'] = X_tsne[:, 0]
    df['tsne_y'] = X_tsne[:, 1]
    df['pca_x'] = X_pca[:, 0]
    df['pca_y'] = X_pca[:, 1]
    
    return {
        "dataframe": df,
        "features": numerical_features,
        "n_clusters": n_clusters,
        "cluster_centers": kmeans.cluster_centers_,
        "scaler": scaler,
        "pca_explained_variance": pca.explained_variance_ratio_
    }


def create_visualizations(clustering_results: Dict, output_dir: str, show_plots: bool = False):
    """Create matplotlib visualizations for clustering analysis."""
    df = clustering_results["dataframe"]
    features = clustering_results["features"]
    
    # Set up matplotlib for non-interactive use
    if not show_plots:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # 1. Cluster visualization (t-SNE)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(df['tsne_x'], df['tsne_y'], c=df['cluster'], 
                         cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter)
    plt.title('Episode Clustering (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Add episode IDs as annotations
    for i, row in df.iterrows():
        plt.annotate(f"{row['episode_id']}", 
                    (row['tsne_x'], row['tsne_y']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(df['pca_x'], df['pca_y'], c=df['cluster'], 
                         cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter)
    plt.title('Episode Clustering (PCA)')
    plt.xlabel(f'PC1 ({clustering_results["pca_explained_variance"][0]:.2%} variance)')
    plt.ylabel(f'PC2 ({clustering_results["pca_explained_variance"][1]:.2%} variance)')
    
    plt.tight_layout()
    clustering_plot = f"{output_dir}/episode_clustering_{timestamp}.png"
    plt.savefig(clustering_plot, dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 2. Feature correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Episode Feature Correlations')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlations_{timestamp}.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 3. Episode characteristics by cluster
    plt.figure(figsize=(15, 10))
    
    # Duration vs Max Attention
    plt.subplot(2, 3, 1)
    scatter = plt.scatter(df['episode_length'], df['attention_max_peak'], 
                         c=df['cluster'], cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter)
    plt.xlabel('Episode Length (frames)')
    plt.ylabel('Max Attention Peak')
    plt.title('Duration vs Max Attention by Cluster')
    
    # Attention Mean vs Entropy
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(df['attention_mean_avg'], df['attention_entropy_avg'], 
                         c=df['cluster'], cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter)
    plt.xlabel('Average Attention Mean')
    plt.ylabel('Average Attention Entropy')
    plt.title('Attention Mean vs Entropy')
    
    # Concentration vs Variability
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(df['attention_concentration_avg'], df['attention_max_std'], 
                         c=df['cluster'], cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter)
    plt.xlabel('Average Attention Concentration')
    plt.ylabel('Max Attention Std Dev')
    plt.title('Concentration vs Variability')
    
    # Cluster distribution by categories
    plt.subplot(2, 3, 4)
    duration_cluster = pd.crosstab(df['duration_category'], df['cluster'])
    duration_cluster.plot(kind='bar', ax=plt.gca())
    plt.title('Cluster Distribution by Duration')
    plt.xlabel('Duration Category')
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    
    plt.subplot(2, 3, 5)
    attention_cluster = pd.crosstab(df['attention_category'], df['cluster'])
    attention_cluster.plot(kind='bar', ax=plt.gca())
    plt.title('Cluster Distribution by Attention Level')
    plt.xlabel('Attention Category')
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    
    # Episode length distribution
    plt.subplot(2, 3, 6)
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]['episode_length']
        plt.hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=15)
    plt.xlabel('Episode Length')
    plt.ylabel('Frequency')
    plt.title('Episode Length Distribution by Cluster')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_characteristics_{timestamp}.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 4. Summary statistics table
    cluster_summary = df.groupby('cluster')[features].mean()
    plt.figure(figsize=(15, 8))
    sns.heatmap(cluster_summary.T, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f')
    plt.title('Cluster Characteristics (Mean Values)')
    plt.xlabel('Cluster')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_summary_{timestamp}.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return clustering_plot


def main():
    parser = argparse.ArgumentParser(description="Compare and cluster episodes based on attention patterns")
    parser.add_argument("--dataset-repo-id", type=str, required=True,
                        help="Repository ID of the dataset to analyze")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the policy checkpoint")
    parser.add_argument("--output-dir", type=str, default="./clustering_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--max-episodes", type=int, default=20,
                        help="Maximum number of episodes to analyze")
    parser.add_argument("--max-frames-per-episode", type=int, default=30,
                        help="Maximum frames to analyze per episode")
    parser.add_argument("--n-clusters", type=int, default=5,
                        help="Number of clusters for K-means")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device to use for inference")
    parser.add_argument("--policy-overrides", type=str, nargs="*",
                        help="Policy config overrides in key=value format")
    parser.add_argument("--show-plots", action="store_true",
                        help="Show interactive plots (otherwise just save to files)")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(args.device)
    model_dtype = torch.float32
    
    print(f"Loading dataset: {args.dataset_repo_id}")
    
    # Load dataset
    try:
        dataset = LeRobotDataset(args.dataset_repo_id)
        print(f"Dataset loaded successfully. Total episodes: {dataset.num_episodes}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load policy
    try:
        print("Loading policy...")
        policy, policy_cfg = load_policy(args.policy_path, dataset.meta, args.policy_overrides)
        
        if hasattr(policy, 'model'):
            policy.model.eval()
            policy.model.to(device)
        elif hasattr(policy, 'eval'):
            policy.eval()
            
        print("Policy loaded successfully")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return
    
    # Determine episodes to analyze
    num_episodes = min(args.max_episodes, dataset.num_episodes)
    episodes_to_analyze = list(range(num_episodes))
    print(f"Will analyze {num_episodes} episodes")
    
    # Extract features from all episodes
    print("\nExtracting features from episodes...")
    episode_features = []
    
    for episode_id in tqdm(episodes_to_analyze, desc="Processing episodes"):
        try:
            features = extract_episode_features(
                dataset=dataset,
                policy=policy,
                episode_id=episode_id,
                device=device,
                model_dtype=model_dtype,
                max_frames=args.max_frames_per_episode
            )
            episode_features.append(features)
        except Exception as e:
            print(f"Error analyzing episode {episode_id}: {e}")
            episode_features.append({"episode_id": episode_id, "valid": False})
    
    # Perform clustering analysis
    print("\nPerforming clustering analysis...")
    clustering_results = perform_clustering_analysis(episode_features, args.n_clusters)
    
    if "error" in clustering_results:
        print(f"Clustering failed: {clustering_results['error']}")
        return
    
    # Create visualizations
    print("\nCreating visualizations...")
    main_plot = create_visualizations(clustering_results, args.output_dir, args.show_plots)
    
    # Save detailed results
    df = clustering_results["dataframe"]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = f"{args.output_dir}/episode_analysis_{timestamp}.csv"
    df.to_csv(results_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CLUSTERING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes analyzed: {len([f for f in episode_features if f.get('valid', False)])}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Main visualization: {main_plot}")
    print(f"Detailed data: {results_file}")
    
    # Show cluster summary
    print(f"\nCluster Summary:")
    for cluster_id in df['cluster'].unique():
        cluster_episodes = df[df['cluster'] == cluster_id]['episode_id'].tolist()
        avg_length = df[df['cluster'] == cluster_id]['episode_length'].mean()
        avg_attention = df[df['cluster'] == cluster_id]['attention_max_peak'].mean()
        print(f"  Cluster {cluster_id}: {len(cluster_episodes)} episodes")
        print(f"    Episodes: {cluster_episodes}")
        print(f"    Avg length: {avg_length:.0f} frames")
        print(f"    Avg max attention: {avg_attention:.3f}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()