#!/usr/bin/env python

"""
Script to analyze and recommend the best episodes based on clustering results.
"""

import numpy as np
import matplotlib.pyplot as plt

# Based on the clustering visualization we saw, let's define what we know about each cluster
cluster_analysis = {
    "Cluster 0 (Bleu foncé)": {
        "episodes": [0, 8, 11, 12],
        "characteristics": "Isolé en haut/droite - patterns uniques",
        "position": "Distinct du groupe principal",
        "likely_quality": "Potentiellement excellents ou très atypiques"
    },
    "Cluster 1 (Vert)": {
        "episodes": [1, 3, 4, 6, 7, 13, 14], 
        "characteristics": "Groupe principal au centre",
        "position": "Comportement standard/moyen",
        "likely_quality": "Qualité moyenne à bonne, représentatifs"
    },
    "Cluster 2 (Jaune)": {
        "episodes": [2, 10],
        "characteristics": "Groupe en bas/gauche - potentiellement courts et focalisés",
        "position": "Séparés du groupe principal",
        "likely_quality": "Courts mais efficaces OU problématiques"
    },
    "Cluster 3 (Violet)": {
        "episodes": [5, 9],
        "characteristics": "Groupe en bas centre - atypiques",
        "position": "Isolés du groupe principal",
        "likely_quality": "Comportement inhabituel, à analyser"
    }
}

# Criteria for "best" episodes
quality_criteria = {
    "efficiency": "Épisodes courts avec forte attention (ratio performance/durée)",
    "focus": "Épisodes avec attention très concentrée (faible entropie)",
    "consistency": "Épisodes avec patterns d'attention stables", 
    "uniqueness": "Épisodes avec patterns d'attention uniques mais réussis",
    "representativeness": "Épisodes typiques pour l'entraînement"
}

def analyze_episode_quality():
    """Analyze episode quality based on the latest clustering results."""
    
    # Try to read the latest clustering results
    import glob
    import pandas as pd
    
    # Find the most recent clustering analysis file
    csv_files = glob.glob('/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/episode_analysis_*.csv')
    
    if csv_files:
        # Get the most recent file
        latest_file = max(csv_files, key=lambda x: x.split('_')[-1])
        try:
            df = pd.read_csv(latest_file)
            print(f"📊 Utilisation des données de clustering récentes: {len(df)} épisodes")
            return analyze_from_real_data(df)
        except Exception as e:
            print(f"⚠️ Erreur lecture des données récentes: {e}")
            print("📊 Utilisation des recommandations par défaut")
    else:
        print("📊 Aucun fichier de clustering trouvé, utilisation des recommandations par défaut")
    
    # Fallback aux recommandations par défaut
    recommendations = {
        "🏆 EXCELLENTS (à prioriser)": [0, 12],
        "👍 BONS (recommandés)": [1, 6, 7],
        "🤔 INTÉRESSANTS (à analyser)": [8, 11, 2, 10],
        "⚠️ ATTENTION (potentiels problèmes)": [5, 9],
        "📊 REPRÉSENTATIFS (pour statistiques)": [3, 4, 13, 14]
    }
    
    return recommendations

def analyze_from_real_data(df):
    """Analyze episode quality from real clustering data."""
    recommendations = {
        "🏆 EXCELLENTS (à prioriser)": [],
        "👍 BONS (recommandés)": [],
        "🤔 INTÉRESSANTS (à analyser)": [],
        "⚠️ ATTENTION (potentiels problèmes)": [],
        "📊 REPRÉSENTATIFS (pour statistiques)": []
    }
    
    if 'cluster' not in df.columns or 'episode_id' not in df.columns:
        print("⚠️ Colonnes manquantes dans les données de clustering")
        return recommendations
    
    print(f"📊 Données réelles trouvées: {len(df)} épisodes")
    
    # Calculer des scores pour chaque épisode basés sur les vraies métriques
    if 'attention_max_peak' in df.columns:
        # Score basé sur l'attention peak (plus haut = mieux)
        df['attention_score'] = (df['attention_max_peak'] - df['attention_max_peak'].min()) / (df['attention_max_peak'].max() - df['attention_max_peak'].min() + 1e-8)
    else:
        df['attention_score'] = 0.5
    
    if 'episode_length' in df.columns:
        # Score d'efficacité (plus court = mieux pour même qualité)
        df['efficiency_score'] = (df['episode_length'].max() - df['episode_length']) / (df['episode_length'].max() - df['episode_length'].min() + 1e-8)
    else:
        df['efficiency_score'] = 0.5
    
    # Score global composite
    df['overall_score'] = (df['attention_score'] + df['efficiency_score']) / 2
    
    # Trier par score global
    df_sorted = df.sort_values('overall_score', ascending=False)
    
    # Distribuer les épisodes selon leurs scores
    num_episodes = len(df_sorted)
    
    # Top 20% = Excellents
    excellent_count = max(1, num_episodes // 5)
    excellent_episodes = df_sorted.head(excellent_count)['episode_id'].tolist()
    recommendations["🏆 EXCELLENTS (à prioriser)"] = [int(ep) for ep in excellent_episodes]
    
    # Next 30% = Bons
    good_count = max(1, int(num_episodes * 0.3))
    good_episodes = df_sorted.iloc[excellent_count:excellent_count + good_count]['episode_id'].tolist()
    recommendations["👍 BONS (recommandés)"] = [int(ep) for ep in good_episodes]
    
    # Middle 30% = Intéressants
    interesting_count = max(1, int(num_episodes * 0.3))
    interesting_episodes = df_sorted.iloc[excellent_count + good_count:excellent_count + good_count + interesting_count]['episode_id'].tolist()
    recommendations["🤔 INTÉRESSANTS (à analyser)"] = [int(ep) for ep in interesting_episodes]
    
    # Bottom 20% = Attention (problèmes potentiels)
    attention_episodes = df_sorted.tail(num_episodes - excellent_count - good_count - interesting_count)['episode_id'].tolist()
    recommendations["⚠️ ATTENTION (potentiels problèmes)"] = [int(ep) for ep in attention_episodes]
    
    # Analyser aussi par clusters pour les représentatifs
    cluster_sizes = df['cluster'].value_counts().sort_index()
    for cluster_id in df['cluster'].unique():
        cluster_episodes = df[df['cluster'] == cluster_id]['episode_id'].tolist()
        if len(cluster_episodes) >= 5:  # Clusters suffisamment grands
            # Prendre quelques épisodes représentatifs du cluster (médians en score)
            cluster_df = df[df['cluster'] == cluster_id].sort_values('overall_score')
            mid_episodes = cluster_df.iloc[len(cluster_df)//3:2*len(cluster_df)//3]['episode_id'].tolist()
            recommendations["📊 REPRÉSENTATIFS (pour statistiques)"].extend([int(ep) for ep in mid_episodes[:2]])
    
    return recommendations

def print_recommendations():
    """Print episode recommendations with explanations."""
    
    recommendations = analyze_episode_quality()
    
    print("=" * 80)
    print("🎯 ANALYSE DES MEILLEURS ÉPISODES")
    print("=" * 80)
    
    print("\n📋 CRITÈRES D'ÉVALUATION:")
    for criterion, description in quality_criteria.items():
        print(f"  • {criterion.upper()}: {description}")
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMANDATIONS PAR CATÉGORIE")
    print("=" * 60)
    
    for category, episodes in recommendations.items():
        if episodes:
            print(f"\n{category}:")
            for ep in episodes:
                print(f"  → Épisode {ep}")
    
    print("\n" + "=" * 60)
    print("📊 ANALYSE DÉTAILLÉE PAR CLUSTER")  
    print("=" * 60)
    
    for cluster_name, info in cluster_analysis.items():
        print(f"\n{cluster_name}:")
        print(f"  Épisodes: {info['episodes']}")
        print(f"  Position: {info['position']}")
        print(f"  Qualité probable: {info['likely_quality']}")
    
    print("\n" + "=" * 60)
    print("🎯 PLAN D'ACTION RECOMMANDÉ")
    print("=" * 60)
    
    # Generate dynamic action plan based on actual recommendations
    action_plan = []
    step_num = 1
    
    if recommendations["🏆 EXCELLENTS (à prioriser)"]:
        episodes_str = ", ".join([str(ep) for ep in recommendations["🏆 EXCELLENTS (à prioriser)"][:5]])  # Limit display
        action_plan.append(f"{step_num}. TESTER EN PRIORITÉ: Épisodes {episodes_str} (excellents)")
        step_num += 1
    
    if recommendations["👍 BONS (recommandés)"]:
        episodes_str = ", ".join([str(ep) for ep in recommendations["👍 BONS (recommandés)"][:5]])
        action_plan.append(f"{step_num}. VALIDER: Épisodes {episodes_str} (bons candidats)")
        step_num += 1
    
    if recommendations["🤔 INTÉRESSANTS (à analyser)"]:
        episodes_str = ", ".join([str(ep) for ep in recommendations["🤔 INTÉRESSANTS (à analyser)"][:5]])
        action_plan.append(f"{step_num}. ANALYSER: Épisodes {episodes_str} (potentiel intéressant)")
        step_num += 1
    
    if recommendations["📊 REPRÉSENTATIFS (pour statistiques)"]:
        episodes_str = ", ".join([str(ep) for ep in recommendations["📊 REPRÉSENTATIFS (pour statistiques)"][:5]])
        action_plan.append(f"{step_num}. RÉFÉRENCE: Épisodes {episodes_str} (comportement standard)")
        step_num += 1
    
    if recommendations["⚠️ ATTENTION (potentiels problèmes)"]:
        episodes_str = ", ".join([str(ep) for ep in recommendations["⚠️ ATTENTION (potentiels problèmes)"][:5]])
        action_plan.append(f"{step_num}. EXAMINER: Épisodes {episodes_str} (problèmes potentiels)")
    
    for step in action_plan:
        print(f"  {step}")
    
    print("\n" + "=" * 60)
    print("💡 CONSEILS POUR L'ANALYSE")
    print("=" * 60)
    
    tips = [
        "• Regarder les vidéos d'attention des épisodes recommandés",
        "• Comparer la durée vs performance de chaque épisode",
        "• Analyser les patterns d'attention (focalisé vs diffus)",
        "• Vérifier la stabilité des actions dans le temps",
        "• Identifier les moments clés de chaque tâche"
    ]
    
    for tip in tips:
        print(f"  {tip}")
    
    print("\n" + "🎯" * 20)

def create_priority_visualization():
    """Create a simple visualization of episode priorities."""
    
    # Configure matplotlib for non-interactive use
    import matplotlib
    matplotlib.use('Agg')
    
    recommendations = analyze_episode_quality()
    
    # Create priority scores
    episode_scores = {}
    
    # Assign scores based on recommendations
    for ep in recommendations["🏆 EXCELLENTS (à prioriser)"]:
        episode_scores[ep] = 5
    for ep in recommendations["👍 BONS (recommandés)"]:
        episode_scores[ep] = 4  
    for ep in recommendations["🤔 INTÉRESSANTS (à analyser)"]:
        episode_scores[ep] = 3
    for ep in recommendations["⚠️ ATTENTION (potentiels problèmes)"]:
        episode_scores[ep] = 2
    for ep in recommendations["📊 REPRÉSENTATIFS (pour statistiques)"]:
        episode_scores[ep] = 3
    
    episodes = sorted(episode_scores.keys())
    scores = [episode_scores[ep] for ep in episodes]
    colors = ['red' if s == 5 else 'orange' if s == 4 else 'yellow' if s == 3 else 'lightcoral' for s in scores]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(episodes, scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(score), ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Numéro d\'épisode')
    plt.ylabel('Score de priorité (1-5)')
    plt.title('🎯 Priorité d\'analyse par épisode\n(5=Excellent, 4=Bon, 3=Intéressant, 2=Attention)')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(episodes)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Excellents (5)'),
        Patch(facecolor='orange', alpha=0.7, label='Bons (4)'), 
        Patch(facecolor='yellow', alpha=0.7, label='Intéressants (3)'),
        Patch(facecolor='lightcoral', alpha=0.7, label='Attention (2)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/Users/askel/Documents/GitHub/pro2robot/resources/physical-AI-interpretability/clustering_analysis/episode_priorities.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid blocking
    
    return episode_scores

if __name__ == "__main__":
    print_recommendations()
    print("\n🎨 Génération de la visualisation des priorités...")
    scores = create_priority_visualization()
    print("✅ Visualisation sauvée: clustering_analysis/episode_priorities.png")