"""
Fragment Clustering Visualization Module.

Generates comprehensive visualizations from clustering results:
  - Fragment similarity heatmap (cosine similarity)
  - Silhouette plots (overall + per-cluster)
  - t-SNE and UMAP dimensionality reduction with cluster coloring
  - Cluster size distribution
  - File type vs. cluster confusion matrix
  - Clustering metrics summary
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Try to import UMAP; fall back gracefully if not available
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings('ignore', category=UserWarning)


def load_clustering_results(cluster_file):
    """Load clustering results from JSON file."""
    with open(cluster_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_features_batch(file_paths):
    """Extract features for visualization (reusing logic from clustering_comparison.py)."""
    from sequence.main import extract_features
    features = []
    for file_path in file_paths:
        features.append(extract_features(file_path))
    return np.vstack(features)


def create_similarity_heatmap(features, labels, file_paths, output_dir, sample_size=50):
    """
    Create a heatmap showing cosine similarity between fragments.
    If too many fragments, sample randomly.
    """
    n_samples = len(features)
    
    # Sample if too many
    if n_samples > sample_size:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        indices = sorted(indices)
        features_sample = features[indices]
        labels_sample = labels[indices]
        file_paths_sample = [file_paths[i] for i in indices]
    else:
        features_sample = features
        labels_sample = labels
        file_paths_sample = file_paths
        indices = np.arange(n_samples)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(features_sample)
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Create color mapping for clusters
    unique_labels = sorted(set(labels_sample))
    cmap = plt.colormaps['tab20']
    label_colors = {lbl: cmap(i / max(len(unique_labels) - 1, 1)) for i, lbl in enumerate(unique_labels)}
    
    # Create labels for heatmap
    short_labels = [os.path.basename(fp).split('_')[0:2] for fp in file_paths_sample]
    short_labels = ['_'.join(lbl[:2]) if len(lbl) >= 2 else lbl[0] for lbl in short_labels]
    
    # Plot heatmap
    ax = sns.heatmap(
        similarity_matrix,
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'},
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=None
    )
    
    # Add cluster color bar on the side
    ax_cbar2 = plt.axes([0.92, 0.15, 0.02, 0.7])
    for i, lbl in enumerate(unique_labels):
        ax_cbar2.barh(i, 1, color=label_colors[lbl], height=0.9)
    ax_cbar2.set_yticks(range(len(unique_labels)))
    ax_cbar2.set_yticklabels([f'Cluster {lbl}' for lbl in unique_labels])
    ax_cbar2.set_xlim(0, 1)
    ax_cbar2.set_xticks([])
    
    plt.title(f'Fragment Cosine Similarity Matrix\n({len(file_paths_sample)} samples)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    output_file = os.path.join(output_dir, '01_similarity_heatmap.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Similarity heatmap saved: {output_file}")
    
    return output_file


def create_silhouette_plot(features, labels, output_dir):
    """Create overall silhouette plot."""
    # Compute silhouette scores
    silhouette_avg = silhouette_score(features, labels, metric='cosine')
    sample_silhouette_values = silhouette_samples(features, labels, metric='cosine')
    
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    
    plt.figure(figsize=(12, 8))
    
    y_lower = 10
    cmap = plt.colormaps['tab20']
    
    for i, label in enumerate(unique_labels):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == label]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cmap(i / max(len(unique_labels) - 1, 1))
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, ith_cluster_silhouette_values,
            facecolor=color, edgecolor=color, alpha=0.7
        )
        
        # Label the silhouette plots with cluster numbers
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10
    
    plt.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, label=f'Avg: {silhouette_avg:.3f}')
    plt.xlabel("Silhouette Coefficient", fontsize=12)
    plt.ylabel("Cluster Label", fontsize=12)
    plt.title(f'Silhouette Plot for Fragment Clusters\n(Avg Score: {silhouette_avg:.3f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, '02_silhouette_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Silhouette plot saved: {output_file}")
    
    return silhouette_avg, output_file


def create_tsne_plot(features, labels, file_paths, output_dir, perplexity=30):
    """Create t-SNE dimensionality reduction visualization."""
    print("  Computing t-SNE (this may take a moment)...")
    
    # t-SNE with cosine metric
    n_samples = len(features)
    perp = min(perplexity, (n_samples - 1) / 3)
    
    tsne = TSNE(
        n_components=2,
        metric='cosine',
        random_state=42,
        perplexity=perp,
        max_iter=1000,
        verbose=0
    )
    embedding = tsne.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps['tab20']
    
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[cmap(unique_labels.index(label) / max(len(unique_labels) - 1, 1))],
            label=f'Cluster {label}',
            s=100, alpha=0.7, edgecolors='black', linewidth=0.5
        )
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Visualization of Clustered Fragments\n(Cosine Metric)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, '03_tsne_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ t-SNE plot saved: {output_file}")
    
    return output_file


def create_umap_plot(features, labels, file_paths, output_dir):
    """Create UMAP dimensionality reduction visualization (if available)."""
    if not HAS_UMAP:
        print("⚠ UMAP not installed. Skipping UMAP plot. Install with: pip install umap-learn")
        return None
    
    print("  Computing UMAP (this may take a moment)...")
    
    reducer = umap.UMAP(
        n_components=2,
        metric='cosine',
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    embedding = reducer.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps['tab20']
    
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[cmap(unique_labels.index(label) / max(len(unique_labels) - 1, 1))],
            label=f'Cluster {label}',
            s=100, alpha=0.7, edgecolors='black', linewidth=0.5
        )
    
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('UMAP Visualization of Clustered Fragments\n(Cosine Metric)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, '04_umap_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ UMAP plot saved: {output_file}")
    
    return output_file


def create_cluster_size_distribution(labels, output_dir):
    """Create bar chart of cluster sizes."""
    unique_labels = sorted(set(labels))
    cluster_sizes = [np.sum(labels == lbl) for lbl in unique_labels]
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    bars = plt.bar(
        [f'Cluster {lbl}' for lbl in unique_labels],
        cluster_sizes,
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Cluster', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Fragments', fontsize=12, fontweight='bold')
    plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, '05_cluster_distribution.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Cluster distribution chart saved: {output_file}")
    
    return output_file


def create_filetype_vs_cluster_matrix(labels, file_paths, output_dir):
    """
    Create a confusion-like matrix showing file type distribution across clusters.
    """
    from sequence.main import infer_file_type_from_name
    
    # Build dataframe
    data = {
        'Cluster': [f'Cluster {lbl}' if lbl != -1 else 'Noise' for lbl in labels],
        'FileType': [infer_file_type_from_name(os.path.basename(fp)) or 'UNKNOWN' for fp in file_paths]
    }
    df = pd.DataFrame(data)
    
    # Create crosstab
    crosstab = pd.crosstab(df['FileType'], df['Cluster'])
    
    # Plot heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        crosstab,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Fragment Count'},
        linewidths=0.5
    )
    plt.title('File Type vs. Cluster Distribution\n(Fragment Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster', fontsize=12, fontweight='bold')
    plt.ylabel('File Type', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, '06_filetype_cluster_matrix.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ File type vs. cluster matrix saved: {output_file}")
    
    return output_file


def generate_metrics_summary(features, labels, file_paths, output_dir):
    """Generate and save a detailed metrics summary."""
    from sequence.main import infer_file_type_from_name
    
    unique_labels = sorted(set(labels))
    
    summary = {
        'Total Fragments': int(len(labels)),
        'Total Clusters': int(len([l for l in unique_labels if l != -1])),
        'Noise/Unclustered': int(np.sum(labels == -1)),
        'Silhouette Score': float(silhouette_score(features, labels, metric='cosine')),
    }
    
    # Per-cluster statistics
    cluster_stats = {}
    for label in unique_labels:
        mask = labels == label
        count = int(np.sum(mask))  # Convert numpy int to Python int
        cluster_files = [os.path.basename(fp) for fp in np.array(file_paths)[mask]]
        
        # File type distribution
        file_types = [infer_file_type_from_name(f) or 'UNKNOWN' for f in cluster_files]
        type_counts = {}
        for ft in file_types:
            type_counts[ft] = type_counts.get(ft, 0) + 1
        
        # Convert all counts to Python ints for JSON serialization
        type_counts = {k: int(v) for k, v in type_counts.items()}
        
        cluster_name = f'Cluster {label}' if label != -1 else 'Noise'
        cluster_stats[cluster_name] = {
            'Fragment Count': count,
            'File Type Distribution': type_counts,
            'Samples': cluster_files[:5],  # First 5 samples
        }
    
    summary['Cluster Statistics'] = cluster_stats
    
    # Write summary to JSON
    summary_file = os.path.join(output_dir, 'clustering_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Also write as readable text
    text_file = os.path.join(output_dir, 'clustering_summary.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("CLUSTERING QUALITY METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Fragments: {summary['Total Fragments']}\n")
        f.write(f"Total Clusters: {summary['Total Clusters']}\n")
        f.write(f"Noise/Unclustered: {summary['Noise/Unclustered']}\n")
        f.write(f"Silhouette Score (Cosine): {summary['Silhouette Score']:.4f}\n")
        f.write("  (Range: -1 to 1; closer to 1 is better)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("PER-CLUSTER STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        for cluster_name, stats in sorted(cluster_stats.items()):
            f.write(f"{cluster_name}:\n")
            f.write(f"  Fragment Count: {stats['Fragment Count']}\n")
            f.write(f"  File Type Distribution:\n")
            for ft, count in sorted(stats['File Type Distribution'].items(), key=lambda x: -x[1]):
                f.write(f"    - {ft}: {count}\n")
            f.write(f"  Sample Fragments:\n")
            for sample in stats['Samples']:
                f.write(f"    - {sample}\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"✓ Clustering metrics summary saved: {summary_file}")
    print(f"✓ Readable summary saved: {text_file}")
    
    return summary_file, text_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive visualizations for clustering results.'
    )
    parser.add_argument(
        '--clusters-json',
        type=str,
        default=None,
        help='Path to clusters.json from clustering_comparison.py. '
             'Defaults to results/clusters.json.',
    )
    parser.add_argument(
        '--fragments-dir',
        type=str,
        default=None,
        help='Path to the fragments directory (same as in clustering_comparison.py). '
             'Defaults to Train/.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations. Defaults to results/visualizations/.',
    )
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    clusters_json = args.clusters_json or os.path.join(project_root, 'results', 'clusters.json')
    fragments_dir = args.fragments_dir or os.path.join(project_root, 'Train')
    output_dir = args.output_dir or os.path.join(project_root, 'results', 'visualizations')
    
    print('=' * 70)
    print('CLUSTERING VISUALIZATION GENERATOR')
    print('=' * 70)
    print(f'\nClusters file: {clusters_json}')
    print(f'Fragments dir: {fragments_dir}')
    print(f'Output dir:    {output_dir}')
    
    # Validate inputs
    if not os.path.exists(clusters_json):
        print(f"\nError: clusters.json not found at '{clusters_json}'")
        print("Please run clustering_comparison.py first.")
        return
    
    if not os.path.exists(fragments_dir):
        print(f"\nError: Fragments directory not found at '{fragments_dir}'")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load clustering results
    print('\nLoading clustering results...')
    cluster_mapping = load_clustering_results(clusters_json)
    
    # Build file paths and labels
    file_paths = []
    labels = []
    
    for rel_path, cluster_label in cluster_mapping.items():
        full_path = os.path.join(fragments_dir, rel_path)
        file_paths.append(full_path)
        
        # Extract numeric label from cluster_label
        if cluster_label.startswith('Cluster'):
            label = int(cluster_label.split()[-1])
        else:
            label = -1  # Noise
        labels.append(label)
    
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    
    print(f'Loaded {len(file_paths)} fragments')
    print(f'Found {len(set(labels))} unique labels')
    
    # Extract features
    print('\nExtracting features from fragments...')
    features = extract_features_batch(file_paths)
    print(f'Features shape: {features.shape}')
    
    # Generate visualizations
    print('\n' + '=' * 70)
    print('GENERATING VISUALIZATIONS')
    print('=' * 70 + '\n')
    
    # 1. Similarity heatmap
    print('Creating similarity heatmap...')
    create_similarity_heatmap(features, labels, file_paths, output_dir, sample_size=50)
    
    # 2. Silhouette plot
    print('Creating silhouette plot...')
    sil_score, _ = create_silhouette_plot(features, labels, output_dir)
    
    # 3. t-SNE plot
    print('Creating t-SNE plot...')
    create_tsne_plot(features, labels, file_paths, output_dir)
    
    # 4. UMAP plot (if available)
    print('Creating UMAP plot...')
    create_umap_plot(features, labels, file_paths, output_dir)
    
    # 5. Cluster size distribution
    print('Creating cluster size distribution...')
    create_cluster_size_distribution(labels, output_dir)
    
    # 6. File type vs cluster matrix
    print('Creating file type vs. cluster matrix...')
    create_filetype_vs_cluster_matrix(labels, file_paths, output_dir)
    
    # 7. Metrics summary
    print('Generating metrics summary...')
    generate_metrics_summary(features, labels, file_paths, output_dir)
    
    print('\n' + '=' * 70)
    print('VISUALIZATION COMPLETE')
    print('=' * 70)
    print(f'\nAll visualizations saved to: {output_dir}')
    print('\nGenerated files:')
    print('  01_similarity_heatmap.png - Cosine similarity between fragments')
    print('  02_silhouette_plot.png - Cluster quality assessment')
    print('  03_tsne_plot.png - 2D fragment distribution (t-SNE)')
    if HAS_UMAP:
        print('  04_umap_plot.png - 2D fragment distribution (UMAP)')
    print('  05_cluster_distribution.png - Fragments per cluster')
    print('  06_filetype_cluster_matrix.png - File type distribution across clusters')
    print('  clustering_summary.json - Detailed metrics and statistics')
    print('  clustering_summary.txt - Human-readable summary')
    print()


if __name__ == '__main__':
    main()
