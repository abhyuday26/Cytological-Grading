"""
AI-Based Cytological Grading Algorithm
=======================================
Unsupervised clustering approach to classify cells into 3-tier cytological grades:
  - Grade 1: Well-differentiated (small, regular nuclei)
  - Grade 2: Moderately-differentiated (intermediate features)
  - Grade 3: Poorly-differentiated (large, irregular nuclei)

Uses 75% training / 25% validation split.
Compares K-Means, Hierarchical Clustering, and Gaussian Mixture Models.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "data", "Results_ImageJ.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
N_CLUSTERS = 3  # 3-tier grading
TRAIN_RATIO = 0.75
RANDOM_STATE = 42

# Morphometric features relevant for grading
MORPHOMETRIC_FEATURES = [
    'Area', 'Mean', 'Perim.', 'Circ.', 'Feret',
    'MinFeret', 'AR', 'Round', 'Solidity'
]

# Create output directories
os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot styling
plt.rcParams.update({
    'figure.figsize': (10, 7),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150
})
sns.set_style("whitegrid")

GRADE_COLORS = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}
GRADE_LABELS = {
    1: 'Grade 1\n(Well-differentiated)',
    2: 'Grade 2\n(Moderately-differentiated)',
    3: 'Grade 3\n(Poorly-differentiated)'
}


# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================
def load_and_clean_data(filepath):
    """Load CSV and clean anomalous rows."""
    print("=" * 60)
    print("STEP 1: DATA LOADING & CLEANING")
    print("=" * 60)

    df = pd.read_csv(filepath)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Cases: {sorted(df['Case no.'].unique())}")
    print(f"  Total unique cases: {df['Case no.'].nunique()}")

    # Remove anomalous rows (e.g., near-zero Area)
    anomalous = df[df['Area'] < 1.0]
    if len(anomalous) > 0:
        print(f"  [WARNING] Removing {len(anomalous)} anomalous rows (Area < 1.0)")
        df = df[df['Area'] >= 1.0].reset_index(drop=True)

    # Check for missing values in morphometric features
    missing = df[MORPHOMETRIC_FEATURES].isnull().sum()
    if missing.sum() > 0:
        print(f"  [WARNING] Missing values found:\n{missing[missing > 0]}")
        df = df.dropna(subset=MORPHOMETRIC_FEATURES).reset_index(drop=True)

    print(f"  Clean dataset: {len(df)} cells across {df['Case no.'].nunique()} cases")
    print()
    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Compute derived nuclear morphometric features."""
    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    # Derived features based on cytopathology literature
    df['Nuclear_Size_Index'] = df['Area'] * df['Feret']
    df['Irregularity_Index'] = 1 - df['Circ.']
    df['Pleomorphism_Index'] = df['AR'] * (1 - df['Round'])
    df['Feret_Ratio'] = df['MinFeret'] / df['Feret']
    df['NC_Compactness'] = (df['Perim.'] ** 2) / (4 * np.pi * df['Area'])

    derived_features = [
        'Nuclear_Size_Index', 'Irregularity_Index',
        'Pleomorphism_Index', 'Feret_Ratio', 'NC_Compactness'
    ]

    all_features = MORPHOMETRIC_FEATURES + derived_features
    print(f"  Original features: {len(MORPHOMETRIC_FEATURES)}")
    print(f"  Derived features:  {len(derived_features)}")
    print(f"  Total features:    {len(all_features)}")
    print(f"  Features: {all_features}")
    print()

    return df, all_features


# ============================================================
# 3. TRAIN-VALIDATION SPLIT (75/25)
# ============================================================
def split_data(df, all_features):
    """Split data 75% training, 25% validation."""
    print("=" * 60)
    print("STEP 3: TRAIN-VALIDATION SPLIT (75/25)")
    print("=" * 60)

    X = df[all_features].values

    # Split preserving case distribution
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=1 - TRAIN_RATIO,
        random_state=RANDOM_STATE,
        stratify=df['Case no.']
    )

    X_train_raw = X[train_idx]
    X_val_raw = X[val_idx]

    print(f"  Training set:   {len(train_idx)} cells ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_idx)} cells ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"  Cases in train: {df.iloc[train_idx]['Case no.'].nunique()}")
    print(f"  Cases in val:   {df.iloc[val_idx]['Case no.'].nunique()}")

    # Standardize (fit on train, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print(f"  [OK] StandardScaler fitted on training data")
    print()

    return X_train, X_val, train_idx, val_idx, scaler


# ============================================================
# 4. PCA DIMENSIONALITY REDUCTION
# ============================================================
def apply_pca(X_train, X_val, all_features):
    """Apply PCA retaining 95% variance."""
    print("=" * 60)
    print("STEP 4: PCA DIMENSIONALITY REDUCTION")
    print("=" * 60)

    # Fit PCA on training data
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train)

    # Determine components for 95% variance
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumvar >= 0.95) + 1

    # Apply PCA with selected components
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    print(f"  Original dimensions:   {X_train.shape[1]}")
    print(f"  PCA components (95%):  {n_components}")
    print(f"  Variance explained:    {cumvar[n_components-1]*100:.2f}%")
    print()

    # --- Plot: Scree Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_, color='#3498db', alpha=0.8)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('Scree Plot')
    ax1.axvline(x=n_components, color='red', linestyle='--', label=f'Selected: {n_components} PCs')
    ax1.legend()

    ax2.plot(range(1, len(cumvar) + 1), cumvar * 100, 'bo-', markersize=6)
    ax2.axhline(y=95, color='red', linestyle='--', label='95% threshold')
    ax2.axvline(x=n_components, color='red', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_pca_scree_plot.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 01_pca_scree_plot.png")

    # --- Plot: Feature Loadings ---
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=all_features
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(loadings[['PC1', 'PC2']], annot=True, cmap='RdBu_r',
                center=0, fmt='.2f', ax=ax)
    ax.set_title('PCA Feature Loadings (PC1 & PC2)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_pca_loadings.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 02_pca_loadings.png")
    print()

    return X_train_pca, X_val_pca, pca, n_components


# ============================================================
# 5. OPTIMAL CLUSTER DETERMINATION
# ============================================================
def find_optimal_clusters(X_train_pca):
    """Determine optimal number of clusters using silhouette & elbow method."""
    print("=" * 60)
    print("STEP 5: OPTIMAL CLUSTER DETERMINATION")
    print("=" * 60)

    k_range = range(2, 7)
    silhouette_scores = []
    inertias = []
    ch_scores = []
    db_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_train_pca)

        sil = silhouette_score(X_train_pca, labels)
        ch = calinski_harabasz_score(X_train_pca, labels)
        db = davies_bouldin_score(X_train_pca, labels)

        silhouette_scores.append(sil)
        inertias.append(km.inertia_)
        ch_scores.append(ch)
        db_scores.append(db)

        print(f"  k={k}: Silhouette={sil:.4f}, CH={ch:.1f}, DB={db:.4f}")

    # --- Plot: Cluster Optimization ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(list(k_range), silhouette_scores, 'go-', markersize=8, linewidth=2)
    axes[0, 0].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k=3 (target)')
    axes[0, 0].set_title('Silhouette Score (Higher = Better)')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()

    axes[0, 1].plot(list(k_range), inertias, 'bs-', markersize=8, linewidth=2)
    axes[0, 1].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k=3 (target)')
    axes[0, 1].set_title('Elbow Method (Inertia)')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Inertia')
    axes[0, 1].legend()

    axes[1, 0].plot(list(k_range), ch_scores, 'r^-', markersize=8, linewidth=2)
    axes[1, 0].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k=3 (target)')
    axes[1, 0].set_title('Calinski-Harabasz Index (Higher = Better)')
    axes[1, 0].set_xlabel('Number of Clusters (k)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()

    axes[1, 1].plot(list(k_range), db_scores, 'mD-', markersize=8, linewidth=2)
    axes[1, 1].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='k=3 (target)')
    axes[1, 1].set_title('Davies-Bouldin Index (Lower = Better)')
    axes[1, 1].set_xlabel('Number of Clusters (k)')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()

    plt.suptitle('Cluster Optimization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_cluster_optimization.png'), bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: 03_cluster_optimization.png")
    print(f"  Using k={N_CLUSTERS} (3-tier grading)\n")

    return silhouette_scores, inertias


# ============================================================
# 6. CLUSTERING MODELS (TRAIN + VALIDATE)
# ============================================================
def train_clustering_models(X_train_pca, X_val_pca):
    """Train K-Means, Hierarchical, GMM and evaluate on validation set."""
    print("=" * 60)
    print("STEP 6: CLUSTERING MODELS (TRAIN + VALIDATE)")
    print("=" * 60)

    results = {}

    # --- 6A: K-Means ---
    print("\n  [A] K-Means Clustering")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    train_labels_km = kmeans.fit_predict(X_train_pca)
    val_labels_km = kmeans.predict(X_val_pca)

    results['K-Means'] = {
        'model': kmeans,
        'train_labels': train_labels_km,
        'val_labels': val_labels_km,
        'train_sil': silhouette_score(X_train_pca, train_labels_km),
        'val_sil': silhouette_score(X_val_pca, val_labels_km),
        'train_ch': calinski_harabasz_score(X_train_pca, train_labels_km),
        'val_ch': calinski_harabasz_score(X_val_pca, val_labels_km),
        'train_db': davies_bouldin_score(X_train_pca, train_labels_km),
        'val_db': davies_bouldin_score(X_val_pca, val_labels_km),
    }
    print(f"      Train - Silhouette: {results['K-Means']['train_sil']:.4f}, CH: {results['K-Means']['train_ch']:.1f}, DB: {results['K-Means']['train_db']:.4f}")
    print(f"      Val   - Silhouette: {results['K-Means']['val_sil']:.4f}, CH: {results['K-Means']['val_ch']:.1f}, DB: {results['K-Means']['val_db']:.4f}")

    # --- 6B: Hierarchical Clustering ---
    print("\n  [B] Hierarchical Clustering (Agglomerative)")
    agg = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
    train_labels_hc = agg.fit_predict(X_train_pca)

    # For validation: assign to nearest cluster center (compute centers from train)
    hc_centers = np.array([X_train_pca[train_labels_hc == c].mean(axis=0) for c in range(N_CLUSTERS)])
    val_labels_hc = np.argmin(cdist(X_val_pca, hc_centers, metric='euclidean'), axis=1)

    results['Hierarchical'] = {
        'model': agg,
        'train_labels': train_labels_hc,
        'val_labels': val_labels_hc,
        'centers': hc_centers,
        'train_sil': silhouette_score(X_train_pca, train_labels_hc),
        'val_sil': silhouette_score(X_val_pca, val_labels_hc),
        'train_ch': calinski_harabasz_score(X_train_pca, train_labels_hc),
        'val_ch': calinski_harabasz_score(X_val_pca, val_labels_hc),
        'train_db': davies_bouldin_score(X_train_pca, train_labels_hc),
        'val_db': davies_bouldin_score(X_val_pca, val_labels_hc),
    }
    print(f"      Train - Silhouette: {results['Hierarchical']['train_sil']:.4f}, CH: {results['Hierarchical']['train_ch']:.1f}, DB: {results['Hierarchical']['train_db']:.4f}")
    print(f"      Val   - Silhouette: {results['Hierarchical']['val_sil']:.4f}, CH: {results['Hierarchical']['val_ch']:.1f}, DB: {results['Hierarchical']['val_db']:.4f}")

    # --- 6C: Gaussian Mixture Model ---
    print("\n  [C] Gaussian Mixture Model (GMM)")
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE,
                          covariance_type='full', n_init=5, max_iter=300)
    train_labels_gmm = gmm.fit_predict(X_train_pca)
    val_labels_gmm = gmm.predict(X_val_pca)

    results['GMM'] = {
        'model': gmm,
        'train_labels': train_labels_gmm,
        'val_labels': val_labels_gmm,
        'train_sil': silhouette_score(X_train_pca, train_labels_gmm),
        'val_sil': silhouette_score(X_val_pca, val_labels_gmm),
        'train_ch': calinski_harabasz_score(X_train_pca, train_labels_gmm),
        'val_ch': calinski_harabasz_score(X_val_pca, val_labels_gmm),
        'train_db': davies_bouldin_score(X_train_pca, train_labels_gmm),
        'val_db': davies_bouldin_score(X_val_pca, val_labels_gmm),
    }
    print(f"      Train - Silhouette: {results['GMM']['train_sil']:.4f}, CH: {results['GMM']['train_ch']:.1f}, DB: {results['GMM']['train_db']:.4f}")
    print(f"      Val   - Silhouette: {results['GMM']['val_sil']:.4f}, CH: {results['GMM']['val_ch']:.1f}, DB: {results['GMM']['val_db']:.4f}")

    # --- Model Comparison Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    # Silhouette
    train_sils = [results[m]['train_sil'] for m in model_names]
    val_sils = [results[m]['val_sil'] for m in model_names]
    axes[0].bar(x - width/2, train_sils, width, label='Train', color='#3498db', alpha=0.8)
    axes[0].bar(x + width/2, val_sils, width, label='Validation', color='#e74c3c', alpha=0.8)
    axes[0].set_title('Silhouette Score\n(Higher = Better)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].legend()

    # Calinski-Harabasz
    train_chs = [results[m]['train_ch'] for m in model_names]
    val_chs = [results[m]['val_ch'] for m in model_names]
    axes[1].bar(x - width/2, train_chs, width, label='Train', color='#3498db', alpha=0.8)
    axes[1].bar(x + width/2, val_chs, width, label='Validation', color='#e74c3c', alpha=0.8)
    axes[1].set_title('Calinski-Harabasz Index\n(Higher = Better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].legend()

    # Davies-Bouldin
    train_dbs = [results[m]['train_db'] for m in model_names]
    val_dbs = [results[m]['val_db'] for m in model_names]
    axes[2].bar(x - width/2, train_dbs, width, label='Train', color='#3498db', alpha=0.8)
    axes[2].bar(x + width/2, val_dbs, width, label='Validation', color='#e74c3c', alpha=0.8)
    axes[2].set_title('Davies-Bouldin Index\n(Lower = Better)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names)
    axes[2].legend()

    plt.suptitle('Model Comparison: Train vs Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '04_model_comparison.png'), bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: 04_model_comparison.png")

    # --- Select best model ---
    # Rank by validation silhouette (primary) then validation DB (secondary, lower=better)
    best_model_name = max(results.keys(), key=lambda m: results[m]['val_sil'])
    print(f"\n  [*] Best Model: {best_model_name} (Val Silhouette: {results[best_model_name]['val_sil']:.4f})")
    print()

    return results, best_model_name


# ============================================================
# 7. GRADE ASSIGNMENT
# ============================================================
def assign_grades(df, results, best_model_name, X_train_pca, X_val_pca,
                  train_idx, val_idx, all_features, scaler, pca):
    """Map clusters to cytological grades based on feature centroids."""
    print("=" * 60)
    print("STEP 7: GRADE ASSIGNMENT")
    print("=" * 60)

    best = results[best_model_name]
    train_labels = best['train_labels']
    val_labels = best['val_labels']

    # Combine all labels into full dataset
    all_labels = np.zeros(len(df), dtype=int)
    all_labels[train_idx] = train_labels
    all_labels[val_idx] = val_labels

    # Compute cluster centroids in original feature space
    X_all = scaler.transform(df[all_features].values)
    cluster_profiles = {}
    for c in range(N_CLUSTERS):
        mask = all_labels == c
        cluster_profiles[c] = {
            'mean_area': df.loc[mask, 'Area'].mean(),
            'mean_circ': df.loc[mask, 'Circ.'].mean(),
            'mean_round': df.loc[mask, 'Round'].mean(),
            'mean_solidity': df.loc[mask, 'Solidity'].mean(),
            'mean_perim': df.loc[mask, 'Perim.'].mean(),
            'mean_ar': df.loc[mask, 'AR'].mean(),
            'mean_irreg': df.loc[mask, 'Irregularity_Index'].mean(),
            'count': mask.sum()
        }

    print("\n  Cluster Feature Profiles:")
    print(f"  {'Cluster':<10} {'Count':<8} {'Avg Area':<12} {'Avg Circ':<12} {'Avg Round':<12} {'Avg Solidity':<12} {'Avg Irreg':<12}")
    print("  " + "-" * 78)
    for c in range(N_CLUSTERS):
        p = cluster_profiles[c]
        print(f"  {c:<10} {p['count']:<8} {p['mean_area']:<12.2f} {p['mean_circ']:<12.4f} {p['mean_round']:<12.4f} {p['mean_solidity']:<12.4f} {p['mean_irreg']:<12.4f}")

    # Grade mapping logic:
    # Score each cluster: higher Area & higher Irregularity & lower Circularity = higher grade (worse)
    # Composite score = normalized(Area) + normalized(Irregularity) - normalized(Circularity) - normalized(Roundness)
    areas = np.array([cluster_profiles[c]['mean_area'] for c in range(N_CLUSTERS)])
    circs = np.array([cluster_profiles[c]['mean_circ'] for c in range(N_CLUSTERS)])
    irregs = np.array([cluster_profiles[c]['mean_irreg'] for c in range(N_CLUSTERS)])
    rounds = np.array([cluster_profiles[c]['mean_round'] for c in range(N_CLUSTERS)])

    # Normalize to 0-1
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else np.zeros_like(x)

    grade_score = norm(areas) + norm(irregs) - norm(circs) - norm(rounds)
    # Sort: lowest score = Grade 1, highest = Grade 3
    sorted_clusters = np.argsort(grade_score)
    cluster_to_grade = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        cluster_to_grade[cluster_id] = rank + 1  # Grade 1, 2, 3

    print(f"\n  Grade Mapping (based on feature analysis):")
    for c, g in sorted(cluster_to_grade.items()):
        print(f"    Cluster {c} -> Grade {g} (score: {grade_score[c]:.4f})")

    # Assign grades to all cells
    df['Cluster'] = all_labels
    df['Grade'] = df['Cluster'].map(cluster_to_grade)
    df['Grade_Label'] = df['Grade'].map({
        1: 'Well-differentiated',
        2: 'Moderately-differentiated',
        3: 'Poorly-differentiated'
    })
    df['Split'] = 'Train'
    df.loc[val_idx, 'Split'] = 'Validation'

    print(f"\n  Grade Distribution:")
    for g in [1, 2, 3]:
        count = (df['Grade'] == g).sum()
        pct = count / len(df) * 100
        print(f"    Grade {g}: {count} cells ({pct:.1f}%)")

    print()
    return df, cluster_to_grade, cluster_profiles


# ============================================================
# 8. CASE-LEVEL GRADING
# ============================================================
def compute_case_grades(df):
    """Aggregate cell-level grades to case-level using majority voting."""
    print("=" * 60)
    print("STEP 8: CASE-LEVEL GRADING (MAJORITY VOTING)")
    print("=" * 60)

    case_grades = []
    for case_no in sorted(df['Case no.'].unique()):
        case_data = df[df['Case no.'] == case_no]
        grade_counts = case_data['Grade'].value_counts()
        majority_grade = grade_counts.idxmax()
        total_cells = len(case_data)
        n_slides = case_data['Slide number'].nunique()

        grade_label = {
            1: 'Well-differentiated',
            2: 'Moderately-differentiated',
            3: 'Poorly-differentiated'
        }[majority_grade]

        case_grades.append({
            'Case_No': case_no,
            'Total_Cells': total_cells,
            'N_Slides': n_slides,
            'Grade_1_Count': grade_counts.get(1, 0),
            'Grade_2_Count': grade_counts.get(2, 0),
            'Grade_3_Count': grade_counts.get(3, 0),
            'Grade_1_Pct': grade_counts.get(1, 0) / total_cells * 100,
            'Grade_2_Pct': grade_counts.get(2, 0) / total_cells * 100,
            'Grade_3_Pct': grade_counts.get(3, 0) / total_cells * 100,
            'Final_Grade': majority_grade,
            'Grade_Label': grade_label
        })

    case_df = pd.DataFrame(case_grades)

    print(f"\n  {'Case':<8} {'Cells':<8} {'G1':<6} {'G2':<6} {'G3':<6} {'Final Grade':<25}")
    print("  " + "-" * 60)
    for _, row in case_df.iterrows():
        print(f"  {row['Case_No']:<8} {row['Total_Cells']:<8} "
              f"{row['Grade_1_Count']:<6} {row['Grade_2_Count']:<6} {row['Grade_3_Count']:<6} "
              f"Grade {row['Final_Grade']} ({row['Grade_Label']})")

    print(f"\n  Summary:")
    for g in [1, 2, 3]:
        n = (case_df['Final_Grade'] == g).sum()
        print(f"    Grade {g} cases: {n}")
    print()

    return case_df


# ============================================================
# 9. VISUALIZATIONS
# ============================================================
def generate_visualizations(df, X_train_pca, X_val_pca, train_idx, val_idx,
                            all_features, case_df, results, best_model_name):
    """Generate comprehensive visualizations."""
    print("=" * 60)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("=" * 60)

    # --- 9A: PCA Scatter Plot (Train + Validation) colored by Grade ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for g in [1, 2, 3]:
        # Training data
        mask_train = df.iloc[train_idx]['Grade'].values == g
        ax1.scatter(X_train_pca[mask_train, 0], X_train_pca[mask_train, 1],
                    c=GRADE_COLORS[g], label=f'Grade {g}', alpha=0.6, s=40, edgecolors='white', linewidth=0.5)

    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Training Set (75%)')
    ax1.legend(fontsize=10)

    for g in [1, 2, 3]:
        # Validation data
        mask_val = df.iloc[val_idx]['Grade'].values == g
        ax2.scatter(X_val_pca[mask_val, 0], X_val_pca[mask_val, 1],
                    c=GRADE_COLORS[g], label=f'Grade {g}', alpha=0.6, s=40,
                    edgecolors='white', linewidth=0.5, marker='D')

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Validation Set (25%)')
    ax2.legend(fontsize=10)

    plt.suptitle(f'PCA Visualization of Cytological Grades ({best_model_name})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '05_pca_grade_scatter.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 05_pca_grade_scatter.png")

    # --- 9B: Feature Distribution Boxplots ---
    key_features = ['Area', 'Perim.', 'Circ.', 'Round', 'Solidity', 'AR',
                    'Irregularity_Index', 'Pleomorphism_Index', 'Nuclear_Size_Index']

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for idx, feat in enumerate(key_features):
        ax = axes[idx // 3, idx % 3]
        data_for_plot = []
        for g in [1, 2, 3]:
            vals = df[df['Grade'] == g][feat].values
            data_for_plot.append(vals)

        bp = ax.boxplot(data_for_plot, labels=['Grade 1', 'Grade 2', 'Grade 3'],
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], [GRADE_COLORS[1], GRADE_COLORS[2], GRADE_COLORS[3]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat, fontweight='bold')
        ax.set_ylabel('Value')

    plt.suptitle('Feature Distributions by Cytological Grade', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_feature_boxplots.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 06_feature_boxplots.png")

    # --- 9C: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df[all_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', ax=ax, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '07_correlation_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 07_correlation_heatmap.png")

    # --- 9D: Dendrogram ---
    fig, ax = plt.subplots(figsize=(16, 8))
    linkage_matrix = linkage(X_train_pca[:100], method='ward')  # Subsample for readability
    dendrogram(linkage_matrix, ax=ax, color_threshold=0,
               leaf_font_size=8, truncate_mode='lastp', p=30)
    ax.set_title('Hierarchical Clustering Dendrogram (Training Subset)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '08_dendrogram.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 08_dendrogram.png")

    # --- 9E: Case-Level Grade Distribution ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of case grades
    grade_summary = case_df['Final_Grade'].value_counts().sort_index()
    bars = ax1.bar(grade_summary.index, grade_summary.values,
                   color=[GRADE_COLORS[g] for g in grade_summary.index],
                   edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, grade_summary.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha='center', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Cytological Grade')
    ax1.set_ylabel('Number of Cases')
    ax1.set_title('Case-Level Grade Distribution')
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Grade 1\n(Well-diff.)', 'Grade 2\n(Mod-diff.)', 'Grade 3\n(Poorly-diff.)'])

    # Stacked bar of grade proportions per case
    case_df_sorted = case_df.sort_values('Case_No')
    x_pos = range(len(case_df_sorted))
    ax2.bar(x_pos, case_df_sorted['Grade_1_Pct'], color=GRADE_COLORS[1], label='Grade 1')
    ax2.bar(x_pos, case_df_sorted['Grade_2_Pct'],
            bottom=case_df_sorted['Grade_1_Pct'], color=GRADE_COLORS[2], label='Grade 2')
    ax2.bar(x_pos, case_df_sorted['Grade_3_Pct'],
            bottom=case_df_sorted['Grade_1_Pct'] + case_df_sorted['Grade_2_Pct'],
            color=GRADE_COLORS[3], label='Grade 3')
    ax2.set_xlabel('Case No.')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Grade Composition per Case')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(case_df_sorted['Case_No'], rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper right')

    plt.suptitle('Case-Level Cytological Grading Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '09_case_grades.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 09_case_grades.png")

    # --- 9F: Cluster Centroid Heatmap ---
    centroid_data = []
    key_feats = ['Area', 'Perim.', 'Circ.', 'Feret', 'Round', 'Solidity', 'AR',
                 'Irregularity_Index', 'Nuclear_Size_Index']
    for g in [1, 2, 3]:
        means = df[df['Grade'] == g][key_feats].mean()
        centroid_data.append(means)

    centroid_df = pd.DataFrame(centroid_data, index=['Grade 1', 'Grade 2', 'Grade 3'])

    # Normalize for heatmap
    centroid_norm = (centroid_df - centroid_df.min()) / (centroid_df.max() - centroid_df.min())

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(centroid_norm, annot=centroid_df.round(2).values, cmap='YlOrRd',
                fmt='', ax=ax, linewidths=1, cbar_kws={'label': 'Normalized Value'})
    ax.set_title('Cluster Centroid Feature Profiles (Actual Values Shown)',
                 fontsize=14, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '10_centroid_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: 10_centroid_heatmap.png")

    print()


# ============================================================
# 10. SAVE OUTPUTS
# ============================================================
def save_outputs(df, case_df, results, best_model_name, all_features):
    """Save graded cells CSV, case grades CSV, and metrics summary."""
    print("=" * 60)
    print("STEP 10: SAVING OUTPUTS")
    print("=" * 60)

    # Save graded cells
    output_cols = ['Serial number', 'Case no.', 'Slide number', 'Cells', 'Split'] + \
                  MORPHOMETRIC_FEATURES + \
                  ['Nuclear_Size_Index', 'Irregularity_Index', 'Pleomorphism_Index',
                   'Feret_Ratio', 'NC_Compactness'] + \
                  ['Cluster', 'Grade', 'Grade_Label']

    cells_path = os.path.join(OUTPUT_DIR, 'graded_cells.csv')
    df[output_cols].to_csv(cells_path, index=False)
    print(f"  [OK] Saved: {cells_path}")

    # Save case grades
    cases_path = os.path.join(OUTPUT_DIR, 'case_grades.csv')
    case_df.to_csv(cases_path, index=False)
    print(f"  [OK] Saved: {cases_path}")

    # Save metrics summary
    metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics.csv')
    metrics_rows = []
    for name, res in results.items():
        metrics_rows.append({
            'Model': name,
            'Train_Silhouette': round(res['train_sil'], 4),
            'Val_Silhouette': round(res['val_sil'], 4),
            'Train_CH': round(res['train_ch'], 2),
            'Val_CH': round(res['val_ch'], 2),
            'Train_DB': round(res['train_db'], 4),
            'Val_DB': round(res['val_db'], 4),
            'Is_Best': name == best_model_name
        })
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"  [OK] Saved: {metrics_path}")

    print()


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  AI-BASED CYTOLOGICAL GRADING ALGORITHM")
    print("  Unsupervised Clustering | 75% Train / 25% Validation")
    print("=" * 60 + "\n")

    # Step 1: Load & Clean
    df = load_and_clean_data(INPUT_CSV)

    # Step 2: Feature Engineering
    df, all_features = engineer_features(df)

    # Step 3: Train-Validation Split
    X_train, X_val, train_idx, val_idx, scaler = split_data(df, all_features)

    # Step 4: PCA
    X_train_pca, X_val_pca, pca, n_components = apply_pca(X_train, X_val, all_features)

    # Step 5: Optimal Clusters
    find_optimal_clusters(X_train_pca)

    # Step 6: Train Clustering Models
    results, best_model_name = train_clustering_models(X_train_pca, X_val_pca)

    # Step 7: Assign Grades
    df, cluster_to_grade, cluster_profiles = assign_grades(
        df, results, best_model_name, X_train_pca, X_val_pca,
        train_idx, val_idx, all_features, scaler, pca
    )

    # Step 8: Case-Level Grading
    case_df = compute_case_grades(df)

    # Step 9: Visualizations
    generate_visualizations(df, X_train_pca, X_val_pca, train_idx, val_idx,
                            all_features, case_df, results, best_model_name)

    # Step 10: Save Outputs
    save_outputs(df, case_df, results, best_model_name, all_features)

    # Final Summary
    print("=" * 60)
    print("  [SUCCESS] PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Best Model:        {best_model_name}")
    print(f"  Total Cells:       {len(df)}")
    print(f"  Training Cells:    {len(train_idx)} ({len(train_idx)/len(df)*100:.0f}%)")
    print(f"  Validation Cells:  {len(val_idx)} ({len(val_idx)/len(df)*100:.0f}%)")
    print(f"  Total Cases:       {df['Case no.'].nunique()}")
    print(f"  PCA Components:    {n_components}")
    print(f"\n  Output Directory:  {OUTPUT_DIR}")
    print(f"  Plots Directory:   {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
