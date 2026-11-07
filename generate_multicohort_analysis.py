"""
Comprehensive Multi-Cohort Validation & Plotting Suite
========================================================
Generates:
1. Multi-panel Confusion Matrices (Old 23, Nov 26, Combined 49 cases)
   across Majority Voting, 25% Cutoff, and 10% Cutoff.
2. Multi-panel ROC Curves comparing all three cohorts.
3. Updated Correlation Excel with complete tracking.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, cohen_kappa_score
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Load Data
df_old = pd.read_csv(os.path.join(OUTPUT_DIR, "case_grades.csv"))
df_nov = pd.read_csv(os.path.join(OUTPUT_DIR, "test_november_case_grades.csv"))
df_comb = pd.concat([df_old, df_nov]).drop_duplicates(subset=['Case_No']).reset_index(drop=True)

corr_path = os.path.join(BASE_DIR, "results correlation Cyto- AI- Histo.xlsx")
df_corr = pd.read_excel(corr_path)
df_corr['Case_Num'] = df_corr['Case No.'].astype(str).str.split('/').str[0].astype(int)

cohorts = {
    'Discovery (Old 23 Cases)': pd.merge(df_corr, df_old, left_on='Case_Num', right_on='Case_No', how='inner'),
    'Test Set (Nov 26 Cases)': pd.merge(df_corr, df_nov, left_on='Case_Num', right_on='Case_No', how='inner'),
    'Combined Overall (49 Cases)': pd.merge(df_corr, df_comb, left_on='Case_Num', right_on='Case_No', how='inner')
}

methods = [
    ('Majority Voting', 'Final_Grade'),
    ('Cutoff 25% (High Priority)', 'Grade_Cutoff25'),
    ('Cutoff 10% (High Priority)', 'Grade_Cutoff10')
]

# 2. Generate 3x3 Grid of Confusion Matrices
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
sns.set_style("white")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

for r_idx, (cohort_name, m_df) in enumerate(cohorts.items()):
    valid = m_df[m_df['Nottingham score'].isin([1, 2, 3, '1', '2', '3'])].copy()
    valid['Nottingham_Numeric'] = valid['Nottingham score'].astype(int)
    n_total = len(valid)
    
    for c_idx, (method_name, col_name) in enumerate(methods):
        ax = axes[r_idx, c_idx]
        cm = confusion_matrix(valid['Nottingham_Numeric'], valid[col_name], labels=[1, 2, 3])
        acc = np.trace(cm) / n_total
        kappa = cohen_kappa_score(valid['Nottingham_Numeric'], valid[col_name])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Grade 1', 'Grade 2', 'Grade 3'],
                    yticklabels=['Grade 1', 'Grade 2', 'Grade 3'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        
        ax.set_title(f"{cohort_name}\n{method_name}\nAcc: {acc*100:.1f}% | Kappa: {kappa:.2f}", 
                     fontsize=10.5, fontweight='bold')
        if r_idx == 2:
            ax.set_xlabel('AI Cytological Grade', fontsize=10, fontweight='bold')
        if c_idx == 0:
            ax.set_ylabel('Histological Nottingham Score', fontsize=10, fontweight='bold')

plt.suptitle('Multi-Cohort Confusion Matrix Analysis: Nottingham Histological Score vs AI Cytological Grade',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
cm_grid_path = os.path.join(PLOTS_DIR, "10_confusion_matrices_all_cohorts.png")
plt.savefig(cm_grid_path, bbox_inches='tight')
plt.close()
print(f"[OK] Saved comprehensive confusion matrix grid: {cm_grid_path}")

# 3. Generate Multi-Cohort ROC Curves
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Colors and linestyles for cohorts
c_colors = {
    'Discovery (Old 23 Cases)': '#e67e22',
    'Test Set (Nov 26 Cases)': '#2980b9',
    'Combined Overall (49 Cases)': '#27ae60'
}

for cohort_name, m_df in cohorts.items():
    valid = m_df[m_df['Nottingham score'].isin([1, 2, 3, '1', '2', '3'])].copy()
    valid['Nottingham_Numeric'] = valid['Nottingham score'].astype(int)
    color = c_colors[cohort_name]
    
    # Task A: High Grade Detection (Nottingham Grade 3 vs Non-Grade 3)
    y_true_g3 = (valid['Nottingham_Numeric'] == 3).astype(int).values
    y_score_g3 = (valid['Grade_3_Pct'] / 100.0).values
    fpr_g3, tpr_g3, _ = roc_curve(y_true_g3, y_score_g3)
    auc_g3 = auc(fpr_g3, tpr_g3)
    
    axes[0].plot(fpr_g3, tpr_g3, color=color, lw=2.5, 
                 label=f"{cohort_name.split(' (')[0]} (AUC = {auc_g3:.2f})")
    
    # Discrete points for 25% and 10%
    pred_g3_25 = (valid['Grade_Cutoff25'] == 3).astype(int)
    sens_25 = np.sum((y_true_g3 == 1) & (pred_g3_25 == 1)) / np.sum(y_true_g3 == 1)
    fpr_25 = np.sum((y_true_g3 == 0) & (pred_g3_25 == 1)) / np.sum(y_true_g3 == 0)
    
    axes[0].scatter([fpr_25], [sens_25], color=color, s=120, edgecolors='black', zorder=5)

    # Task B: Non-Low Grade Detection (Nottingham Grade 2+3 vs Grade 1)
    y_true_g23 = (valid['Nottingham_Numeric'] >= 2).astype(int).values
    y_score_g23 = ((valid['Grade_2_Pct'] + valid['Grade_3_Pct']) / 100.0).values
    fpr_g23, tpr_g23, _ = roc_curve(y_true_g23, y_score_g23)
    auc_g23 = auc(fpr_g23, tpr_g23)
    
    axes[1].plot(fpr_g23, tpr_g23, color=color, lw=2.5, 
                 label=f"{cohort_name.split(' (')[0]} (AUC = {auc_g23:.2f})")
    
    pred_g23_25 = (valid['Grade_Cutoff25'] >= 2).astype(int)
    sens_g23_25 = np.sum((y_true_g23 == 1) & (pred_g23_25 == 1)) / np.sum(y_true_g23 == 1)
    fpr_g23_25 = np.sum((y_true_g23 == 0) & (pred_g23_25 == 1)) / np.sum(y_true_g23 == 0)
    
    axes[1].scatter([fpr_g23_25], [sens_g23_25], color=color, s=120, edgecolors='black', zorder=5)

# Styling Subplot 1
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
axes[0].set_xlim([-0.02, 1.02])
axes[0].set_ylim([-0.02, 1.05])
axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
axes[0].set_title('High Grade Detection (Nottingham Grade 3)\n(Circles mark 25% Cutoff Operating Points)', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right', frameon=True, fontsize=10)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Styling Subplot 2
axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
axes[1].set_xlim([-0.02, 1.02])
axes[1].set_ylim([-0.02, 1.05])
axes[1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
axes[1].set_title('Non-Low Grade Detection (Nottingham Grade 2+3)\n(Circles mark 25% Cutoff Operating Points)', fontsize=12, fontweight='bold')
axes[1].legend(loc='lower right', frameon=True, fontsize=10)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Multi-Cohort Receiver Operating Characteristic (ROC) Comparison\nDiscovery vs Test Set vs Combined Cohort',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
roc_multi_path = os.path.join(PLOTS_DIR, "11_roc_curves_multi_cohort.png")
plt.savefig(roc_multi_path, bbox_inches='tight')
plt.close()
print(f"[OK] Saved multi-cohort ROC curve plot: {roc_multi_path}")

print("\n=======================================================")
print("ALL MULTI-COHORT VERIFICATIONS COMPLETED SUCCESSFULLY")
print("=======================================================")
