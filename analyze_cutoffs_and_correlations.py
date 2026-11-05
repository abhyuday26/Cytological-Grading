"""
Analysis Script: Cutoff-Based Cytological Grading, ROC Analysis, and Nottingham Score Correlation
===================================================================================================
1. Applies 25% and 10% higher-grade priority rules:
   - If Grade 3 % >= cutoff -> Final Grade = 3 (High grade)
   - Else if Grade 2 % >= cutoff -> Final Grade = 2 (Intermediate grade)
   - Else -> Final Grade = 1 (Low grade)
2. Updates case grading files with both 25% and 10% thresholds alongside Majority Voting.
3. Fills AI grade predictions into 'results correlation Cyto- AI- Histo.xlsx'.
4. Computes Multi-Class and Binary Confusion Matrices between Nottingham Histological Score and AI Grade.
5. Generates high-resolution ROC Curves comparing 25% vs 10% cutoffs and continuous risk predictors.
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
df_nov = pd.read_csv(os.path.join(OUTPUT_DIR, "test_november_case_grades.csv"))
df_old = pd.read_csv(os.path.join(OUTPUT_DIR, "case_grades.csv"))
corr_path = os.path.join(BASE_DIR, "results correlation Cyto- AI- Histo.xlsx")
df_corr = pd.read_excel(corr_path)

# 2. Add Cutoff Grading Logic
def assign_cutoff_grade(df):
    df = df.copy()
    # 25% threshold
    df['Grade_Cutoff25'] = np.where(df['Grade_3_Pct'] >= 25.0, 3, 
                           np.where(df['Grade_2_Pct'] >= 25.0, 2, 1))
    # 10% threshold
    df['Grade_Cutoff10'] = np.where(df['Grade_3_Pct'] >= 10.0, 3, 
                           np.where(df['Grade_2_Pct'] >= 10.0, 2, 1))
    
    label_map = {1: 'Low grade', 2: 'Intermediate grade', 3: 'High grade'}
    df['Grade_Label_Cutoff25'] = df['Grade_Cutoff25'].map(label_map)
    df['Grade_Label_Cutoff10'] = df['Grade_Cutoff10'].map(label_map)
    return df

df_nov_updated = assign_cutoff_grade(df_nov)
df_old_updated = assign_cutoff_grade(df_old)

# Save updated case grades
df_nov_updated.to_csv(os.path.join(OUTPUT_DIR, "test_november_case_grades.csv"), index=False)
df_old_updated.to_csv(os.path.join(OUTPUT_DIR, "case_grades.csv"), index=False)
print("[OK] Updated case grades saved with Cutoff25 and Cutoff10.")

# 3. Combine All Cases for Histopathology Correlation
df_all = pd.concat([df_old_updated, df_nov_updated]).drop_duplicates(subset=['Case_No']).reset_index(drop=True)
df_corr['Case_Num'] = df_corr['Case No.'].astype(str).str.split('/').str[0].astype(int)

# Merge correlation sheet with AI predictions
merged_corr = pd.merge(df_corr, df_all[['Case_No', 'Grade_1_Pct', 'Grade_2_Pct', 'Grade_3_Pct', 
                                       'Final_Grade', 'Grade_Cutoff25', 'Grade_Cutoff10',
                                       'Grade_Label', 'Grade_Label_Cutoff25', 'Grade_Label_Cutoff10']], 
                       left_on='Case_Num', right_on='Case_No', how='left')

# Fill 'AI grade' column in correlation excel with 25% cutoff (recommended) or majority voting
df_corr_filled = df_corr.copy()
ai_map_25 = merged_corr.set_index('Case No.')['Grade_Cutoff25'].to_dict()
ai_map_maj = merged_corr.set_index('Case No.')['Final_Grade'].to_dict()
ai_map_10 = merged_corr.set_index('Case No.')['Grade_Cutoff10'].to_dict()

df_corr_filled['AI grade (Cutoff 25%)'] = df_corr_filled['Case No.'].map(ai_map_25)
df_corr_filled['AI grade (Cutoff 10%)'] = df_corr_filled['Case No.'].map(ai_map_10)
df_corr_filled['AI grade (Majority Voting)'] = df_corr_filled['Case No.'].map(ai_map_maj)
df_corr_filled['AI grade'] = df_corr_filled['AI grade (Cutoff 25%)'] # Default to 25% cutoff

# Save updated correlation excel
updated_corr_path = os.path.join(OUTPUT_DIR, "results_correlation_Cyto_AI_Histo_updated.xlsx")
with pd.ExcelWriter(updated_corr_path, engine='openpyxl') as writer:
    df_corr_filled.to_excel(writer, index=False, sheet_name='Correlation')
print(f"[OK] Saved updated correlation Excel: {updated_corr_path}")

# Also update the base excel file directly
with pd.ExcelWriter(corr_path, engine='openpyxl') as writer:
    df_corr_filled.to_excel(writer, index=False, sheet_name='Sheet1')
print(f"[OK] Updated base correlation Excel: {corr_path}")

# 4. Filter for Valid Histopathology Nottingham Grades (1, 2, 3)
valid_histo = merged_corr[merged_corr['Nottingham score'].isin([1, 2, 3, '1', '2', '3']) & merged_corr['Final_Grade'].notna()].copy()
valid_histo['Nottingham_Numeric'] = valid_histo['Nottingham score'].astype(int)
valid_histo['Final_Grade'] = valid_histo['Final_Grade'].astype(int)
valid_histo['Grade_Cutoff25'] = valid_histo['Grade_Cutoff25'].astype(int)
valid_histo['Grade_Cutoff10'] = valid_histo['Grade_Cutoff10'].astype(int)

print(f"\nTotal cases evaluated against Nottingham Histological Score: {len(valid_histo)}")
print("Nottingham Grade breakdown in cohort:")
print(valid_histo['Nottingham_Numeric'].value_counts().sort_index())

# 5. Confusion Matrices Generation
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
sns.set_style("white")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
methods = [
    ('Majority Voting', 'Final_Grade'),
    ('Cutoff 25% (High Priority)', 'Grade_Cutoff25'),
    ('Cutoff 10% (High Priority)', 'Grade_Cutoff10')
]

for idx, (title, col) in enumerate(methods):
    cm = confusion_matrix(valid_histo['Nottingham_Numeric'], valid_histo[col], labels=[1, 2, 3])
    acc = np.trace(cm) / len(valid_histo)
    kappa = cohen_kappa_score(valid_histo['Nottingham_Numeric'], valid_histo[col])
    
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Grade 1', 'Grade 2', 'Grade 3'],
                yticklabels=['Grade 1', 'Grade 2', 'Grade 3'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f"{title}\nAccuracy: {acc*100:.1f}% | Kappa: {kappa:.2f}", fontsize=12, fontweight='bold')
    ax.set_xlabel('AI Cytological Grade', fontsize=11, fontweight='bold')
    ax.set_ylabel('Histological Nottingham Score', fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrix: Nottingham Histological Score vs AI Cytological Grade (N=26 Cases)', 
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
cm_plot_path = os.path.join(PLOTS_DIR, "08_confusion_matrix_nottingham_vs_ai.png")
plt.savefig(cm_plot_path, bbox_inches='tight')
plt.close()
print(f"[OK] Saved Confusion Matrix plot: {cm_plot_path}")

# 6. ROC Curves Generation
# Binary Task A: Detection of High Grade (Nottingham Grade 3 vs Non-Grade 3)
# Binary Task B: Detection of Intermediate/High Grade (Nottingham Grade 2+3 vs Grade 1)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Task A: High Grade Detection (Nottingham Grade 3)
y_true_g3 = (valid_histo['Nottingham_Numeric'] == 3).astype(int).values
y_score_g3 = (valid_histo['Grade_3_Pct'] / 100.0).values
fpr_g3_cont, tpr_g3_cont, _ = roc_curve(y_true_g3, y_score_g3)
auc_g3 = auc(fpr_g3_cont, tpr_g3_cont)

# Discrete points for 25% and 10% cutoffs on High Grade:
# Cutoff 25%:
pred_g3_25 = (valid_histo['Grade_Cutoff25'] == 3).astype(int)
sens_g3_25 = np.sum((y_true_g3 == 1) & (pred_g3_25 == 1)) / np.sum(y_true_g3 == 1)
fpr_g3_25 = np.sum((y_true_g3 == 0) & (pred_g3_25 == 1)) / np.sum(y_true_g3 == 0)

# Cutoff 10%:
pred_g3_10 = (valid_histo['Grade_Cutoff10'] == 3).astype(int)
sens_g3_10 = np.sum((y_true_g3 == 1) & (pred_g3_10 == 1)) / np.sum(y_true_g3 == 1)
fpr_g3_10 = np.sum((y_true_g3 == 0) & (pred_g3_10 == 1)) / np.sum(y_true_g3 == 0)

# Majority Voting point:
pred_g3_maj = (valid_histo['Final_Grade'] == 3).astype(int)
sens_g3_maj = np.sum((y_true_g3 == 1) & (pred_g3_maj == 1)) / np.sum(y_true_g3 == 1)
fpr_g3_maj = np.sum((y_true_g3 == 0) & (pred_g3_maj == 1)) / np.sum(y_true_g3 == 0)

ax1 = axes[0]
ax1.plot(fpr_g3_cont, tpr_g3_cont, color='#2980b9', lw=2.5, label=f'Continuous ROC (AUC = {auc_g3:.2f})')
ax1.scatter([fpr_g3_25], [sens_g3_25], color='#e67e22', s=160, zorder=5, label=f'25% Cutoff (Sens={sens_g3_25:.2f}, FPR={fpr_g3_25:.2f})')
ax1.scatter([fpr_g3_10], [sens_g3_10], color='#e74c3c', s=160, zorder=5, label=f'10% Cutoff (Sens={sens_g3_10:.2f}, FPR={fpr_g3_10:.2f})')
ax1.scatter([fpr_g3_maj], [sens_g3_maj], color='#7f8c8d', s=140, marker='s', zorder=5, label=f'Majority Voting (Sens={sens_g3_maj:.2f}, FPR={fpr_g3_maj:.2f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.05])
ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax1.set_title('ROC Curve: Detection of High Grade (Nottingham Grade 3)\nPredictor: AI Nuclear Grade 3 Percentage', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, linestyle=':', alpha=0.6)

# Task B: Non-Low Grade Detection (Nottingham Grade 2 & 3 vs Grade 1)
y_true_g23 = (valid_histo['Nottingham_Numeric'] >= 2).astype(int).values
y_score_g23 = ((valid_histo['Grade_2_Pct'] + valid_histo['Grade_3_Pct']) / 100.0).values
fpr_g23_cont, tpr_g23_cont, _ = roc_curve(y_true_g23, y_score_g23)
auc_g23 = auc(fpr_g23_cont, tpr_g23_cont)

# Cutoff 25% on Grade 2+:
pred_g23_25 = (valid_histo['Grade_Cutoff25'] >= 2).astype(int)
sens_g23_25 = np.sum((y_true_g23 == 1) & (pred_g23_25 == 1)) / np.sum(y_true_g23 == 1)
fpr_g23_25 = np.sum((y_true_g23 == 0) & (pred_g23_25 == 1)) / np.sum(y_true_g23 == 0)

# Cutoff 10% on Grade 2+:
pred_g23_10 = (valid_histo['Grade_Cutoff10'] >= 2).astype(int)
sens_g23_10 = np.sum((y_true_g23 == 1) & (pred_g23_10 == 1)) / np.sum(y_true_g23 == 1)
fpr_g23_10 = np.sum((y_true_g23 == 0) & (pred_g23_10 == 1)) / np.sum(y_true_g23 == 0)

# Majority Voting on Grade 2+:
pred_g23_maj = (valid_histo['Final_Grade'] >= 2).astype(int)
sens_g23_maj = np.sum((y_true_g23 == 1) & (pred_g23_maj == 1)) / np.sum(y_true_g23 == 1)
fpr_g23_maj = np.sum((y_true_g23 == 0) & (pred_g23_maj == 1)) / np.sum(y_true_g23 == 0)

ax2 = axes[1]
ax2.plot(fpr_g23_cont, tpr_g23_cont, color='#27ae60', lw=2.5, label=f'Continuous ROC (AUC = {auc_g23:.2f})')
ax2.scatter([fpr_g23_25], [sens_g23_25], color='#e67e22', s=160, zorder=5, label=f'25% Cutoff (Sens={sens_g23_25:.2f}, FPR={fpr_g23_25:.2f})')
ax2.scatter([fpr_g23_10], [sens_g23_10], color='#e74c3c', s=160, zorder=5, label=f'10% Cutoff (Sens={sens_g23_10:.2f}, FPR={fpr_g23_10:.2f})')
ax2.scatter([fpr_g23_maj], [sens_g23_maj], color='#7f8c8d', s=140, marker='s', zorder=5, label=f'Majority Voting (Sens={sens_g23_maj:.2f}, FPR={fpr_g23_maj:.2f})')
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.05])
ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
ax2.set_title('ROC Curve: Detection of High/Intermediate Grade (Nottingham Grade 2+3)\nPredictor: AI Nuclear Grade (2+3) Percentage', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', frameon=True, fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Receiver Operating Characteristic (ROC) Analysis: 25% vs 10% Cutoffs against Histopathology', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
roc_plot_path = os.path.join(PLOTS_DIR, "09_roc_curves_cutoffs.png")
plt.savefig(roc_plot_path, bbox_inches='tight')
plt.close()
print(f"[OK] Saved ROC curve plot: {roc_plot_path}")

print("\n=======================================================")
print("STATISTICAL SUMMARY REPORT FOR PATHOLOGIST")
print("=======================================================")
for title, col in methods:
    cm = confusion_matrix(valid_histo['Nottingham_Numeric'], valid_histo[col], labels=[1, 2, 3])
    acc = np.trace(cm) / len(valid_histo)
    kappa = cohen_kappa_score(valid_histo['Nottingham_Numeric'], valid_histo[col])
    rho, pval = spearmanr(valid_histo['Nottingham_Numeric'], valid_histo[col])
    print(f"\n--- {title} ---")
    print(f"Confusion Matrix (Rows=Nottingham 1,2,3; Cols=AI 1,2,3):\n{cm}")
    print(f"Overall Accuracy: {acc*100:.2f}% | Cohen's Kappa: {kappa:.4f} | Spearman Rho: {rho:.4f} (p={pval:.4f})")
