"""
Test Set Inference and Doctor Report Generator
================================================
Runs the trained cytological grading model on 'Test set November.xlsx',
computes case-level and cell-level statistics, saves CSVs, and generates
visual summary plots.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "output", "models", "cytological_grading_model.pkl")
TEST_FILE = os.path.join(BASE_DIR, "Test set November.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEST_PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots", "test_set_november")
os.makedirs(TEST_PLOTS_DIR, exist_ok=True)

# 1. Load Model Pipeline
print("Loading model pipeline...")
pipeline = joblib.load(MODEL_PATH)
scaler = pipeline['scaler']
pca = pipeline['pca']
centers = pipeline['cluster_centers']
cluster_to_grade = pipeline['cluster_to_grade_mapping']
features = pipeline['features']

# 2. Load and Clean Test Data
print("Loading Test set November.xlsx...")
df_test = pd.read_excel(TEST_FILE)
df_test.columns = df_test.columns.str.strip()

# Derived nuclear morphometrics
df_test['Nuclear_Size_Index'] = df_test['Area'] * df_test['Feret']
df_test['Irregularity_Index'] = 1 - df_test['Circ.']
df_test['Pleomorphism_Index'] = df_test['AR'] * (1 - df_test['Round'])
df_test['Feret_Ratio'] = df_test['MinFeret'] / df_test['Feret']
df_test['NC_Compactness'] = (df_test['Perim.'] ** 2) / (4 * np.pi * df_test['Area'])

# 3. Model Inference
X = df_test[features].values
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

dists = cdist(X_pca, centers, metric='euclidean')
clusters = np.argmin(dists, axis=1)
grades = [int(cluster_to_grade[c]) for c in clusters]
df_test['Cluster'] = clusters
df_test['Grade'] = grades

grade_labels = {
    1: 'Grade 1 (Low grade)',
    2: 'Grade 2 (Intermediate grade)',
    3: 'Grade 3 (High grade)'
}
short_labels = {
    1: 'Low grade',
    2: 'Intermediate grade',
    3: 'High grade'
}
df_test['Grade_Label'] = df_test['Grade'].map(short_labels)

# 4. Case-Level Aggregation
cases = []
for case_no, group in df_test.groupby('Case no.'):
    tot = len(group)
    n_slides = group['Slide no.'].nunique() if 'Slide no.' in group else 1
    g1 = (group['Grade'] == 1).sum()
    g2 = (group['Grade'] == 2).sum()
    g3 = (group['Grade'] == 3).sum()
    maj = group['Grade'].value_counts().idxmax()
    cases.append({
        'Case_No': int(case_no),
        'Total_Cells': int(tot),
        'N_Slides': int(n_slides),
        'Grade_1_Count': int(g1),
        'Grade_2_Count': int(g2),
        'Grade_3_Count': int(g3),
        'Grade_1_Pct': round(g1 / tot * 100, 1),
        'Grade_2_Pct': round(g2 / tot * 100, 1),
        'Grade_3_Pct': round(g3 / tot * 100, 1),
        'Final_Grade': int(maj),
        'Grade_Label': short_labels[maj]
    })
df_cases = pd.DataFrame(cases)
df_cases = df_cases.sort_values(by='Case_No').reset_index(drop=True)

# 5. Save Graded CSVs
test_cells_csv = os.path.join(OUTPUT_DIR, "test_november_graded_cells.csv")
test_cases_csv = os.path.join(OUTPUT_DIR, "test_november_case_grades.csv")
df_test.to_csv(test_cells_csv, index=False)
df_cases.to_csv(test_cases_csv, index=False)
print(f"[OK] Saved: {test_cells_csv}")
print(f"[OK] Saved: {test_cases_csv}")

# 6. Generate Clinical Visualization Plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
grade_colors = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}

# Plot 1: Case Grades Summary Bar Plot
fig, ax = plt.subplots(figsize=(14, 7))
cases_sorted = df_cases.sort_values(by=['Final_Grade', 'Case_No'])
x = np.arange(len(cases_sorted))
case_names = [f"Case {c}" for c in cases_sorted['Case_No']]

p1 = ax.bar(x, cases_sorted['Grade_1_Pct'], label='Grade 1 (Low)', color='#2ecc71', alpha=0.85)
p2 = ax.bar(x, cases_sorted['Grade_2_Pct'], bottom=cases_sorted['Grade_1_Pct'], label='Grade 2 (Intermediate)', color='#f39c12', alpha=0.85)
p3 = ax.bar(x, cases_sorted['Grade_3_Pct'], bottom=cases_sorted['Grade_1_Pct'] + cases_sorted['Grade_2_Pct'], label='Grade 3 (High)', color='#e74c3c', alpha=0.85)

ax.set_ylabel('Percentage of Cells (%)', fontsize=12, fontweight='bold')
ax.set_title('Cytological Grading Profile per Case - November Test Set (N=26 Cases)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(case_names, rotation=60, ha='right', fontsize=9)
ax.set_ylim(0, 105)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plot1_path = os.path.join(TEST_PLOTS_DIR, '01_case_grade_breakdown.png')
plt.savefig(plot1_path, bbox_inches='tight')
plt.close()

# Plot 2: Morphometric Parameter Distributions across Grades (Boxplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics_to_plot = [
    ('Area', 'Nuclear Area (μm²)', 'Nuclear Size'),
    ('Circ.', 'Circularity Index', 'Nuclear Contour Smoothness'),
    ('Solidity', 'Solidity (Convexity)', 'Nuclear Surface Regularity'),
    ('Irregularity_Index', 'Irregularity Index (1 - Circ.)', 'Nuclear Atypia / Pleomorphism')
]

for idx, (col, ylabel, title) in enumerate(metrics_to_plot):
    r, c = idx // 2, idx % 2
    ax = axes[r, c]
    sns.boxplot(x='Grade_Label', y=col, data=df_test, order=['Low grade', 'Intermediate grade', 'High grade'],
                palette=['#2ecc71', '#f39c12', '#e74c3c'], ax=ax, fliersize=2)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('')

plt.suptitle('Morphometric Feature Comparison across Cytological Grades (Test Set)', fontsize=15, fontweight='bold')
plt.tight_layout()
plot2_path = os.path.join(TEST_PLOTS_DIR, '02_morphometric_distributions.png')
plt.savefig(plot2_path, bbox_inches='tight')
plt.close()

# Print statistics for report
print("\n" + "="*50)
print("TEST SET NOVEMBER STATISTICAL SUMMARY")
print("="*50)
print(f"Total Cases: {len(df_cases)}")
print("Cases per Grade:")
print(df_cases['Grade_Label'].value_counts())
print("\nCell count per Grade:")
print(df_test['Grade_Label'].value_counts())

print("\nMean Morphometrics by Grade:")
stats_table = df_test.groupby('Grade_Label')[['Area', 'Circ.', 'Round', 'Solidity', 'Irregularity_Index', 'Perim.']].mean()
print(stats_table)
print("\n[OK] All tasks completed successfully.")
