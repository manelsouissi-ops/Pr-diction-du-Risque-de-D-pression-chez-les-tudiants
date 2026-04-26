"""Run EDA — saves all charts to webapp/static/img/"""
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'data' / 'Student_Depression_Dataset.csv'
IMG_DIR   = PROJECT_ROOT / 'webapp' / 'static' / 'img'
os.makedirs(IMG_DIR, exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        pass

print(f'Loading data from {DATA_PATH}')
df = pd.read_csv(DATA_PATH)
print(f'Shape: {df.shape}')

# 1. Depression distribution
counts = df['Depression'].value_counts()
labels = ['Not Depressed (0)', 'Depressed (1)']
colors = ['#16A34A', '#DC2626']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title("Depression Distribution", fontsize=14, fontweight='bold')
axes[1].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_title("Students per Class", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Number of Students")
for i, v in enumerate(counts.values):
    axes[1].text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'depression_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved depression_distribution.png')

# 2. Sleep vs depression
sleep_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others']
sleep_dep = (
    df.groupby('Sleep Duration')['Depression']
    .mean()
    .reindex([s for s in sleep_order if s in df['Sleep Duration'].unique()])
    * 100
)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(sleep_dep.index, sleep_dep.values, color='#2563EB', edgecolor='white', linewidth=1.5)
ax.set_title('Depression Rate by Sleep Duration', fontsize=14, fontweight='bold')
ax.set_xlabel('Sleep Duration')
ax.set_ylabel('Depression Rate (%)')
ax.set_ylim(0, 100)
for bar, val in zip(bars, sleep_dep.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(IMG_DIR / 'sleep_vs_depression.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved sleep_vs_depression.png')

# 3. Academic pressure vs depression
fig, ax = plt.subplots(figsize=(10, 5))
dep_rate = df.groupby('Academic Pressure')['Depression'].mean() * 100
bars = ax.bar(dep_rate.index.astype(str), dep_rate.values, color='#7C3AED', edgecolor='white')
ax.set_title('Depression Rate by Academic Pressure', fontsize=14, fontweight='bold')
ax.set_xlabel('Academic Pressure (1=low, 5=high)')
ax.set_ylabel('Depression Rate (%)')
ax.set_ylim(0, 100)
for bar, val in zip(bars, dep_rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'academic_pressure_vs_depression.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved academic_pressure_vs_depression.png')

# 4. CGPA vs depression
fig, ax = plt.subplots(figsize=(10, 5))
for val, label, color in [(0, 'Not Depressed', '#16A34A'), (1, 'Depressed', '#DC2626')]:
    subset = df[df['Depression'] == val]['CGPA'].dropna()
    subset.plot.kde(ax=ax, label=label, color=color, linewidth=2.5)
ax.set_title('CGPA Distribution by Class', fontsize=14, fontweight='bold')
ax.set_xlabel('CGPA')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / 'cgpa_vs_depression.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved cgpa_vs_depression.png')

# 5. Dietary habits vs depression
diet_dep = df.groupby('Dietary Habits')['Depression'].value_counts(normalize=True).unstack() * 100
diet_order = [d for d in ['Unhealthy', 'Moderate', 'Healthy', 'Others'] if d in diet_dep.index]
diet_dep = diet_dep.reindex(diet_order)
fig, ax = plt.subplots(figsize=(10, 5))
diet_dep.plot(kind='bar', ax=ax, color=['#16A34A', '#DC2626'], edgecolor='white', rot=0)
ax.set_title("Depression Rate by Dietary Habits", fontsize=14, fontweight='bold')
ax.set_xlabel("Dietary Habits")
ax.set_ylabel("Proportion (%)")
ax.legend(['Not Depressed', 'Depressed'])
plt.tight_layout()
plt.savefig(IMG_DIR / 'dietary_vs_depression.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved dietary_vs_depression.png')

# 6. Financial stress vs depression
fin_dep = df.groupby('Financial Stress')['Depression'].mean() * 100
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(fin_dep.index.astype(str), fin_dep.values, color='#D97706', edgecolor='white')
ax.set_title('Depression Rate by Financial Stress', fontsize=14, fontweight='bold')
ax.set_xlabel('Financial Stress (1=low, 5=high)')
ax.set_ylabel('Depression Rate (%)')
ax.set_ylim(0, 100)
for bar, val in zip(bars, fin_dep.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'financial_stress_vs_depression.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved financial_stress_vs_depression.png')

# 7. Correlation heatmap
num_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
            'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
            'Financial Stress', 'Depression']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, linewidths=.5, ax=ax, cbar_kws={'shrink': .8})
ax.set_title('Correlation Heatmap — Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved correlation_heatmap.png')

print()
print('=== EDA complete — all charts saved ===')
