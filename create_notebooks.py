"""Generates notebooks/01_EDA.ipynb and notebooks/02_Modeling.ipynb"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(ROOT, 'notebooks')
os.makedirs(NB_DIR, exist_ok=True)


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": os.urandom(8).hex(),
        "metadata": {},
        "outputs": [],
        "source": src if isinstance(src, list) else [src],
    }


def md(src):
    return {
        "cell_type": "markdown",
        "id": os.urandom(8).hex(),
        "metadata": {},
        "source": [src] if isinstance(src, str) else src,
    }


# ─── 01_EDA ──────────────────────────────────────────────────────────────────
EDA_CODE = r"""
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

NOTEBOOK_DIR = Path(os.getcwd())
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
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

print('Project root:', PROJECT_ROOT)
df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)
print(df.isnull().sum())

# 1. Depression distribution
counts = df['Depression'].value_counts()
labels = ['Not Depressed (0)', 'Depressed (1)']
colors = ['#16A34A', '#DC2626']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Depression Distribution', fontsize=14, fontweight='bold')
axes[1].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_title('Students per Class', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Students')
for i, v in enumerate(counts.values):
    axes[1].text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'depression_distribution.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved depression_distribution.png')

# 2. Sleep vs depression
sleep_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others']
sleep_dep = (df.groupby('Sleep Duration')['Depression']
    .mean()
    .reindex([s for s in sleep_order if s in df['Sleep Duration'].unique()])
    * 100)
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
dep_rate = df.groupby('Academic Pressure')['Depression'].mean() * 100
fig, ax = plt.subplots(figsize=(10, 5))
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
    df[df['Depression'] == val]['CGPA'].dropna().plot.kde(ax=ax, label=label, color=color, linewidth=2.5)
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
fig, ax = plt.subplots(figsize=(10, 5))
diet_dep.reindex(diet_order).plot(kind='bar', ax=ax, color=['#16A34A', '#DC2626'], edgecolor='white', rot=0)
ax.set_title('Depression Rate by Dietary Habits', fontsize=14, fontweight='bold')
ax.set_xlabel('Dietary Habits')
ax.set_ylabel('Proportion (%)')
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
ax.set_title('Correlation Heatmap - Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved correlation_heatmap.png')
print('=== EDA complete ===')
""".strip()

# ─── 02_Modeling ─────────────────────────────────────────────────────────────
MODEL_CODE = r"""
import os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import shap

NOTEBOOK_DIR = Path(os.getcwd())
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_DIR = PROJECT_ROOT / 'webapp' / 'static' / 'img'
os.makedirs(IMG_DIR, exist_ok=True)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        pass

from src.preprocessing import Preprocessor, make_target_label

DATA_PATH = PROJECT_ROOT / 'data' / 'Student_Depression_Dataset.csv'
df = pd.read_csv(DATA_PATH)
print('Dataset shape:', df.shape)

preprocessor = Preprocessor()
X = preprocessor.fit_transform(df)
y = make_target_label(df)
print('Features:', X.shape[1], '  Samples:', len(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {X_train.shape}  Test: {X_test.shape}')

model = lgb.LGBMClassifier(
    objective='binary', n_estimators=500, learning_rate=0.05,
    max_depth=6, is_unbalance=True, random_state=42, verbose=-1)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
print('Best iteration:', model.best_iteration_)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f'Accuracy:{acc:.4f}  F1:{f1:.4f}  AUC-ROC:{auc:.4f}')

metrics = {'accuracy': round(acc,4), 'f1': round(f1,4), 'auc_roc': round(auc,4),
           'n_features': int(X.shape[1]), 'train_size': int(len(X_train)), 'test_size': int(len(X_test))}
with open(PROJECT_ROOT / 'webapp' / 'static' / 'metrics.json', 'w') as fh:
    json.dump(metrics, fh, indent=2)
print('Metrics saved.')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved confusion_matrix.png')

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_n = min(20, len(feat_imp))
fig, ax = plt.subplots(figsize=(10, 8))
feat_imp.head(top_n).sort_values().plot(kind='barh', ax=ax, color='#2563EB')
ax.set_title(f'Top {top_n} Feature Importances - LightGBM', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(IMG_DIR / 'feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved feature_importance.png')

# SHAP
explainer = shap.TreeExplainer(model)
X_sample = X_test.iloc[:min(500, len(X_test))]
shap_values = explainer.shap_values(X_sample)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv, X_sample, plot_type='bar', show=False, max_display=20)
plt.title('SHAP - Global Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'shap_summary.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved shap_summary.png')

# Save artifacts
joblib.dump(model, PROJECT_ROOT / 'model.pkl')
preprocessor.save(PROJECT_ROOT / 'preprocessor.pkl')
print('model.pkl and preprocessor.pkl saved.')
print('=== Modeling complete ===')
""".strip()

eda_cells   = [md("# EDA - Student Depression Dataset"), code(EDA_CODE)]
model_cells = [md("# Modeling - LightGBM Depression Predictor"), code(MODEL_CODE)]

eda_path   = os.path.join(NB_DIR, '01_EDA.ipynb')
model_path = os.path.join(NB_DIR, '02_Modeling.ipynb')

with open(eda_path,   'w', encoding='utf-8') as f:
    json.dump(nb(eda_cells),   f, indent=1, ensure_ascii=False)

with open(model_path, 'w', encoding='utf-8') as f:
    json.dump(nb(model_cells), f, indent=1, ensure_ascii=False)

print(f'Created: {eda_path}')
print(f'Created: {model_path}')
