"""Train LightGBM model and save artifacts."""
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

PROJECT_ROOT = Path(__file__).parent
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
print(f'Loading data from {DATA_PATH}')
df = pd.read_csv(DATA_PATH)
print(f'Shape: {df.shape}')

preprocessor = Preprocessor()
X = preprocessor.fit_transform(df)
y = make_target_label(df)
print(f'Features: {X.shape[1]}, Samples: {len(y)}')
print('Target:', y.value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    is_unbalance=True,
    random_state=42,
    verbose=-1,
)
print('Training LightGBM...')
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
)
print(f'Training complete. Best iteration: {model.best_iteration_}')

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f'Accuracy : {acc:.4f}')
print(f'F1-Score : {f1:.4f}')
print(f'AUC-ROC  : {auc:.4f}')

metrics = {
    'accuracy':   round(acc, 4),
    'f1':         round(f1, 4),
    'auc_roc':    round(auc, 4),
    'n_features': int(X.shape[1]),
    'train_size': int(len(X_train)),
    'test_size':  int(len(X_test)),
}
metrics_path = PROJECT_ROOT / 'webapp' / 'static' / 'metrics.json'
with open(metrics_path, 'w') as fh:
    json.dump(metrics, fh, indent=2)
print('Metrics saved to', metrics_path)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(IMG_DIR / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved confusion_matrix.png')

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_n = min(20, len(feat_imp))
fig, ax = plt.subplots(figsize=(10, 8))
feat_imp.head(top_n).sort_values().plot(kind='barh', ax=ax, color='#2563EB')
ax.set_title(f'Top {top_n} Feature Importances — LightGBM', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(IMG_DIR / 'feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved feature_importance.png')

# SHAP
print('Computing SHAP values...')
explainer = shap.TreeExplainer(model)
sample_size = min(500, len(X_test))
X_sample = X_test.iloc[:sample_size]
shap_values = explainer.shap_values(X_sample)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(sv, X_sample, plot_type='bar', show=False, max_display=20)
plt.title('SHAP — Global Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'shap_summary.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved shap_summary.png')

# Save model and preprocessor
model_path = PROJECT_ROOT / 'model.pkl'
prep_path  = PROJECT_ROOT / 'preprocessor.pkl'
joblib.dump(model, model_path)
preprocessor.save(prep_path)
print(f'Model saved to      : {model_path}')
print(f'Preprocessor saved  : {prep_path}')

print()
print('=== Modeling complete ===')
print(f'  Accuracy : {acc:.4f}')
print(f'  F1-Score : {f1:.4f}')
print(f'  AUC-ROC  : {auc:.4f}')
