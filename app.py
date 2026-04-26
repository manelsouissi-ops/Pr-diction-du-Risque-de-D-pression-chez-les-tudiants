import os
import json
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'webapp', 'templates'),
    static_folder=os.path.join(BASE_DIR, 'webapp', 'static'),
    static_url_path='/static',
)

# Load model artifacts at startup
model = None
preprocessor = None
DEMO_MODE = False

try:
    model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
    preprocessor = joblib.load(os.path.join(BASE_DIR, 'preprocessor.pkl'))
    print("[OK] Model and preprocessor loaded.")
except Exception as e:
    DEMO_MODE = True
    print(f"[DEMO MODE] Artifacts not found ({e}). Run notebooks to generate them.")


def _build_input_df(form):
    return pd.DataFrame([{
        'Gender':                        form.get('gender', 'Male'),
        'Age':                           float(form.get('age', 22)),
        'City':                          form.get('city', 'Other'),
        'Academic Pressure':             float(form.get('academic_pressure', 3)),
        'Work Pressure':                 float(form.get('work_pressure', 1)),
        'CGPA':                          float(form.get('cgpa', 7.5)),
        'Study Satisfaction':            float(form.get('study_satisfaction', 3)),
        'Job Satisfaction':              float(form.get('job_satisfaction', 0)),
        'Sleep Duration':                form.get('sleep_duration', '7-8 hours'),
        'Dietary Habits':                form.get('dietary_habits', 'Moderate'),
        'Degree':                        form.get('degree', 'B.Tech'),
        'Work/Study Hours':              float(form.get('work_study_hours', 6)),
        'Financial Stress':              float(form.get('financial_stress', 3)),
        'Family History of Mental Illness': form.get('family_history', 'No'),
    }])


@app.route('/')
def index():
    return render_template('index.html', active='home')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if DEMO_MODE:
            return jsonify({
                'demo_mode': True,
                'depression_risk': 0,
                'risk_label': 'Faible (demo)',
                'risk_color': 'success',
                'confidence': 0.72,
                'probability_depressed': 0.28,
                'probability_not_depressed': 0.72,
                'message': 'Exécutez les notebooks pour générer le modèle réel.',
            })

        try:
            df = _build_input_df(request.form)
            X = preprocessor.transform(df)
            pred = int(model.predict(X)[0])
            prob = model.predict_proba(X)[0].tolist()

            return jsonify({
                'depression_risk':         pred,
                'risk_label':             'Élevé' if pred == 1 else 'Faible',
                'risk_color':             'danger' if pred == 1 else 'success',
                'confidence':             round(float(max(prob)) * 100, 1),
                'probability_depressed':  round(float(prob[1]) * 100, 1),
                'probability_not_depressed': round(float(prob[0]) * 100, 1),
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('predict.html', active='predict')


@app.route('/eda')
def eda():
    img_dir = os.path.join(BASE_DIR, 'webapp', 'static', 'img')
    charts = [
        ('depression_distribution.png',          'Distribution de la Dépression',
         'Répartition des étudiants déprimés vs non déprimés.'),
        ('sleep_vs_depression.png',               'Sommeil vs Dépression',
         'Taux de dépression selon la durée de sommeil.'),
        ('academic_pressure_vs_depression.png',   'Pression Académique vs Dépression',
         'Impact de la pression académique sur la dépression.'),
        ('cgpa_vs_depression.png',                'CGPA vs Dépression',
         'Distribution du CGPA pour les deux classes.'),
        ('dietary_vs_depression.png',             'Alimentation vs Dépression',
         'Taux de dépression selon les habitudes alimentaires.'),
        ('financial_stress_vs_depression.png',    'Stress Financier vs Dépression',
         'Impact du stress financier sur la dépression.'),
        ('correlation_heatmap.png',               'Carte de Corrélation',
         'Corrélations entre les variables numériques.'),
    ]
    available = [(fn, title, desc)
                 for fn, title, desc in charts
                 if os.path.exists(os.path.join(img_dir, fn))]
    missing = len(charts) - len(available)
    return render_template('eda.html', active='eda', charts=available, missing=missing)


@app.route('/models')
def models():
    metrics_path = os.path.join(BASE_DIR, 'webapp', 'static', 'metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    img_dir = os.path.join(BASE_DIR, 'webapp', 'static', 'img')
    model_charts = [
        ('confusion_matrix.png',    'Matrice de Confusion'),
        ('feature_importance.png',  'Importance des Variables'),
        ('shap_summary.png',        'Analyse SHAP'),
    ]
    available_charts = [(fn, title)
                        for fn, title in model_charts
                        if os.path.exists(os.path.join(img_dir, fn))]
    return render_template('models.html', active='models',
                           metrics=metrics, charts=available_charts,
                           demo_mode=DEMO_MODE)


@app.route('/about')
def about():
    return render_template('about.html', active='about')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
