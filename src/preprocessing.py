import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

SLEEP_MAP = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3,
    'Others': 1,
}

DIET_MAP = {
    'Unhealthy': 0,
    'Moderate': 1,
    'Healthy': 2,
    'Others': 1,
}

TOP_CITIES = [
    'Kalyan', 'Srinagar', 'Hyderabad', 'Vasai-Virar', 'Lucknow',
    'Thane', 'Ludhiana', 'Agra', 'Surat', 'Kolkata',
]

TOP_DEGREES = [
    'Class 12', 'B.Ed', 'B.Com', 'B.Arch', 'BCA',
    'MSc', 'B.Tech', 'MCA', 'M.Tech', 'BHM',
]

NUMERIC_COLS = [
    'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
    'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
    'Financial Stress', 'Sleep Duration', 'Dietary Habits',
    'Family History of Mental Illness',
]

DROP_COLS = ['id', 'Profession', 'Have you ever had suicidal thoughts ?']

TARGET_COL = 'Depression'


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.financial_stress_median_ = None

    def _base_encode(self, df):
        df = df.copy()

        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        if TARGET_COL in df.columns:
            cols_to_drop.append(TARGET_COL)
        df = df.drop(columns=cols_to_drop)

        df['Sleep Duration'] = df['Sleep Duration'].map(SLEEP_MAP).fillna(1).astype(float)
        df['Dietary Habits'] = df['Dietary Habits'].map(DIET_MAP).fillna(1).astype(float)
        df['Family History of Mental Illness'] = (
            df['Family History of Mental Illness']
            .map({'Yes': 1, 'No': 0})
            .fillna(0)
            .astype(float)
        )

        df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')
        if self.financial_stress_median_ is not None:
            df['Financial Stress'] = df['Financial Stress'].fillna(self.financial_stress_median_)

        df['Gender_Female'] = (df['Gender'] == 'Female').astype(int)
        df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
        df = df.drop(columns=['Gender'])

        city_cols = {}
        for city in TOP_CITIES:
            city_cols[f'City_{city}'] = (df['City'] == city).astype(int)
        city_cols['City_Other'] = (~df['City'].isin(TOP_CITIES)).astype(int)
        df = df.drop(columns=['City'])
        df = pd.concat([df, pd.DataFrame(city_cols, index=df.index)], axis=1)

        degree_cols = {}
        for deg in TOP_DEGREES:
            safe = deg.replace(' ', '_').replace('.', '')
            degree_cols[f'Degree_{safe}'] = (df['Degree'] == deg).astype(int)
        degree_cols['Degree_Other'] = (~df['Degree'].isin(TOP_DEGREES)).astype(int)
        df = df.drop(columns=['Degree'])
        df = pd.concat([df, pd.DataFrame(degree_cols, index=df.index)], axis=1)

        for col in ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
                    'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def fit_transform(self, df):
        fs = pd.to_numeric(df['Financial Stress'], errors='coerce')
        self.financial_stress_median_ = fs.median()

        encoded = self._base_encode(df)
        encoded = encoded.fillna(encoded.median(numeric_only=True))
        encoded = encoded.fillna(0)

        self.feature_names_ = list(encoded.columns)
        scaled = self.scaler.fit_transform(encoded)
        return pd.DataFrame(scaled, columns=self.feature_names_, index=encoded.index)

    def transform(self, df):
        encoded = self._base_encode(df)

        if 'Financial Stress' in encoded.columns:
            encoded['Financial Stress'] = encoded['Financial Stress'].fillna(
                self.financial_stress_median_ if self.financial_stress_median_ is not None else 3.0
            )

        encoded = encoded.fillna(0)

        for col in self.feature_names_:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded[self.feature_names_]

        scaled = self.scaler.transform(encoded)
        return pd.DataFrame(scaled, columns=self.feature_names_, index=encoded.index)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def make_target_label(df):
    return df[TARGET_COL].astype(int)
