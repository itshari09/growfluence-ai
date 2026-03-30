import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load data
df = pd.read_csv('../data/data.csv')

# Features
X = df[['caption', 'comments', 'shares']]
y = df['likes']

# Preprocessing
text_features = 'caption'
numeric_features = ['comments', 'shares']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), text_features),
        ('num', 'passthrough', numeric_features)
    ])

# Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

# Save
joblib.dump(model, '../models/viral_model.pkl')

print("✅ Advanced model trained!")