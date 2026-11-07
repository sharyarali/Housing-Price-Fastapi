# import liabraries

import pandas as pd 
import numpy as np 
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# load Data 
df = pd.read_csv("Housing.csv")
#target variable
target = "price"

# Features
X = df.drop(columns=[target])
y = df[target]

# identify feature tyes
numeric_feature = X.select_dtypes(include=['int64','float64']).columns
categorical_feature = X.select_dtypes(include=['object']).columns

# preprocessing
numeric_transform = StandardScaler()
categorical_transform = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transform, numeric_feature),
        ('cat', categorical_transform, categorical_feature)
    ]
)

# ML Model 
model = RandomForestRegressor(n_estimators=200, random_state=42)

pipeline= Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',model)
])

# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)
# Train model 
pipeline.fit(X_train, y_train)

# Evalute 
preds = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

# Save Model 

joblib.dump(pipeline, 'Housing_price_model.pkl')

print("Model saved successfully")