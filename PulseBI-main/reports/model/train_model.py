import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector

# Load dataset
data = pd.read_excel("dataset/PulseBI_Test_Sales_Data.xlsx")

# Select input features and target
X = data[
    ["Quantity", "Unit_Price", "Discount_Percent",
     "Category", "Region", "Sales_Channel"]
]

y = data["Profit"]

# Identify numerical and categorical columns
numeric_features = ["Quantity", "Unit_Price", "Discount_Percent"]
categorical_features = ["Category", "Region", "Sales_Channel"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# Create model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("Model Training Completed!")
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Root Mean Squared Error:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2 Score:", r2_score(y_test, predictions))

# Save trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/sales_prediction_model.pkl")

print("Trained model saved successfully!")