import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("hi.csv")  


target_col = 'NObeyesdad'  


df = df.dropna(subset=[target_col])


X = df.drop(columns=[target_col])
y = df[target_col]


if y.dtype == 'object' or y.dtype.name == 'category':
    y = LabelEncoder().fit_transform(y)


X = pd.get_dummies(X, drop_first=True)


imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)


selector = SelectKBest(score_func=f_regression, k='all')  # You can change 'k'
X = selector.fit_transform(X, y)


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lasso = Lasso(alpha=0.1, random_state=42)  # Adjust alpha to control regularization strength
lasso.fit(X_train, y_train)


y_pred = lasso.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Lasso Regression")
print(f"MSE: {mse:.2f}, R²: {r2:.2f}")


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title("Lasso Regression: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()
