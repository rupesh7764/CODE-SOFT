import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset (replace 'Task 3/churn_data.csv' with your actual file)
# The CSV should have columns like: 'feature1', 'feature2', ..., 'churn'
data = pd.read_csv('Task 3/churn_data.csv')

# Features and target
X = data.drop('churn', axis=1)
y = data['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
}

for name, model in models.items():
    print(f'\nTraining {name}...')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f'Accuracy for {name}:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Example prediction
example = X_test.iloc[0:1]
example_scaled = scaler.transform(example)
pred_rf = models['Random Forest'].predict(example_scaled)
print('Random Forest prediction for first test customer:', 'Churn' if pred_rf[0] == 1 else 'No Churn')
