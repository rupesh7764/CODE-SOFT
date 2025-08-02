import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset (replace 'Task 2/creditcard.csv' with your actual file)
# The CSV should have columns like: 'Time', 'V1', ..., 'V28', 'Amount', 'Class'
data = pd.read_csv('Task 2/creditcard.csv')

# Features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
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
print('Random Forest prediction for first test transaction:', 'Fraud' if pred_rf[0] == 1 else 'Legitimate')
