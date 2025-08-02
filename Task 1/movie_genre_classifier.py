import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset (replace 'movies.csv' with your actual file)
# The CSV should have columns: 'plot', 'genre'
data = pd.read_csv('movies.csv')

# Preprocessing
X = data['plot']
y = data['genre'] 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict
y_pred = clf.predict(X_test_tfidf)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Example prediction
example_plot = ["A young boy discovers he has magical powers and attends a school for wizards."]
example_tfidf = vectorizer.transform(example_plot)
predicted_genre = clf.predict(example_tfidf)
print('Predicted genre:', predicted_genre[0])
