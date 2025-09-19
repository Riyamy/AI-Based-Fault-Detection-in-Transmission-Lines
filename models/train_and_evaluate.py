import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load features
df = pd.read_csv("data/features.csv")
X = df.drop(columns=['label'])
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.pkl")

# Train SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
joblib.dump(svm, "models/svm.pkl")

# Evaluate models
print("\n=== Random Forest Report ===")
print(classification_report(y_test, rf.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf.predict(X_test)))

print("\n=== SVM Report ===")
print(classification_report(y_test, svm.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm.predict(X_test)))
