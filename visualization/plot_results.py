import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load features and model
df = pd.read_csv("data/features.csv")
X = df.drop(columns=['label'])
y = df['label']

model = joblib.load("models/random_forest.pkl")
y_pred = model.predict(X)

# Confusion matrix
cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()), cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("visualization/confusion_matrix.png")
plt.show()
