import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('Iris.csv')

# Drop the ID column
df.drop(columns=['Id'], inplace=True)

# Feature-target split
X = df.drop('Species', axis=1)
y = df['Species']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluate different values of K
accuracy_list = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracy_list.append(acc)
    print(f"K = {k}, Accuracy = {acc:.4f}")

# Plot Accuracy vs K
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), accuracy_list, marker='o')
plt.title('K vs Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

# Train and evaluate best model
best_k = np.argmax(accuracy_list) + 1
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
y_pred = final_knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=final_knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_knn.classes_)
disp.plot()
plt.title(f'Confusion Matrix (K = {best_k})')
plt.show()
