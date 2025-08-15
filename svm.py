import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
X = data.data
y = data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def run_svm(kernel_type, C_value=1.0, gamma_value='scale'):
    model = SVC(kernel=kernel_type, C=C_value, gamma=gamma_value)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cv_score = np.mean(cross_val_score(model, X_scaled, y, cv=5))
    print(f"\nKernel: {kernel_type}, C={C_value}, gamma={gamma_value}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Cross-Validation Accuracy: {cv_score:.4f}")
run_svm('linear', C_value=1.0)
run_svm('rbf', C_value=1.0, gamma_value=0.1)
def plot_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

X_2d = X_scaled[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.2, random_state=42)

linear_model = SVC(kernel='linear', C=1.0).fit(X_train_2d, y_train_2d)
plot_boundary(linear_model, X_2d, y, "SVM (Linear Kernel)")

rbf_model = SVC(kernel='rbf', C=1.0, gamma=0.1).fit(X_train_2d, y_train_2d)
plot_boundary(rbf_model, X_2d, y, "SVM (RBF Kernel)")
