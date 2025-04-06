import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Візуалізація меж класифікації (для перших двох ознак)
def plot_decision_boundary(X_set, y_set, model, title):
    X1, X2 = X_set[:, 1], X_set[:, 2]  # Візуалізуємо за Age і Salary
    X1_grid, X2_grid = np.meshgrid(
        np.arange(start=X1.min()-1, stop=X1.max()+1, step=0.01),
        np.arange(start=X2.min()-1, stop=X2.max()+1, step=0.01)
    )
    plt.contourf(
        X1_grid, X2_grid,
        model.predict(np.array([X_set[:, 0].mean(), X1_grid.ravel(), X2_grid.ravel()]).T.reshape(-1, 3)).reshape(X1_grid.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
    )
    plt.scatter(X1, X2, c=y_set, cmap=ListedColormap(('red', 'green')), edgecolor='k')
    plt.title(title)
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.show()

plot_decision_boundary(X_train, y_train, log_clf, "Logistic Regression (Train)")
plot_decision_boundary(X_test, y_test, log_clf, "Logistic Regression (Test)")
