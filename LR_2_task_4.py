# LR_2_task_4.py

# КРОК 1. ІМПОРТ БІБЛІОТЕК
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# КРОК 2. ЗАВАНТАЖЕННЯ ДАНИХ
# ⚠ Переконайся, що файл знаходиться у тій самій папці
filename = 'adult.data'

# ⚠️ Налаштуй параметри, якщо структура файлу інша (наприклад, інший роздільник)
dataset = pd.read_csv(filename)

# КРОК 3. ПЕРЕВІРКА ДАНИХ
print("Перші 5 рядків:\n", dataset.head())
print("\nФорма датасету:", dataset.shape)
print("\nУнікальні мітки класу:", dataset.iloc[:, -1].unique())

# КРОК 4. РОЗДІЛЕННЯ НА ВХІДНІ ОЗНАКИ ТА МІТКИ
X = dataset.iloc[:, :-1].values  # всі колонки, крім останньої — ознаки
y = dataset.iloc[:, -1].values  # остання колонка — мітки

# КРОК 5. РОЗБИВКА НА НАВЧАЛЬНУ І ТЕСТОВУ ВИБІРКИ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# КРОК 6. ПІДГОТОВКА АЛГОРИТМІВ
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# КРОК 7. ОЦІНКА КОЖНОГО АЛГОРИТМУ
results = []
names = []

print("\n=== Точність класифікаторів (Cross-Validation, 10 folds) ===\n")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} (+/- {cv_results.std():.4f})")

# КРОК 8. ВІЗУАЛІЗАЦІЯ ПОРІВНЯННЯ
plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів класифікації')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# КРОК 9. ТЕСТУВАННЯ НА КОНТРОЛЬНІЙ ВИБІРЦІ (НАПРИКЛАД, З SVM)
final_model = SVC(gamma='auto')
final_model.fit(X_train, y_train)
predictions = final_model.predict(X_test)

print("\n=== Оцінка якості на тестовій вибірці (SVM) ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
