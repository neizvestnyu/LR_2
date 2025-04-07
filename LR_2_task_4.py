from sklearn.metrics import classification_report, confusion_matrix

# Функція для оцінки моделі
def evaluate_model(model, X_test, y_test, name):
    print(f"--- {name} ---")
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

evaluate_model(log_clf, X_test, y_test, "Логістична регресія")
evaluate_model(knn_clf, X_test, y_test, "k-NN")
evaluate_model(tree_clf, X_test, y_test, "Дерево рішень")
