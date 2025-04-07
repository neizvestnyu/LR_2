from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Логістична регресія
log_clf = LogisticRegression(random_state=0)
log_clf.fit(X_train, y_train)

# k-NN
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

# Дерево рішень
tree_clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
tree_clf.fit(X_train, y_train)
