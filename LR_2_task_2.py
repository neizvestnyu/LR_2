from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Вибір ознак та цільової змінної
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

# Розбиття на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Масштабування ознак
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
