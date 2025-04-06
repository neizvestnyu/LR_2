import pandas as pd

# Зчитування даних з CSV-файлу
df = pd.read_csv("Social_Network_Ads.csv")

# Виведення перших 5 рядків
print(df.head())

# Видалення зайвих стовпців
df.drop(['User ID'], axis=1, inplace=True)

# Перетворення категоріальних даних (стать)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

print(df.info())
