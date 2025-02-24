import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import xgboost as xgb

# Загрузка данных для обучения модели
df = pd.read_csv('parkinsons.data')



# Подготовка данных
all_features = df.loc[:, df.columns != 'status'].values[:, 1:]
out_come = df.loc[:, 'status'].values



# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(all_features, out_come, test_size=0.2, random_state=1, stratify=out_come)


# Масштабирование данных
scaler = MinMaxScaler((-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Гиперпараметры модели
param_grid = {
    'max_depth': 50,
    'learning_rate': 0.05,
    'n_estimators': 5000,
    'reg_alpha': 0.1,
    'reg_lambda': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'gamma': 0.1,
}

# Обучение модели XGBoost с GridSearchCV
xgb_clf = xgb.XGBClassifier(**param_grid)

xgb_clf.fit(X_train, y_train)



# Вывод результатов
print('Точность классификатора XGBoost на обучающих данных: {:.2f}'.format(xgb_clf.score(X_train, y_train) * 100))
print('Точность классификатора XGBoost на тестовых данных: {:.2f}'.format(xgb_clf.score(X_test, y_test) * 100))


# Графики

# Пары
sns.pairplot(data=df[df.columns[0:24]])
plt.title('Pair plot с учетом статуса пациентов')
plt.show()


# Тепловая карта
plt.figure(figsize=(15,8))
plt.title('Тепловая карта корреляции признаков')
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='cubehelix_r') 
plt.show()

# Гистограмма распределения
sns.histplot(df['MDVP:Fo(Hz)'], bins=30, kde=True)
plt.title('Гистограмма распределения MDVP:Fo(Hz)')
plt.show()

# График плотности
sns.kdeplot(df['MDVP:Jitter(%)'], shade=True)
plt.title('График плотности MDVP:Jitter(%)')
plt.show()

# Столбчатая диаграмма
sns.countplot(x='status', data=df)
plt.title('Столбчатая диаграмма по статусу пациентов')
plt.xlabel('Статус (0 - здоров, 1 - больной)')
plt.ylabel('Количество')
plt.show()


