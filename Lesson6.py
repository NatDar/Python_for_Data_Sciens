#!/usr/bin/env python
# coding: utf-8

# Задание 1
# Импортируйте библиотеки pandas и numpy.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn. Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 30% от всех данных, при этом аргумент random_state должен быть равен 42.
# Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля sklearn.linear_model.
# Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.
# Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.
# 

# In[33]:


import numpy as np


# In[4]:


import pandas as pd


# In[3]:


from sklearn.datasets import load_boston


# In[4]:


boston = load_boston()
data = boston["data"]


# In[5]:


feature_names = boston["feature_names"]

X = pd.DataFrame(data, columns=feature_names)
X.head()


# In[6]:


target = boston["target"]

Y = pd.DataFrame(target, columns=["price"])
Y.head()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


lr = LinearRegression()


# In[11]:


lr.fit(X_train, Y_train)


# In[12]:


y_pred_lr = lr.predict(X_test)
check_test_lr = pd.DataFrame({
    "Y_test": Y_test["price"], 
    "Y_pred_lr": y_pred_lr.flatten()})

check_test_lr.head()


# In[13]:


from sklearn.metrics import mean_squared_error

mean_squared_error_lr = mean_squared_error(check_test_lr["Y_pred_lr"], check_test_lr["Y_test"])
print(mean_squared_error_lr)


# Задание 2
# Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble.
# 
# Сделайте агрумент n_estimators равным 1000, max_depth должен быть равен 12 и random_state сделайте равным 42.
# 
# Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression, но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0], чтобы получить из датафрейма одномерный массив Numpy, так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо датафрейма.
# 
# Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания.
# 
# Напишите в комментариях к коду, какая модель в данном случае работает лучше.

# In[14]:


from sklearn.ensemble import RandomForestRegressor


# In[15]:


clf = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)


# In[16]:


clf.fit(X_train, Y_train.values[:, 0])


# In[17]:


y_pred_clf = clf.predict(X_test)
check_test_clf = pd.DataFrame({
    "Y_test": Y_test["price"], 
    "Y_pred_clf": y_pred_clf.flatten()})

check_test_clf.head()


# In[18]:


mean_squared_error_clf = mean_squared_error(check_test_clf["Y_pred_clf"], check_test_clf["Y_test"])
print(mean_squared_error_clf)


# In[19]:


print(mean_squared_error_lr, mean_squared_error_clf)


# Алгоритм "Случайный лес" показывает более точные результаты, чем "линейная регрессия". Примерно в 3 раза.

# In[ ]:


Задание 3
Вызовите документацию для класса , найдите информацию об атрибуте featureimportances.

С помощью этого атрибута найдите сумму всех показателей важности, установите, какие два признака показывают наибольшую важность.


# In[20]:


print(clf.feature_importances_)


# In[21]:


feature_importance = pd.DataFrame({'name':X.columns, 
                                   'feature_importance':clf.feature_importances_}, 
                                  columns=['feature_importance', 'name'])
feature_importance


# In[22]:


feature_importance.nlargest(2, 'feature_importance')


# Признаки LSTAT и RM обладают наибольшей важностью.

# In[ ]:


Задание 4
В этом задании мы будем работать с датасетом, с которым мы уже знакомы по домашнему заданию по библиотеке Matplotlib, это датасет Credit Card Fraud Detection.

Для этого датасета мы будем решать задачу классификации - будем определять, какие из транзакциции по кредитной карте являются мошенническими.

Данный датасет сильно несбалансирован (так как случаи мошенничества относительно редки), так что применение метрики accuracy не принесет пользы и не поможет выбрать лучшую модель.

Мы будем вычислять AUC, то есть площадь под кривой ROC.

Импортируйте из соответствующих модулей RandomForestClassifier, GridSearchCV и train_test_split.

Загрузите датасет creditcard.csv и создайте датафрейм df.

С помощью метода value_counts с аргументом normalize=True убедитесь в том, что выборка несбалансирована.

Используя метод info, проверьте, все ли столбцы содержат числовые данные и нет ли в них пропусков.

Примените следующую настройку, чтобы можно было просматривать все столбцы датафрейма:

pd.options.display.max_columns = 100.

Просмотрите первые 10 строк датафрейма df.

Создайте датафрейм X из датафрейма df, исключив столбец Class.

Создайте объект Series под названием y из столбца Class.

Разбейте X и y на тренировочный и тестовый наборы данных при помощи функции train_test_split, используя аргументы: test_size=0.3, random_state=100, stratify=y.

У вас должны получиться объекты X_train, X_test, y_train и y_test.

Просмотрите информацию о их форме.

Для поиска по сетке параметров задайте такие параметры:

parameters = [{'n_estimators': [10, 15],

'max_features': np.arange(3, 5),

'max_depth': np.arange(4, 7)}]

Создайте модель GridSearchCV со следующими аргументами:

estimator=RandomForestClassifier(random_state=100),

param_grid=parameters,

scoring='roc_auc',

cv=3.

Обучите модель на тренировочном наборе данных (может занять несколько минут).

Просмотрите параметры лучшей модели с помощью атрибута bestparams.

Предскажите вероятности классов с помощью полученнной модели и метода predict_proba.

Из полученного результата (массив Numpy) выберите столбец с индексом 1 (вероятность класса 1) и запишите в массив y_pred_proba.

Из модуля sklearn.metrics импортируйте метрику roc_auc_score.

Вычислите AUC на тестовых данных и сравните с результатом, полученным на тренировочных данных, используя в качестве аргументов массивы y_test и y_pred_proba.


# In[21]:


df = pd.read_csv('creditcard.csv')


# In[22]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[23]:


class_list = creditcard['Class'].value_counts()
print(class_list)


# In[24]:


df['Class'].value_counts(normalize=True)


# In[25]:


df.info()


# In[26]:


pd.options.display.max_columns=100


# In[27]:


df.head(10)


# In[28]:


X = df.drop('Class', axis=1)


# In[29]:


y = df['Class']


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)


# In[31]:


print('X_train ', X_train.shape)
print('X_test ', X_test.shape)
print('y_train ', y_train.shape)
print('y_test ', y_test.shape)


# In[53]:


parameters = [{'n_estimators': [10, 15], 
    'max_features': np.arange(3, 5), 
    'max_depth': np.arange(4, 7)}] 
clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=100),
    param_grid=parameters,
    scoring='roc_auc',
    cv=3,)
clf.fit(X_train, y_train)


# In[47]:


clf.best_params_


# In[48]:


clf = RandomForestClassifier(max_depth=6, max_features=3, n_estimators=15)

clf.fit(X_train, y_train)


# In[49]:


y_pred = clf.predict_proba(X_test)


# In[50]:


y_pred_proba = y_pred[:, 1]


# In[51]:


from sklearn.metrics import roc_auc_score


# In[52]:


roc_auc_score(y_test, y_pred_proba)


# ## Дополнительные задания

# 1. Загрузите датасет Wine из встроенных датасетов sklearn.datasets с помощью функции load_wine в переменную data.

# In[54]:


from sklearn.datasets import load_wine
data = load_wine()


#  2. Полученный датасет не является датафреймом. Это структура данных, имеющая ключи аналогично словарю. Просмотрите тип данных этой структуры данных и создайте список data_keys, содержащий ее ключи.
# 

# In[55]:


data_keys = data.keys()
print(data_keys)


#  3. Просмотрите данные, описание и названия признаков в датасете. Описание нужно вывести в виде привычного, аккуратно оформленного текста, без обозначений переноса строки, но с самими переносами и т.д

# In[56]:


data.data


# In[57]:


print(data.DESCR)


# In[58]:


data.feature_names


# 4. Сколько классов содержит целевая переменная датасета? Выведите названия классов.
# 

# In[59]:


print(set(data.target))
print(len(set(data.target)))


# In[60]:


data.target_names


# 5. На основе данных датасета (они содержатся в двумерном массиве Numpy) и названий признаков создайте датафрейм под названием X.
# 

# In[61]:


X = pd.DataFrame(data.data, columns=data.feature_names)
X.head()


# 6. Выясните размер датафрейма X и установите, имеются ли в нем пропущенные значения.
# 

# In[62]:


X.shape


# In[63]:


X.info()


# 7. Добавьте в датафрейм поле с классами вин в виде чисел, имеющих тип данных numpy.int64. Название поля - 'target'.
# 

# In[76]:


X['target'] = data.target


# In[77]:


X.head()


# 8. Постройте матрицу корреляций для всех полей X. Дайте полученному датафрейму название X_corr.
# 

# In[78]:


X_corr = X.corr()
X_corr


# 9. Создайте список high_corr из признаков, корреляция которых с полем target по абсолютному значению превышает 0.5 (причем, само поле target не должно входить в этот список).
# 

# In[79]:


high_corr = X_corr.loc[(abs(X_corr['target']) > 0.5) & (X_corr.index != 'target'), X_corr.columns != 'target'].index
high_corr


# 10. Удалите из датафрейма X поле с целевой переменной. Для всех признаков, названия которых содержатся в списке high_corr, вычислите квадрат их значений и добавьте в датафрейм X соответствующие поля с суффиксом '_2', добавленного к первоначальному названию признака. Итоговый датафрейм должен содержать все поля, которые, были в нем изначально, а также поля с признаками из списка high_corr, возведенными в квадрат. Выведите описание полей датафрейма X с помощью метода describe.
# 

# In[80]:


X = X.drop('target', axis=1)
X.head()


# In[82]:


for feature_name in high_corr:
    X[f'{feature_name}_2'] = X.apply(lambda row: row[feature_name] ** 2, axis=1)


# In[83]:


X.describe()


# In[ ]:




