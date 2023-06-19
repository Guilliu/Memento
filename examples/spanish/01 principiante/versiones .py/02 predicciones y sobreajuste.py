# -------------------------------------------------------------------------------------------------
# En este script vemos cómo usar los 'modelos entrenados' que produce 
# la clase Scorecard para hacer predicciones en dos datasets de Kaggle 😎
# -------------------------------------------------------------------------------------------------

# Importamos los módulos
import sys, numpy as np, pandas as pd, memento as me

# Cargamos el primer dataset, del Titanic
df_train = pd.read_csv('train titanic.csv')
L = list(me.proc_freq(df_train, 'Survived').frequency)
print('Sobrevivieron {}. Murieron {} (entre ellos Leonardo DiCaprio)'.format(L[1], L[0]))
X, y = df_train.drop('Survived', axis=1), df_train.Survived.values

# Sacamos una primera scorecard excluyendo las variables 'PassengerId' y 'Name'
modelo_titanic1 = me.Scorecard(excluded_vars=['PassengerId', 'Name']).fit(X, y)

# Tremendo batacazo del train al test... Esto es un claro ejemplo de sobreajuste, de hecho,
# el propio log del modelo nos avisa de que cuidado con 'Ticket'. Vemos que tiene sentido:
# es una variable con un iv desmesuradamente alto y con muchísima granularidad
modelo_titanic1.tabla_ivs.head(3)
print('Valores distintos variable Ticket: {}'.format(len(X['Ticket'].value_counts())))

# Sacamos una segunda versión del modelo excluyendo esta variable 'Ticket'
modelo_titanic2 = me.Scorecard(excluded_vars=['PassengerId', 'Name', 'Ticket']).fit(X, y)

# Mucho mejor, sin embargo, seguimos teniendo cierto desplome
print('Caida de Gini de train a test: {:.2f}%'.format(
100*(modelo_titanic2.gini_train - modelo_titanic2.gini_test) / modelo_titanic2.gini_train))

# Vemos que la diferencia entre tain y test se acentúa tras añadir la variable 'Cabin'...
# Sacamos una tercera versión excluyendo también a 'Cabin'
modelo_titanic3 = me.Scorecard(excluded_vars=['PassengerId', 'Name', 'Ticket', 'Cabin']).fit(X, y)

# Este último modelo solo tiene 3 variables.. Bueno y qué pasa?! '#minimalism' 😊
print(modelo_titanic3.scorecard)

# Vamos a usar estos modelos para hacer predicciones 
# y subir las submissions resultantes a Kaggle
df_test = pd.read_csv('test titanic.csv')

# El método '.predict()' es el responsable de aplicar el modelo a un nuevo dataframe.
# La variable que contiene la puntuación final es 'scorecardpoints', esta puntuación está calibrada
# con PEO (points equal odds, donde la tasa de malos es 1/2) en 500 puntos y un PDO (points to 
# double odds) de 20 puntos. También se genera, para cada variable del modelo final, una columna
# con la puntuación parcial que aporta dicha variable (las columnas 'scr_'). Por último,
# podemos darle un umbral para sacar la columna 'prediction' con una predicción binaria
modelo_titanic1.predict(df_test, keep_columns=['PassengerId'], binary_treshold=500)

# Generamos las submissions como archivos .csv para subirlas a Kaggle
modelo_titanic1.predict(df_test, keep_columns=['PassengerId'], binary_treshold=500)\
.rename(columns={'prediction': 'Survived'})[['PassengerId', 'Survived']]\
.to_csv('submission_titanic1.csv', index=False) # Score en Kaggle = 0.44258

modelo_titanic2.predict(df_test, keep_columns=['PassengerId'], binary_treshold=500)\
.rename(columns={'prediction': 'Survived'})[['PassengerId', 'Survived']]\
.to_csv('submission_titanic2.csv', index=False) # Score en Kaggle = 0.72727

modelo_titanic3.predict(df_test, keep_columns=['PassengerId'], binary_treshold=500)\
.rename(columns={'prediction': 'Survived'})[['PassengerId', 'Survived']]\
.to_csv('submission_titanic3.csv', index=False) # Score en Kaggle = 0.77272

# Vamos ahora con otro dataframe de Kaggle similar: Spaceship Titanic
df_train = pd.read_csv('train dimension.csv')
df_train['Transported'] = np.where(df_train['Transported'] == True, 1, 0)
L = list(me.proc_freq(df_train, 'Transported').frequency)
print('Trasportados: {}. No trasportados: {}'.format(L[1], L[0]))
X, y = df_train.drop('Transported', axis=1), df_train.Transported.values

# Tiramos un primer modelo excluyendo las variables 'PassengerId' y 'Name'
modelo_spaceship1 = me.Scorecard(excluded_vars=['PassengerId', 'Name']).fit(X, y)

# Hacemos tres observaciones:
# 1) La primera variable que está entrando, 'CryoSleep', es booleana, no hay ningún problema
me.proc_freq(X, 'CryoSleep')

# 2) Nos avisa de que la variable 'VIP' no ha podido agruparse por excesiva concentración y
# resulta que efectivamente hay un valor que se repite en más del 95% de los casos (podría
# ser que no fuera superior al 95% en la muestra total y si lo fuera la submuestra del train).
# Si quisiéramos que el modelo intentará incluir esta variable habría que pedir más flexibilidad
# en el autogrouping rebajando el umbral mínimo de porcentaje de población por grupo,
# por ejemplo pasando como argumento a scorecard autogrp_dict_min_pct={'VIP': 0.02})
me.proc_freq(X, 'VIP')

# 3) Al igual que antes nos avisa de posible sobreajuste por 'Cabin', probamos a quitarla
modelo_spaceship2 = me.Scorecard(excluded_vars=['PassengerId', 'Name', 'Cabin']).fit(X, y)

# En esta segunda versión ya no teneos sobreajuste ninguno... 
# Sin embargo, podemos echar un vistazo a los p-valores a ver
modelo_spaceship2.pvalues

# Claramente, 'CryoSleep' y 'Destination' tienen unos pvalores que, si bien sí son superiores 
# al umbral por defecto de 0.01 (porque si no habrían salido en algún paso del stepwise), son muy
# inferiores a la media del resto de features...<br>Podemos mirar si prescindiendo de alguna de
# ellas (o de ambas) obtenemos un modelo con métricas similares tanto en train como en test
features1 = [i for i in modelo_spaceship2.features if i != 'CryoSleep']
# features2 = [i for i in modelo_spaceship2.features if i != 'Destination']
# features3 = [i for i in modelo_spaceship2.features if i != 'CryoSleep' and i!= 'Destination']

modelo_spaceship2_alt1 = me.Scorecard(features=features1).fit(X, y)
# modelo_spaceship2_alt2 = me.Scorecard(features=features2).fit(X, y)
# modelo_spaceship2_alt3 = me.Scorecard(features=features3).fit(X, y)

# En las tres pruebas obtenemos Ginis muy similares... Dejamos como opción 3 la que elimina solo
# al 'CryoSleep' ya que da un Gini ligeramente superior tanto en train como en test respecto a lo
# que teníamos. Al final cuánto más sencillo sea un modelo, mejor. '#make_it_simple'
modelo_spaceship3 = modelo_spaceship2_alt1

# Hacemos las predicciones y subimos los resultados a Kaggle 🚀
df_test = pd.read_csv('test dimension.csv')

prediction1 = modelo_spaceship1.predict(df_test, keep_columns=['PassengerId'], binary_treshold=500)\
.rename(columns={'prediction': 'Transported'})[['PassengerId', 'Transported']]
prediction1['Transported'] = np.where(prediction1['Transported'] == 1, True, False)
prediction1.to_csv('submission_spaceship1.csv', index=False) # Score en Kaggle = 0.51718

prediction2 = modelo_spaceship2.predict(df_test, keep_columns=['PassengerId'], binary_treshold=500)\
.rename(columns={'prediction': 'Transported'})[['PassengerId', 'Transported']]
prediction2['Transported'] = np.where(prediction2['Transported'] == 1, True, False)
prediction2.to_csv('submission_spaceship2.csv', index=False) # Score en Kaggle = 0.77788

prediction3 = modelo_spaceship3.predict(df_test, keep_columns=['PassengerId'], binary_treshold=500)\
.rename(columns={'prediction': 'Transported'})[['PassengerId', 'Transported']]
prediction3['Transported'] = np.where(prediction3['Transported'] == 1, True, False)
prediction3.to_csv('submission_spaceship3.csv', index=False) # Score en Kaggle = 0.77788

# Mostramos la scorecard del último modelo, que desempeña igual que el anterior y es más sencillo
print(modelo_spaceship3.scorecard)

