# -------------------------------------------------------------------------------------------------
# Vemos como desarrollar un modelo usando el 'parceling' para la inferencia de denegados
# -------------------------------------------------------------------------------------------------

# Importamos los módulos
import sys, numpy as np, pandas as pd, memento as me

# Cargamos los datos
df = pd.read_csv('hmeq.csv')

# Ponemos las columnas en minúsculas, renombramos el target a 'target_original' y añadimos un 'id'
df.columns = ['target_original'] + [col.lower() for col in df.columns[1:]]
df.insert(0, 'id', [str(i).zfill(4) for i in range(1, len(df)+1)])

# Generamos los denegados aleatoriamente (esto no debería ser así porque en general los denegados 
# tienen un peor perfil... pero bueno es un ejemplo) marcando como denegados al 25% de la muestra
mask_rejected = np.array([True]*round(len(df)*0.75)+[False]*(len(df)-round(len(df)*0.75)))
np.random.seed(123) # Importante fijar semilla para que sea replicable
np.random.shuffle(mask_rejected)
df['decision'] = np.where(mask_rejected, 'aprobado', 'denegado')
df['target'] = np.where(mask_rejected, df['target_original'], -3)
df.head()

# Vemos la distribución de denegados, buenos y malos que tenemos
me.proc_freq(df, 'decision', 'target')

# Lo primero es sacar una scorecard solo con aceptados
df_aceptados = df[df.decision == 'aprobado']
X, y = df_aceptados.drop('target', axis=1), df_aceptados.target.values
modelo_aceptados = me.scorecard(excluded_vars=['id', 'target_original', 'decision']).fit(X, y)

# Con esta scorecard de aceptados vamos a inferir cual hubiera sido el target de los denegados
prediction = modelo_aceptados.predict(df, keep_columns=['id'])[['id', 'scorecardpoints']]
df2 = df.merge(prediction.rename(columns={'scorecardpoints': 'scorecardpoints_acep'}), 'left', 'id')

# Aplicamos el parceling
df3, c = me.parceling(df2)

# Teniendo los denegados ya un target inferido desarrollamos otra scorecard 
# con una nueva partición 70-30 (usando todo: aceptados + denegados)
X_def, y_def = df3[X.columns], df3.target_def
modelo_def = me.scorecard(
    excluded_vars=['id', 'target_original', 'decision'], save_tables='all'
).fit(X_def, y_def)

# Evaluamos el modelo también solo sobre los aceptados (en el 70-30 del último modelo)
data_train = modelo_def.X_train.copy()
data_train['target'] = modelo_def.y_train
data_train_oa = data_train[data_train.decision == 'aprobado'].reset_index(drop=True)
data_train_final_oa = modelo_def.predict(data_train_oa, target_name='target')
ks_train, gini_train = me.compute_metrics(data_train_final_oa, 'target', ['gini', 'ks'], True)
print('-'*80)
data_test = modelo_def.X_test.copy()
data_test['target'] = modelo_def.y_test
data_test_oa = data_test[data_test.decision == 'aprobado'].reset_index(drop=True)
data_test_final_oa = modelo_def.predict(data_test_oa, target_name='target')
ks_test, gini_test = me.compute_metrics(data_test_final_oa, 'target', ['gini', 'ks'], True)

# Mostramos la scorecard final
print(modelo_def.scorecard)

