# -------------------------------------------------------------------------------------------------
# En este script se revela paso a paso el 'funcionamiento interno' de la clase scorecard en su
# modo automático.Usamos el mismo dataset del ejemplo 01 para ir comprobando los resultados.
# -------------------------------------------------------------------------------------------------

# Importamos los módulos
import sys, pandas as pd, memento as me

# Cargamos los datos
from sklearn.datasets import load_breast_cancer as lbc
X, y = pd.DataFrame(lbc().data, columns=lbc().feature_names), lbc().target

# Sustituimos los espacios en blanco por guiones bajos en el nombre de las columnas
X.columns = [i.replace(' ', '_') for i in X.columns]

# Lo primero que hace la clase 'scorecard' es generar una partición train-test.
# Por defecto hace un 70-30, estratificado en el target y con semilla
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=123, stratify=y)
X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)

# Después, aplica la clase 'autogrouping' a todas las variables y guarda los objetos resultantes 
# en un diccionario. Para generar estos buckets automáticos se usan árboles de decisión
variables, autogroupings = X_train.columns, {}
variables_no_agrupadas_error = []
for variable in variables:
    try:
        x = X_train[variable].values
        frenken = me.autogrouping().fit(x, y_train)
        autogroupings[variable] = frenken
    except: variables_no_agrupadas_error.append(variable)

# También genera un DataFrame ordenado por IV. No considerará a las variables 
# que tengan un IV inverior al umbral mínimo metodológico (0.015, modificable)
tabla_ivs, contador = pd.DataFrame(columns=['variable', 'iv']), 0
for variable in autogroupings:
    tabla_ivs.loc[contador] = variable, autogroupings[variable].iv
    contador += 1
tabla_ivs = tabla_ivs.sort_values('iv', ascending=False).reset_index(drop=True)
variables_filtro_iv = tabla_ivs[tabla_ivs['iv'] >= 0.015]['variable']
variables_def = list(set(variables_filtro_iv) - set(variables_no_agrupadas_error))

# Vemos cuáles son las variables con mayor IV (= Information Value). 
# Podríamos probar a generar una scorecard seleccionando las n variables de mayor IV
tabla_ivs.head(3)

# Por ejemplo, veamos cómo sería la scorecard con las tres primeras variables por IV: 
# 'worst_concave_points', 'worst_radius', 'worst_perimeter'. 
# En features pondríamos las variables que queremos que formen parte de la scorecard
features = ['worst_concave_points', 'worst_radius', 'worst_perimeter']

# Si quisiéramos usar agrupaciones manuales las pondríamos dentro del 
# diccionario user_breakpoints, por ahora lo dejamos vacío para usar las automáticas
user_breakpoints = {}

# En final_breakpoints se actualizarían los grupos en caso
# de haber introducido agrupaciones manuales en user_breakpoints
final_breakpoints = me.compute_final_breakpoints(variables_def, autogroupings, user_breakpoints)

# Antes de calcular el modelo, la clase scorecard aplica un tratamiento a las columnas
info = me.compute_info(X_train, variables_def, final_breakpoints)
df_train = me.adapt_data(X_train, y_train, variables_def, final_breakpoints, 'target')

# Calculamos ya la scorecard, pero solo como tarjeta de puntuación 
# en forma de DataFrame, el objeto tipo modelo se obtiene usando clase scorecard

scorecard, features_length = me.compute_scorecard(df_train, features, info, 'target')

# Podemos aplicar esta tarjeta de puntuación al data de entrenamiento y ver las métricas asociadas
df_train_final = me.apply_scorecard(df_train, scorecard, info, 'target')
ks_train, gini_train = me.compute_metrics(df_train_final, 'target', ['gini', 'ks'], True)

# Hacemos lo mismo con los datos del test (la validación del 30%, habitualmente llamada hold out)
df_test = me.adapt_data(X_test, y_test, variables_def, final_breakpoints, 'target')
df_test_final = me.apply_scorecard(df_test, scorecard, info, 'target')
ks_test, gini_test = me.compute_metrics(df_test_final, 'target', ['gini', 'ks'], True)

# No ha salido un mal modelo, también porque este ejemplo es MUY de juguete... 
# La clase scorecard hace todo igual que hasta ahora salvo la selección de variables: 
# no elige las variables en función de su IV sino que se utilizan una de estas dos aproximaciones.
# (Disclaimer: esto es algo muy teórico, no es necesario comprenderlo perfectamente ni mucho menos)
    
# 1) Método 'forward' con métrica de 'ks' o 'gini':  En el primer paso se parte del modelo vacío,
# es decir el modelo que no tiene ninguna variable. Entonces para cada variable candidata (aquellas
# agrupadas con un IV >= 0.015) se genera un modelo que tienen a esa variable como única variable
# del modelo, así de entre todos estos modelos univariable se mira cual sería el que da un mayor
# valor de la métrica (ks o gini) en el data  del train. Aquella variable cuyo modelo de el MÁXIMO
# valor de la métrica se selecciona. En el segundo paso se parte del conjunto que ya tiene la
# primera variable seleccionada del paso anterior. Ahora, se consideran TODAS las demás variables 
# candidatas para generar TODOS los modelos de dos variables distintos posibles donde la primera es
# la que ya habíamos seleccionado en el paso anterior. Bien, pues de entre todos estos modelos 
# 2-variables aquel que de el MÁXIMO valor de la métrica en el train nos indica cual es la segunda 
# variable que seleccionamos: la que se ha añadido a la que ya teníamos para generar este modelo. 
# Note que, si el número de variables candidatas es n, entonces en el paso 1 se consideran n 
# modelos 1-variable y en el paso 2 se considerarían (n-1) modelos 2-variables, sin embargo, 
# estos modelos 2-variables son más costosos desde el punto de vista computacional por tener una 
# variable más y esto hace que los tiempos vayan aumentando en cada paso. El proceso se detiene 
# cuando se alcanza el máximo número de pasos permitidos o cuando la métrica no mejora más de 
# un umbral (0.20 para KS y 0.30 para Gini) la del paso anterior.
    
# 2) Método 'stepwise' con métrica de 'p-valor' (configuración por defecto de la clase scorecard): 
# Se realiza un forward como el anteriormente descrito, pero con algunas modificaciones. 
# Se empieza buscando la variable que genera el modelo 1-variable en donde esta variable tiene el 
# p-valor más bajo (los p-valores se calculan a nivel variable, no a nivel modelo). 
# En un segundo paso se consideran todos los modelos 2-variables donde la primera variable es la 
# elegida en el paso anterior y la segunda es cualquiera de las candidatas restantes. Bien, pues 
# en cada uno de  estos modelos se CALCULAN los p-valores de las dos variables y se elige el que 
# tenga el p-valor  de la variable que se está probando más bajo. La gracia ahora es que aquí en 
# cada paso se RECALCULAN los p-valores de TODAS las variables  involucradas en el modelo de turno, 
# no solo el de la variable candidata a entrar si no también los p-valores del resto de variables 
# ya seleccionadas de forma que si en algún momento alguno de ellos es superior a 0.01 (nivel de 
# significancia metodológico) entonces esta variable SALE  del modelo. Esto ocurrirá cuando la 
# variable que acaba de entrar está ciertamente correlada con  la que está saliendo pero por algún 
# motivo la que entra se combina mejor con el resto de variables del modelo y tiene una aportación 
# 'mayor' al modelo (estadísticamente hablando). El proceso se detiene cuando se alcanza el máximo 
# número de pasos permitidos o cuando ninguna de las variables retadas tiene un p-valor inferior 
# a 0.01 o también cuando la métrica no mejora más de un umbral (0.20 para KS y 0.30 para Gini).

# La función 'feature_selection' realiza este proceso, la usamos con sus 
# valores por defecto para obtener las mismas variables que en el ejemplo 01
features = me.features_selection(df_train, [], variables_def, info, 'target')

# Ahora sí la scorecard sale igual que ene ejemplo 01
scorecard, features_length = me.compute_scorecard(df_train, features, info, 'target')
df_train_final = me.apply_scorecard(df_train, scorecard, info, 'target')
df_test_final = me.apply_scorecard(df_test, scorecard, info, 'target')
ks_train, gini_train = me.compute_metrics(df_train_final, 'target', ['gini', 'ks'], True)
print('-'*80)
ks_test, gini_test = me.compute_metrics(df_test_final, 'target', ['gini', 'ks'], True)

