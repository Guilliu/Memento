# -------------------------------------------------------------------------------------------------
# En este script vemos como hacer 'agrupaciones manuales' de manera interactiva, algo muy útil
# para desarrollos más expertos. También mostramos como se manejan internamente los 'missings'.
# -------------------------------------------------------------------------------------------------

# Importamos los módulos
import sys, numpy as np, pandas as pd, memento as me

# Cargamos los datos
data = pd.read_csv('stroke_data.csv')
X, y = data.drop('stroke', axis=1), data['stroke']
print('El dataset tiene {} filas y {} columnas'.format(X.shape[0], X.shape[1]))

# La variable numérica 'bmi' es la única que tiene missings
X.isna().sum()

# Metemos missings también la variable de texto 'work_type' quitando el valor 'Self-employed'
X['work_type'] = X['work_type'].replace('Self-employed', np.nan)
X.isna().sum()

# Sacamos el modelo automático excluyendo la variable 'id' por motivos evidentes
modelo1 = me.Scorecard(excluded_vars=['id']).fit(X, y)

# ¿Y si no quiero usar las agrupaciones del autogrouping? ¿Y si quiero modificarlas o directamente 
# usar las que a mí me de la gana?# Lo ideal es usar la función 'reagrupa_var', podemos llamarla
# solo pasándole el modelo y el nombre de la variable y así vemos cuales son los puntos de corte
# que generan la agrupación automática y podemos modificar estos puntos de corte en el argumento
# new_bp para ver como quedaría reagrupada
# me.reagrupa_var(modelo1, 'age')
me.reagrupa_var(modelo1, 'age', [30, 60])

# Para usar esta nueva agrupación de la variable 'age' lanzamos de nuevo una scorecard con la
# agrupación en el diccionario 'user_breakpoints'
modelo2 = me.Scorecard(excluded_vars=['id'], user_breakpoints={'age': [30, 60]}).fit(X, y)

# Observación Ahora está entrando también la variable 'hypertension', cosa que antes no pasaba...
# Esto ocurre porque con la nueva agrupación 'age' es aparentemente menos discriminante: ahora
# en el primer paso el modelo tiene un 56.54% de gini en train cuando antes, con la agrupación
# automática, en el primer paso el modelo tenía un 64.73%. Por este motivo ahora al final el
# método de selección de variables acaba escogiendo también a 'hypertension'. Si se quisiera
# evitar esto, se puede introducir las variables exactas que queremos formen parte de la scorecard
# en el parámetro 'features' y así comparar mejor el impacto de la agrupación manual
modelo3 = me.Scorecard(
    features=['age', 'bmi', 'avg_glucose_level'],
    user_breakpoints={'age': [30, 60]}
).fit(X, y)


# Vemos como estaría quedando la scorecard
print(modelo3.scorecard)

# Vemos que también entró una de las dos variables con missings: el 'bmi'. De hecho, ha puesto en
# un grupo a parte a estos missings y esto no es casualidad: En el autogrouping de una variable
# numérica con missings siempre se le dará un grupo aparte para estos missings. Siempre y cuando
# haya al menos un malo y un bueno, independientemente de su volumen (si te fijas en este caso ese
# grupo ni si quiera llega al 5% mínimo que se suele exigir, da igual). Esto se hace con una 
# asignación inicial de los missings al valor -12345678, que entendemos va a ser el mínimo de esa 
# variable de forma que con un corte inmediatamente posterior nos garantizamos que estos missing 
# están en un grupo aparte.
    
# Ok, pero... Y si quiero juntar esos missings con otro grupo... ¿Cómo lo hago? Se pasa un 
# diccionario indicando por un lado los puntos de corte (pudiendo ser los mismos del autogrouping
# o no) eliminando de ellos el valor -12345670.0 si se quiere juntar a los missings con otro grupo 
# y por otro lado indicando a qué grupo se desea mandar los missings
me.reagrupa_var(modelo1, 'bmi', {'bp': [20, 30], 'mg': 2})

# Vamos a lanzar otra scorecard con esta agrupación en el 'bmi'. 
# Dado que esta es peor que la automática debería salir una scorecard con menos Gini
modelo4 = me.Scorecard(
    features=['age', 'bmi', 'avg_glucose_level'],
    user_breakpoints={
        'age': [30, 60],
        'bmi': {'bp': [20, 30], 'mg': 2}
    }
).fit(X, y)

# Echamos un ojo a como quedaría la scorecard
print(modelo4.scorecard)

# ¿Y si la variable que tiene missings es de tipo texto? Mucho más fácil: En una variable de texto
# el missing se trata como una categoría más, no hay distinción con el resto de categorías. 
# Vamos a verlo con la variable 'worktype' a la que metimos missings artificialmente
modelo5 = me.Scorecard(
    features=['age', 'bmi', 'avg_glucose_level', 'work_type'],
    user_breakpoints={
        'age': [30, 60],
        'bmi': {'bp': [20, 30], 'mg': 2}
    }
).fit(X, y)

# Reagrupamos con una lista normal, independientemente de si la variable tiene missings o no
me.reagrupa_var(modelo5, 'work_type',
[['Private', 'Govt_job'], ['Missing', 'Never_worked'], ['children']])

# Lanzamos la última scorecard con todas las reagrupaciones que hemos hecho
modelo6 = me.Scorecard(
    features=['age', 'bmi', 'avg_glucose_level', 'work_type'],
    user_breakpoints={
        'age': [30, 60],
        'bmi': {'bp': [20, 30], 'mg': 2},
        'work_type': [['Private', 'Govt_job'], ['Missing', 'Never_worked'], ['children']]
    }
).fit(X, y)


print(modelo6.scorecard)

