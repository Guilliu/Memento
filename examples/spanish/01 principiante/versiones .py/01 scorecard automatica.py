# -------------------------------------------------------------------------------------------------
# Este script muestra cómo sacar un modelo automático aplicando la clase scorecard a un dataset
# -------------------------------------------------------------------------------------------------

# Importamos los módulos
import sys, pandas as pd, memento as me

# Cargamos los datos separando las variables predictoras 
# (guardadas en X) de la variable objetivo (guardada en y)
from sklearn.datasets import load_breast_cancer as lbc
X, y = pd.DataFrame(lbc().data, columns=lbc().feature_names), lbc().target

# Sustituimos los espacios en blanco por guiones bajos en el nombre de las columnas
X.columns = [i.replace(' ', '_') for i in X.columns]

# Todas las variables predictoras son numéricas, pero no habría ningún problema si hubiera
# variables de texto, booleanas, con o sin missings... Funcionaría sin necesidad de tratamiento
X.dtypes.unique()

# Aplicamos la clase scorecard para sacar el modelo automático
modelo = me.scorecard().fit(X, y)

# Visualizamos la scorecard: si usas ventana interactiva en VSC con blanco como color de tema
# puedes visualizarla con colores usando el método me.pretty_scorecard(), (debes tener instalado
# Jinja2, puedes con pip), si ejecutas en terminal la tabla resumen está en modelo.scorecard
print(modelo.scorecard) # Prueba me.pretty_scorecard(modelo) si no ejecutas en terminal

# Podemos guardar la scorecard en un excel con el que enseñar los resultados 
# y poder interpretar bien el sentido y la aportación de cada variable
modelo.create_excel('scorecard_ejemplo_01.xlsx')

