import numpy as np, pandas as pd

####################################################################################################


def string_categories1(x, y):

    return dict(map(reversed, enumerate(pd.Series(y).groupby(
    x, dropna=False).mean().sort_values(ascending=False).index.values)))


def string_categories2(breakpoints_cat):

    try: return dict(map(reversed, enumerate([i for j in breakpoints_cat for i in j])))
    except: return {}


def string_to_num(x, categories):

    return pd.Series(x).map(categories).values


def breakpoints_to_str(breakpoints_num, categories):
 
    breakpoints_str = []

    for i in range(len(breakpoints_num)):
        if i == 0:
            breakpoints_str.append([j[0] for j in categories.items()
            if j[1] < breakpoints_num[i]])
        else:
            breakpoints_str.append([j[0] for j in categories.items()
            if breakpoints_num[i-1] <= j[1] < breakpoints_num[i]])
        if i == len(breakpoints_num) - 1:
            breakpoints_str.append([j[0] for j in categories.items()
            if breakpoints_num[i] <= j[1]])

    return breakpoints_str


def breakpoints_to_num(breakpoints_cat):

    if isinstance(breakpoints_cat[0], list):

        L, suma = [], 0
        for i in breakpoints_cat[:-1]:
            suma += len(i)
            L.append(suma-0.5)
        return np.array(L)

    else: return breakpoints_cat


def remapeo_missing(v, bp, old_value=-12345678):

    if isinstance(bp, dict):

        breakpoints = bp['bp']
        missing_group = bp['mg']

        if missing_group!= 0:
            if missing_group == 1:
                return np.where(v == old_value, breakpoints[missing_group-1]-(np.e-2), v)
            if missing_group >= 2:
                return np.where(v == old_value, breakpoints[missing_group-2]+(np.e-2), v)
        else: return v
    else: return v


def data_convert(x, categories):

    x_original = x
    
    if x.dtype in ('O', 'bool'):
        if categories == {}:
            raise ValueError('En una variable de tipo texto o booleana '
            'es necesario especificar el diccionario de categorias')
        if pd.Series(x).isna().sum() > 0:
            x = pd.Series(x).replace(np.nan, 'Missing').values
        x_initial = x
        x = string_to_num(x, categories)
        x_converted = x
    
    else:
        x_initial = x
        x_converted = x

    if x.dtype not in ('O', 'bool') and np.isnan(x).sum() > 0:
        x_final = np.nan_to_num(x, nan=-12345678)

    else: x_final = x_converted

    return x_original, x_initial, x_converted, x_final


def adapt_data(X, y, variables, breakpoints, target_name='target_4815162342'):
    
    df = pd.DataFrame()
    for variable in variables:
        
            bp = breakpoints[variable]
            
            if not isinstance(bp, dict):
                df[variable] = data_convert(X[variable].values, string_categories2(bp))[3]
                
            else:
                df[variable] = remapeo_missing(data_convert(
                X[variable].values, string_categories2(bp))[3], bp)
        
    df[target_name] = y

    return df

