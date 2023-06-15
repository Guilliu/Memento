import pandas as pd, datetime

from .todd import *
from .diane import *

####################################################################################################


N = 120

def compute_final_breakpoints(variables, autogroupings, user_breakpoints):

    final_breakpoints = {}

    for variable in variables:
        try: final_breakpoints[variable] = user_breakpoints[variable]
        except: final_breakpoints[variable] = autogroupings[variable].breakpoints

    return final_breakpoints


def compute_info(X, variables, breakpoints):
    
    info = {}
    for variable in variables:
        
            info[variable] = {}
            bp = breakpoints[variable]
            
            if not isinstance(bp, dict):
                info[variable]['breakpoints_num'] = breakpoints_to_num(bp)
                info[variable]['group_names'] = compute_group_names(X[variable].values.dtype, bp)
                
            else:
                info[variable]['breakpoints_num'] = breakpoints_to_num(bp['bp'])
                info[variable]['group_names'] = compute_group_names(
                X[variable].values.dtype, bp['bp'], bp['mg'])
                
    return info


def features_selection(data, features, var_list, info, target_name='target_4815162342',
method='stepwise', metric='pvalue', threshold=0.01, stop_ks_gini=True, 
max_iters=14, included_vars=[], muestra_test=None, show='gini',
check_overfitting=True, logistic_method='newton'):
    
    # if log_file: file_prints = open('log_modelo.txt', 'a')
    # else: file_prints = None

    if features != []: included_vars, max_iters = features, 0

    if method not in ('forward', 'stepwise'):
        raise ValueError("Valor inválido para el parámetro 'method', "
        "solo están pertimidos los valores 'forward' y 'stepwise'")

    if metric not in ('pvalue', 'ks', 'gini'):
        raise ValueError("Valor inválido para el parámetro 'metric', "
        "solo están pertimidos los valores 'pvalue', 'ks' y 'gini")


    if max_iters > len(var_list):
        print('Cuidado, has puesto un valor numero máximo de iteraciones ({})'
        ' superior al número de variables candidatas ({})'.format(max_iters, len(var_list)))
        print('-' * N)
        max_iters = len(var_list)
    
    if check_overfitting:
        
        posible_sobreajuste = []
        for var in var_list:
            
            try:
                scorecard, features_length = compute_scorecard(
                data, [var], info, target_name=target_name, logistic_method=logistic_method)
                data_final = apply_scorecard(data, scorecard, info, target_name)
                gini_train = compute_metrics(data_final, target_name, ['gini'])
                if not isinstance(muestra_test, type(None)):
                    test_final = apply_scorecard(muestra_test, scorecard, info, target_name)
                    gini_test = compute_metrics(test_final, target_name, ['gini'])
                if gini_test + 0.25 < gini_train:
                    posible_sobreajuste.append(var)
                    
            except: pass
        
        if len(posible_sobreajuste) > 0:
            print('Posibles variables con overfitting: {}'.format(posible_sobreajuste))
            print('-' * N)
            
    old_ks, old_gini = 0, 0
    
    features = []
    
    num_included = len(included_vars)
    for i in range(num_included + max_iters):

        if i < num_included:

            new_var = included_vars.pop(0)
            features.append(new_var)

            if metric == 'pvalue':

                scorecard, features_length, pvalues = compute_scorecard(
                data, features, info, target_name=target_name, pvalues=True, logistic_method=logistic_method)
                train_final = apply_scorecard(data, scorecard, info, target_name)
                ks_train, gini_train = compute_metrics(train_final, target_name, ['ks', 'gini'])
                if not isinstance(muestra_test, type(None)):
                    test_final = apply_scorecard(muestra_test, scorecard, info, target_name)
                    ks_test, gini_test = compute_metrics(test_final, target_name, ['ks', 'gini'])
                
                if pvalues[-1] < 1e-100: pvalorcito = 0
                else: pvalorcito = pvalues[-1]

                if isinstance(muestra_test, type(None)):
                    if show == 'ks':
                        print('Step {} | 0:00:00.000000 | pv = {:.2e} '
                        '| KS train = {:.2f}% ---> Feature selected: {}'
                        .format(str(i+1).zfill(2), pvalorcito, ks_train*100, var))
                    if show == 'gini':
                        print('Step {} | 0:00:00.000000 | pv = {:.2e} '
                        '| Gini train = {:.2f}% ---> Feature selected: {}'
                        .format(str(i+1).zfill(2), pvalorcito, gini_train*100, var))
                    if show == 'both':
                        print('Step {} | 0:00:00.000000 | pv = {:.2e} '
                        '| KS train = {:.2f}% | Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalorcito, 
                        ks_train*100, gini_train*100, new_var))

                else:
                    if show == 'ks':
                        print('Step {} | 0:00:00.000000 | pv = {:.2e} '
                        '| KS train = {:.2f}% | KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalorcito,
                        ks_train*100, ks_test*100, new_var))
                    if show == 'gini':
                        print('Step {} | 0:00:00.000000 | pv = {:.2e} '\
                        '| Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), pvalorcito,
                        gini_train*100, gini_test*100, new_var))
                    if show == 'both':
                        print('Step {} | 0:00:00.000000 | pv = {:.2e} | KS train = {:.2f}% '
                        '| KS test = {:.2f}% | Gini train = {:.2f}% | Gini test '
                        '= {:.2f}% ---> Feature selected: {}'.format(str(i+1).zfill(2), pvalorcito,
                         ks_train*100, ks_test*100, gini_train*100, gini_test*100, new_var)) 

            else:

                scorecard, features_length = compute_scorecard(
                data, features, info, target_name=target_name, logistic_method=logistic_method)
                train_final = apply_scorecard(data, scorecard, info, target_name)
                ks_train, gini_train = compute_metrics(train_final, target_name, ['ks', 'gini'])
                if not isinstance(muestra_test, type(None)):
                    test_final = apply_scorecard(muestra_test, scorecard, info, target_name)
                    ks_test, gini_test = compute_metrics(test_final, target_name, ['ks', 'gini'])

                if isinstance(muestra_test, type(None)):
                    if show == 'ks':
                        print('Step {} | 0:00:00.000000 | KS train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), ks_train*100, new_var),)
                    if show == 'gini':
                        print('Step {} | 0:00:00.000000 | Gini train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), gini_train*100, new_var))
                    if show == 'both':
                        print('Step {} | 0:00:00.000000 | KS train = {:.2f}% | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), ks_train*100, gini_train*100, new_var))
                else:
                    if show == 'ks':
                        print('Step {} | 0:00:00.000000 | KS train = {:.2f}% | '
                        'KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), ks_train*100, ks_test*100, new_var))
                    if show == 'gini':
                        print('Step {} | 0:00:00.000000 | Gini train = {:.2f}% | '
                        'Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), gini_train*100, gini_test*100, new_var))
                    if show == 'both':
                        print('Step {} | 0:00:00.000000 | KS train = {:.2f}% | '
                        'KS test = {:.2f}% | Gini train = {:.2f}% | Gini test = {:.2f}% '
                        '---> Feature selected: {}'.format(str(i+1).zfill(2), ks_train*100,
                        ks_test*100, gini_train*100, gini_test*100, new_var))

        else:

            if method == 'forward':

                if metric not in ('ks', 'gini'):
                    raise ValueError("El método 'forward' "
                    "solo se puede usar con las métricas 'ks' o 'gini'")

                start = datetime.datetime.now()

                contador = 0
                aux = pd.DataFrame(columns=['var', 'metric'])
                
                for var in var_list:

                    if var not in features:

                        features.append(var)
                        scorecard, features_length = compute_scorecard(
                        data, features, info, target_name=target_name, logistic_method=logistic_method)
                        data_final = apply_scorecard(data, scorecard, info, target_name)
                        metrica = compute_metrics(data_final, target_name, [metric])
                        aux.loc[contador] = [var, metrica]
                        features.pop()
                        contador += 1

                aux = aux.sort_values('metric', ascending=False)
                new_var = aux.iloc[0]['var']
                features.append(new_var)

                scorecard, features_length = compute_scorecard(
                data, features, info, target_name=target_name, logistic_method=logistic_method)
                train_final = apply_scorecard(data, scorecard, info, target_name)
                ks_train, gini_train = compute_metrics(train_final, target_name, ['ks', 'gini'])
                
                if metric == 'ks':
                    if ks_train <= old_ks+0.0020:
                        print('-' * N)
                        print('En el siguiente paso el KS no sube '
                        'más de un 0.20, detenemos el proceso')
                        features.pop()
                        break
                
                elif metric == 'gini':
                    if gini_train <= old_gini+0.0030:
                        print('-' * N)
                        print('En el siguiente paso el Gini no sube '
                        'más de un 0.30, detenemos el proceso')
                        features.pop()
                        break
                    
                old_ks, old_gini = ks_train, gini_train
                
                if not isinstance(muestra_test, type(None)):
                    test_final = apply_scorecard(muestra_test, scorecard, info, target_name)
                    ks_test, gini_test = compute_metrics(test_final, target_name, ['ks', 'gini'])

                if isinstance(muestra_test, type(None)):
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print('Step {} | {} | KS train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, new_var))
                    if show == 'gini':
                        print('Step {} | {} | Gini train = {:.2f}% '
                        '---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, gini_train*100, new_var))
                    if show == 'both':
                        print('Step {} | {} | KS train = {:.2f}% | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, gini_train*100, new_var))
                else:
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print('Step {} | {} | KS train = {:.2f}% | '
                        'KS test = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, ks_train*100, ks_test*100, new_var))
                    if show == 'gini':
                        print('Step {} | {} | Gini train = {:.2f}% | '
                        'Gini test = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, gini_train*100, gini_test*100, new_var))
                    if show == 'both':
                        print('Step {} | {} | KS train = {:.2f}% | '
                        'KS test = {:.2f}% | Gini train = {:.2f}% | Gini test = {:.2f}% '
                        '---> Feature selected: {}'.format(str(i+1).zfill(2), end - start,
                        ks_train*100, ks_test*100, gini_train*100, gini_test*100, new_var))

            elif method == 'stepwise':

                if metric != 'pvalue':
                    raise ValueError("El método 'stepwise' "
                    "solo se puede usar con la métrica 'pvalue'")

                start = datetime.datetime.now()

                contador = 0
                aux = pd.DataFrame(columns=['var', 'pvalue', 'gini_auxiliar'])
                
                for var in var_list:

                    if var not in features:

                        features.append(var)
                        scorecard, features_length, pvalues = compute_scorecard(
                        data, features, info, target_name=target_name, pvalues=True, logistic_method=logistic_method)
                        pvalue = pvalues[-1]
                        if pvalue == 0:
                            gini_auxiliar = compute_metrics(apply_scorecard(
                            data, scorecard, info, target_name), target_name, ['gini'])
                        else: gini_auxiliar = 0
                        aux.loc[contador] = [var, pvalue, gini_auxiliar]
                        features.pop()
                        contador += 1

                aux = aux.sort_values(['pvalue', 'gini_auxiliar'], ascending=[True, False])
                best_pvalue = aux.iloc[0]['pvalue']

                if best_pvalue >= threshold:
                    print('-' * N)
                    print('Ya ninguna variable tiene un p-valor'
                    ' < {}, detenemos el proceso'.format(threshold))
                    break

                new_var = aux.iloc[0]['var']
                features.append(new_var)

                scorecard, features_length, pvalues = compute_scorecard(
                data, features, info, target_name=target_name, pvalues=True, logistic_method=logistic_method)
                new_pvalue = pvalues[-1]
                train_final = apply_scorecard(data, scorecard, info, target_name)
                ks_train, gini_train = compute_metrics(train_final, target_name, ['ks', 'gini'])
                
                if stop_ks_gini:
                    if ks_train <= old_ks+0.002 and gini_train <= old_gini+0.003:
                        print('-' * N)
                        print('En el siguiente paso ni el KS ni el GINI del train '
                        'suben más de un 0.20 o un 0.30 respectivamente, detenemos el proceso')
                        features.pop()
                        break
                old_ks, old_gini = ks_train, gini_train
                
                if not isinstance(muestra_test, type(None)):
                    test_final = apply_scorecard(muestra_test, scorecard, info, target_name)
                    ks_test, gini_test = compute_metrics(test_final, target_name, ['ks', 'gini'])

                if new_pvalue < 1e-100: pvalorcito = 0
                else: pvalorcito = new_pvalue

                if isinstance(muestra_test, type(None)):
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print('Step {} | {} | pv = {:.2e} | '
                        'KS train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, pvalorcito, ks_train*100, new_var))
                    if show == 'gini':
                        print('Step {} | {} | pv = {:.2e} | '
                        'Gini train = {:.2f}% ---> Feature selected: {}'.format(
                        str(i+1).zfill(2), end - start, pvalorcito, gini_train*100, new_var))
                    if show == 'both':
                        print('Step {} | {} | pv = {:.2e} | '
                        'KS train = {:.2f}% | Gini train = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, pvalorcito,
                        ks_train*100, gini_train*100, new_var))

                else:
                    end = datetime.datetime.now()
                    if show == 'ks':
                        print('Step {} | {} | pv = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, pvalorcito,
                        ks_train*100, ks_test*100, new_var))
                    if show == 'gini':
                        print('Step {} | {} | pv = {:.2e} | '
                        'Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature selected: {}'\
                        .format(str(i+1).zfill(2), end - start, pvalorcito,
                        gini_train*100, gini_test*100, new_var))
                    if show == 'both':
                        print('Step {} | {} | pv = {:.2e} | '
                        'KS train = {:.2f}% | KS test = {:.2f}% | Gini train = {:.2f}% | Gini test '
                        '= {:.2f}% ---> Feature selected: {}'.format(str(i+1).zfill(2),
                        end - start, pvalorcito, ks_train*100, ks_test*100,
                        gini_train*100, gini_test*100, new_var))

                dict_pvalues = dict(zip(features, pvalues[1:]))
                to_delete = {}
                for v in dict_pvalues:
                    if dict_pvalues[v] >= threshold:
                        to_delete[v] = dict_pvalues[v]
                if to_delete != {}:
                    for v in to_delete:
                        features.remove(v)
                        scorecard, features_length = compute_scorecard(
                        data, features, info, target_name=target_name, logistic_method=logistic_method)
                        train_final = apply_scorecard(data, scorecard, info, target_name)
                        ks_train, gini_train = compute_metrics(
                        train_final, target_name, ['ks', 'gini'])
                        old_ks, old_gini = ks_train, gini_train
                        if not isinstance(muestra_test, type(None)):
                            test_final = apply_scorecard(
                            muestra_test, scorecard, info, target_name)
                            ks_test, gini_test = compute_metrics(
                            test_final, target_name, ['ks', 'gini'])

                            if isinstance(muestra_test, type(None)):
                                end = datetime.datetime.now()
                                if show == 'ks':
                                    print('Step {} | 0:00:00.000000 | pv '
                                    '= {:.2e} | KS train = {:.2f}% ---> Feature deleted : {}'\
                                    .format(str(i+1).zfill(2), dict_pvalues[v], ks_train*100, v))
                                if show == 'gini':
                                    print('Step {} | 0:00:00.000000 | pv '
                                    '= {:.2e} | Gini train = {:.2f}% ---> Feature deleted : {}'\
                                    .format(str(i+1).zfill(2), dict_pvalues[v], gini_train*100, v))
                                if show == 'both':
                                    print('Step {} | 0:00:00.000000 | pv '
                                    '= {:.2e} | KS train = {:.2f}% | Gini train '
                                    '= {:.2f}% ---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], ks_train*100, gini_train*100, v))

                            else:
                                end = datetime.datetime.now()
                                if show == 'ks':
                                    print('Step {} | 0:00:00.000000 | pv '
                                    '= {:.2e} | KS train = {:.2f}% | KS test = {:.2f}% '
                                    '---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], ks_train*100, ks_test*100, v))
                                if show == 'gini':
                                    print('Step {} | 0:00:00.000000 | pv '
                                    '= {:.2e} | Gini train = {:.2f}% | Gini test = {:.2f}% '
                                    '---> Feature deleted : {}'.format(str(i+1).zfill(2),
                                    dict_pvalues[v], gini_train*100, gini_test*100, v))
                                if show == 'both':
                                    print('Step {} | 0:00:00.000000 | pv '
                                    '= {:.2e} | KS train = {:.2f}% | KS test = {:.2f}% | '
                                    'Gini train = {:.2f}% | Gini test = {:.2f}% ---> Feature '
                                    'deleted : {}'.format(str(i+1).zfill(2), dict_pvalues[v],
                                    ks_train*100, ks_test*100, gini_train*100, gini_test*100, v))

    print('-' * N)
    print('Selección terminada: {}'.format(features))
    print('-' * N)

    return features    


def display_table_ng(modelo_newgroups, candidate_var, objeto, bp):

    if not isinstance(bp, dict):
        vector = data_convert(
        modelo_newgroups.X_train[candidate_var].values, string_categories2(bp))[3]
        breakpoints_num = breakpoints_to_num(bp)
        groups_names = compute_group_names(objeto.dtype, bp)
        display(compute_table(vector, modelo_newgroups.y_train, breakpoints_num, groups_names)[0])

    else:
        vector = remapeo_missing(data_convert(
        modelo_newgroups.X_train[candidate_var].values, string_categories2(bp))[3], bp)
        breakpoints_num = breakpoints_to_num(bp['bp'])
        groups_names = compute_group_names(objeto.dtype, bp['bp'], bp['mg'])                                 
        display(compute_table(vector, modelo_newgroups.y_train, breakpoints_num, groups_names)[0])


def reagrupa_var(modelo, variable, bp=[], decimals=4):

    objeto = modelo.autogroupings[variable]
    if objeto.dtype != 'O':
        L = [round(i, decimals) for  i in list(objeto.breakpoints)]
        print('Agrupación automática (puntos de corte '
        'redondeados a {} decimales): {}'.format(decimals, L))
    else:
        L = objeto.breakpoints
        print('Agrupación automática: {}'.format(L))
    
    display(objeto.table)
    
    if bp != []:
        print('-'*80)
        print('Agrupación propuesta:')
        display_table_ng(modelo, variable, objeto, bp)

