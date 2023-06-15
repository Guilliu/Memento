import numpy as np, pandas as pd, statsmodels.api as sm

from scipy import special
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from .todd import * 

####################################################################################################


def compute_group_names(dtype, breakpoints, missing_group=None, decimals=2):

    if dtype in ('O', 'bool'): return breakpoints

    else:

        groups = np.concatenate([[-np.inf], breakpoints, [np.inf]])
        group_names1, group_names2 = [], []

        for i in range(len(groups) - 1):

            if np.isinf(groups[i]):
                a = '({0:.{2}f}, {1:.{2}f})'\
                .format(groups[i], groups[i+1], decimals)
            else: a = '[{0:.{2}f}, {1:.{2}f})'.format(groups[i], groups[i+1], decimals)
            group_names1.append(a)

        for group in group_names1:
            if '-12345670.00)' in group: group = 'Missing'
            if '[-12345670.00' in group: group = group.replace('[-12345670.00', '(-inf')
            group_names2.append(group)

        if isinstance(missing_group, int):
            if missing_group > 0:
                group_names2[missing_group-1] += ', Missing'

        return group_names2


def compute_table(x, y, breakpoints_num, group_names, compute_totals=True):

    x_groups = np.digitize(x, breakpoints_num)
 
    ngroups = len(breakpoints_num) + 1
    g = np.zeros(ngroups).astype(np.int64)
    b = np.zeros(ngroups).astype(np.int64)

    for i in range(ngroups):

        g[i] = np.sum([(y == 0) & (x_groups == i)])
        b[i] = np.sum([(y == 1) & (x_groups == i)])

    e = g + b

    total_g = g.sum()
    total_b = b.sum()
    total_e = e.sum()

    pct_g = g / total_g
    pct_b = b / total_b
    pct_e = e / total_e

    b_rate = b / e
    woe = np.log(1 / b_rate - 1) + np.log(total_b / total_g)
    iv = special.xlogy(pct_b - pct_g, pct_b / pct_g)

    total_b_rate = total_b / total_e
    total_iv = iv.sum()

    table = pd.DataFrame({

        'Group': group_names,
        'Count': e,
        'Percent': pct_e,
        'Goods': g,
        'Bads': b,
        'Bad rate': b_rate,
        'WoE': woe,
        'IV': iv,
    })

    if compute_totals:
        table.loc['Totals'] = ['', total_e, 1, total_g, total_b, total_b_rate, '', total_iv]

    return table, total_iv


def transform_to_woes(x, y, breakpoints_num):

    x_groups = np.digitize(x, breakpoints_num)

    ngroups = len(breakpoints_num) + 1
    g = np.empty(ngroups).astype(np.int64)
    b = np.empty(ngroups).astype(np.int64)

    for i in range(ngroups):

        g[i] = np.sum([(y == 0) & (x_groups == i)])
        b[i] = np.sum([(y == 1) & (x_groups == i)])

    e = g + b

    total_g = g.sum()
    total_b = b.sum()
    b_rate = b / e

    woe = np.log(1 / b_rate - 1) + np.log(total_b / total_g)

    mapeo_indices_woes = dict(zip([i for i in range(len(woe))], list(woe)))
    x_woes = pd.Series([mapeo_indices_woes[i] for i in x_groups])

    return x_woes


def calib_score(points, num_variables, intercept):

    n = num_variables
    pdo, odds, scorecard_points = 20, 1, 500

    factor = pdo / np.log(2)
    offset = scorecard_points - factor * np.log(odds)

    new_points = -(points + intercept / n) * factor + offset / n

    return new_points


def compute_scorecard(data, features, info, target_name='target_4815162342',
pvalues=False, ret_coefs=False, redondeo=True, logistic_method='newton'):

    X = data.drop(target_name, axis=1).copy()
    y = data[target_name].values

    Xwoes = pd.DataFrame()
    scorecard, features_length = pd.DataFrame(), np.array([], 'int64')

    for feature in features:

        x = X[feature].values
        breakpoints_num = info[feature]['breakpoints_num']
        group_names = info[feature]['group_names']

        table = compute_table(x, y, breakpoints_num, group_names, False)[0]
        table.insert(0, 'Variable', feature)
        scorecard = pd.concat([scorecard, table])
        features_length = np.append(features_length, len(table))

        Xwoes[feature] = transform_to_woes(x, y, breakpoints_num)

    log_reg = sm.Logit(y, sm.add_constant(Xwoes.values)).fit(method=logistic_method, disp=0)
    coefs, intercept = np.array([log_reg.params[1:]]), np.array([log_reg.params[0]])

    scorecard['Raw score'] = scorecard['WoE'] * np.repeat(coefs.ravel(), features_length)
    scorecard['Aligned score'] = calib_score(scorecard['Raw score'], len(features), intercept)
    if redondeo: scorecard['Aligned score'] = scorecard['Aligned score'].round().astype('int')
    scorecard = scorecard.reset_index(drop=True)

    if pvalues and ret_coefs: return scorecard, features_length, log_reg.pvalues, coefs.ravel()
    if pvalues and not ret_coefs: return scorecard, features_length, log_reg.pvalues
    if not pvalues and ret_coefs: return scorecard, features_length, coefs.ravel()
    if not pvalues and not ret_coefs: return scorecard, features_length


def transform_to_points(x, breakpoints_num, mapeo_points):

    return pd.Series([mapeo_points[i] for i in np.digitize(x, breakpoints_num)])


def apply_scorecard(data, scorecard, info, target_name='',
binary_treshold=0.0, score_name='scorecardpoints'):

    features = list(scorecard['Variable'].unique())
    if target_name != '': columnas = features + [target_name]
    else: columnas = features
                 
    data_final = data[columnas].copy()
    data_final = data.copy()
    data_final[score_name] = 0

    for feature in features:

        x = data_final[feature]
        breakpoints_num = info[feature]['breakpoints_num']

        mapeo_points = scorecard[scorecard['Variable'] == feature]\
        .reset_index(drop=True)['Aligned score'].to_dict()

        data_final['scr_{}'.format(feature)] = \
        transform_to_points(x, breakpoints_num, mapeo_points)
        data_final[score_name] += data_final['scr_{}'.format(feature)]
    
    columnas = list(data_final.columns)
    columnas.remove(score_name)
    data_final = data_final[columnas + [score_name]]

    if binary_treshold != 0.0:
        data_final['prediction'] = np.where(data_final[score_name] >= binary_treshold, 0, 1)

    return data_final

def compute_metrics(data, target_name, metrics, print_log=False, score_name='scorecardpoints'):

    if metrics not in (['ks'], ['gini'], ['ks', 'gini'], ['gini', 'ks']):
        raise ValueError("Valor erroneo para 'metrics'. Los valores "
        "váidos son: ['ks'], ['gini'], ['ks', 'gini'], ['gini', 'ks']")

    if 'ks' in metrics:
        g = data.loc[data[target_name] == 0, score_name]
        b = data.loc[data[target_name] == 1, score_name]
        ks = ks_2samp(g, b)[0]

    if 'gini' in metrics:
        gini = abs(2*(1 - roc_auc_score(data[target_name], data[score_name])) - 1)

    if metrics == ['ks']:
        if print_log:
            print('El modelo tiene un {:.2f}% de KS en esta muestra'.format(round(ks*100, 2)))
        return ks

    if metrics == ['gini']:
        if print_log:
            print('El modelo tiene un {:.2f}% de Gini en esta muestra'.format(round(gini*100, 2), ))
        return gini

    if metrics in (['ks', 'gini'], ['gini', 'ks']):
        if print_log:
            print('El  modelo tiene un {:.2f}% de KS y un {:.2f}% de Gini '
            'en esta muestra'.format(round(ks*100, 2), round(gini*100, 2)))
        return ks, gini

