import pandas as pd
import numpy as np
import os

from datetime import datetime


def process_behavior(demgeo_df, behavior_df, 
                     client_id = 'bridge_company_id', 
                     filter_rows = {'last_recurrence' : 'Trimestral'}, 
                     became_date_col = 'became_customer_date', churn_date_col = 'churn_date',
                     behavior_volume_col = 'total_issues', behavior_date_col = 'days',
                     time_unit = 'MS', period_len = 3,
                     rolling_window = None,
                     include_term = True, include_month = True,
                     split_by_date = datetime(2017,1,1)): #other options 

    """
    Arguments:
    ==========
    Note: all dates in datetime format

    DataFrames:
    -----------
    demgeo_df : a df with demographics and geographic data
    behavior_df_raw : a df with behavior data, in format (client_id, date, behavior_volume). 

    demgeo_df parameters:
    ---------------------
    client_id : column name with the id of the customer. Note: the column name should be the same name in behavior_df  
    filter_rows : None or a dict with {col_name: filter_value} to eliminate some filter_rows. filter_value contains the values to keep.
    became_date_col : column name with the date this id became a customer
    churn_date_col: column name with the date the customer cancelled the service. NaT indicates the customer is active

    behavior_df parameters:
    -----------------------
    behavior_volume_col : column name with the volume (qty) of observations of the behavioral feature
    behavior_date_col : column name with the date the behavioral volume was recorded

    function option parameters:
    ---------------------------
    time_unit      : how to group the data into periods. Ex: 'W', 'MS', ...
    period_len     : quantity of periods to record as features
    rolling_window : None or the size of window (in periods) to compute rolling mean statistics in a new column
                     A negative number results in volume_of_current_month - rolling_mean 
    include_term   : include a column with the term of customer, i.e., the number of past renewals 
    include_month  : include a column with the month of the begining of the term. If int, offset the month by this value.  
    split_by_date  : None or the actual date for split. If not None generates a bool column called 'train'
                     split_by_date is the start of the term, and not the month of the renewal decision

    Returns:
    ========
    Dataframe where each row is an observation of the customer term. Term is the duration of a customer subscription plan.
    Ex: A single customer with behaviors: [M_0, M_1, M_2, M_3, M_4, M_5] is segmented into 2 observations,
        * [M_0, M_1, M_2, label_0]
        * [M_3, M_4, M_5, label_1]
        where M_k is the volume of behaviors (e.g., issues) in the k-th period
        label_j is churn at the end of the j-th term

    Other added columns may be included:
      * 'term' int column with the term
      * 'train' bool column identifying rows that happen before split_by_date


    Assumptions and simplifications:
    ================================
    cancellations in the middle of a plan are considered to be related to the previous subscription term
    # por exemplo, numa sequencia de assinatura trimestral com volumes mensais: (M0, M1, M2, M3, M4) e churn no M4, 
    # os behaviors observados nos meses (M3, M4) sao descartados, e apenas uma obs (M0, M1, M2) & churn = 1 sera considerada 
    # nota 1: clientes com menos meses que um termo serão descartados, ou seja aqueles que não chegaram nem a 1a renovacao
    # nota 2: o ultimo termo (incompleto) de um cliente ativo tambem sera descartado

    TODO's:
    =======

    suspeita de leak
    ----------------
    os comportamentos (ex. total issues) NÃO PODEM ser relacionados com o cancelamento de serviço. 
    ou seja todos aqueles issues referentes ao cancelamento precisam ser retirados da soma do behavior_volume_col

    time_unit diferente de 'MS'
    ---------------------------
    * Funciona somente para time_unit = 'MS'. Se time_unit = 'W' sera necessario corrigir. 
      se freq ='W' no pd.Grouper a data que representa o periodo eh o domingo *posterior* aos eventos e datas.
      por consistencia, deveria ser o domingo *anterior*.
    * Alternativamente, o pd.Grouper ancorado no domingo *anterior*  (freq = 'WS' nao funciona) aos eventos e datas.

    periodos a descartar
    --------------------
    a quantidade de periodos a descartar deve ser para cada cliente segundo sua recorrencia. 
    por simplicidade, deve-se usar filter_rows para limitar as recorrencias assim todos os termos terao o mesmo valor.
    ex: Mensal => termo = 1; Trimestral => termo = 3; Semestral => termo = 6; Anual => termo = 12

    janela de features diferente do termo
    -------------------------------------
    implementar uma quantidade de colunas de features diferente, por exemplo maior, do que o tamanho do termo.
    ex: termo de 3 meses, e uma janela com 4 meses. [M_0, M_1, M_2, M_3, M_4, M_5] geraria:
        * [-1 , M_0, M_1, M_2, label_0]
        * [M_2, M_3, M_4, M_5, label_1]


    """
    
    # consistency checks
    if not (demgeo_df[ demgeo_df[churn_date_col].notnull() ][churn_date_col] >= 
      demgeo_df[ demgeo_df[churn_date_col].notnull() ][became_date_col]).all(): 
        raise ValueError('existem datas churn nao nulas anteriores as datas became customer')

    # seleciona colunas do demgeo_df
    demgeo_df_listtokeep = [client_id, became_date_col, churn_date_col] 
    if filter_rows is not None:
        demgeo_df_listtokeep += [k for k in filter_rows]
    demgeo_df = demgeo_df[demgeo_df_listtokeep].copy()

    # aplica restrição nas linhas do demgeo. 
    # Ex: somente recorrencia trimestral
    for col in filter_rows:
        demgeo_df = demgeo_df[demgeo_df[col] == filter_rows[col]]
        demgeo_df.drop(col, axis = 1, inplace = True)

    # restringindo clientes do behavior_df para os que permanecerem na base do demgeo_df
    behavior_df = behavior_df[ behavior_df[client_id].isin(demgeo_df[client_id]) ]

    # inserindo os clientes que nunca abriram issues. 
    # cria uma linha artificial contendo behavior_volume_col = 0, na data do became_customer 
    ids_no_behavior = set(demgeo_df[client_id]) - set(behavior_df[client_id])
    ids_no_behavior = demgeo_df[demgeo_df[client_id].isin(ids_no_behavior)].drop('churn_date', axis = 1)
    ids_no_behavior.rename(columns = {'became_customer_date': behavior_date_col}, inplace = True)
    ids_no_behavior[behavior_volume_col] = 0
    behavior_df = pd.concat([behavior_df, ids_no_behavior])

    # agrupa segundo cliente e periodo, no momento apenas funciona com time_unit = 'MS'
    # ja que 'WS' nao funciona. ver TODO acima
    period_grouper = pd.Grouper(key = behavior_date_col, freq = time_unit)
    behavior_df = behavior_df.groupby([client_id, period_grouper])[behavior_volume_col].sum()
    behavior_df = behavior_df.to_frame()

    # cria um ranges de datas, entre (data mais antiga: data mais recente: intervalo), para o reindex
    min_date = behavior_df.index.get_level_values(-1).min()
    max_date = behavior_df.index.get_level_values(-1).max()
    date_range = pd.date_range(start = min_date, end = max_date, freq = time_unit)

    # reindexa o nivel -1, i.e., as datas
    # importante: inclui fill_value = 0
    reindex_last_level = lambda x: x.reset_index(-1).set_index(behavior_date_col).reindex(date_range, fill_value = 0)
    behavior_df = behavior_df.groupby(level = 0).apply(reindex_last_level)

    assert behavior_df.index.get_level_values(0).unique().shape == demgeo_df[client_id].unique().shape

    # junta as colunas de became e churn
    # note: join ao inves de concat, porque o behavior_df tem client_id no nivel 0
    # todo: checar duplicidades de client_id no demgeo
    behavior_df = behavior_df.join(demgeo_df.set_index('bridge_company_id'))

    # transforma datas para o primeiro dia do mes
    for col in [became_date_col, churn_date_col]:
        behavior_df[col] = behavior_df[col].values.astype('datetime64[M]')
        behavior_df[col + '_flag'] = behavior_df[col] == behavior_df.index.get_level_values(1)
        behavior_df[col + '_flag'] = behavior_df[col + '_flag'].groupby(level = 0).cumsum()

    # flag para indicar usuario churned    
    behavior_df['churned'] = behavior_df[churn_date_col].notnull()

    # flag para indicar periodo ativo, gerado a partir do (cumsum became ) & (not cumsum churn)
    behavior_df['active_flag'] = behavior_df[became_date_col + '_flag'] & ~behavior_df[churn_date_col + '_flag']

    behavior_df.drop([became_date_col, churn_date_col, 
                      became_date_col + '_flag', churn_date_col + '_flag'], axis = 1, inplace = True)

    # apaga as linhas para os meses inativos, deletar 'active_flag' nao mais necessario
    behavior_df = behavior_df[behavior_df.active_flag]
    behavior_df.drop('active_flag', axis = 1, inplace = True)

    # simplificacao para obter o term. ver TODO periodos_a_descartar
    term_dict = {'Anual' : 12, 'Trimestral': 3, 'Mensal': 1, 'Semestral': 6}
    term = term_dict[filter_rows['last_recurrence']]

    # remover os periodos de behavior que ocorrem no meio de um termo incompleto. ver docstring
    keep_month_func = lambda x: (term * (len(x) // term)) * [True] + (len(x) % term) * [False]
    behavior_df['keep_month'] = behavior_df.groupby(level = 0).churned.transform(keep_month_func)
    behavior_df = behavior_df[behavior_df.keep_month]
    behavior_df.drop('keep_month', axis = 1, inplace = True)

    # para os clientes churned, manter True apenas no ultimo termo, True para todos termos anteriores
    # para os clientes nao churned sera True para todos os termos
    fix_churn_func = lambda x: (len(x) - 1) * [False] + [True] if x.iloc[0] == True else len(x) * [False]
    behavior_df['churned'] = behavior_df.groupby(level = 0).churned.transform(fix_churn_func)

    # inclui coluna com o termo
    # sorted(list(range(len(x) // term)) * term) eh somente uma maneira nao otimizada para transformar
    # [0,1,2,0,1,2,0,1,2] => [0,0,0,1,1,1,2,2,2]
    behavior_df['term'] = behavior_df.groupby(level = 0).churned.transform(
      lambda x: sorted(list(range(len(x) // term)) * term)).astype(int)

    # inclui as medias moveis, nao incluindo o mes corrente na media
    # ex: media movel dos tres ultimos meses 
    if rolling_window:
        if rolling_window < 0 : 
            rolling_window = abs(rolling_window)
            take_diff = True
        else:
            take_diff = False
        behavior_volume_col_rm = behavior_volume_col + '_rm' + str(rolling_window)
        behavior_df[behavior_volume_col_rm] = behavior_df.groupby(level = 0)[behavior_volume_col].apply(
            lambda x: x.rolling(rolling_window, min_periods = 1).mean().shift().fillna(0))
        if take_diff:
            behavior_df[behavior_volume_col_rm] = behavior_df[behavior_volume_col] - behavior_df[behavior_volume_col_rm]

    # segmenta o behavior_df em diferentes amostras, uma para cada periodo
    behavior_df['date'] = behavior_df.index.get_level_values(1)

    behavior_agg_dict = {behavior_volume_col: lambda x: tuple(x),  # gera tuplas na celula  # todo: dict comprehension 
                         behavior_volume_col_rm: lambda x: tuple(x),   # gera tuplas na celula
                         'churned' : 'sum',
                         'date': 'first',
                         }

    behavior_df = behavior_df.groupby([behavior_df.index.get_level_values(0), 'term']).agg(behavior_agg_dict)

    # inclui termo
    if include_term:
        behavior_df['term'] = behavior_df.index.get_level_values(1)

    # inclui mes
    if include_month:
        behavior_df['month'] = behavior_df.date.apply(lambda x: x.month)
        if type(include_month) is int:
            behavior_df['month'] = behavior_df.month.apply(lambda x: (x + include_month) % 12)


    # expande as tuplas para colunas
    for col in [behavior_volume_col, behavior_volume_col_rm]: 
        behavior_tmp = behavior_df[col].apply(pd.Series)
        behavior_tmp.columns = [col + '_' + str(j) for j in behavior_tmp]
        behavior_df = pd.concat([behavior_df.drop(col, axis = 1), behavior_tmp], axis = 1)
        
    # variavel churned de boolean para int
    behavior_df['churned'] = behavior_df.churned.astype(int)

    # separar entre treino e teste, criando uma coluna identificando treino/teste: train = True, test = False
    if split_by_date is not None:
        behavior_df['train'] = behavior_df.date < split_by_date

    behavior_df.drop('date', axis = 1, inplace = True)


    return behavior_df

