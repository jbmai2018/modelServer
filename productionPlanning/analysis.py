import datetime
import time
import pandas as pd
import numpy as np
from math import ceil
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def addtime(timeList):
    summ = datetime.timedelta()
    for i in timeList:
        (h, m, s) = i.split(':')
        d = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        summ += d
    return str(summ)#.split('day')[-1].strip()

def concat_common_parts(df,columns_to_add):
    df = df.copy()
    for part in common_parts:
        if len(df[df['Part No.']==part[0]]) & len(df[df['Part No.']==part[1]]):
            print(part)
            for col in columns_to_add:
                df.loc[(df['Part No.']==part[0]),col] = df[df['Part No.']==part[0]][col].values[0] + df[df['Part No.']==part[1]][col].values[0]
            df.loc[(df['Part No.']==part[0]),'Part No.'] = part[2]
    return df


def extract_int(text):
    num = ""
    number = []
    flag = False
    for c in text:
        if c.isdigit():
            num+=c
            flag=True
        elif flag:
            number.append(int(num))
            num=""
            flag = False
    return sum(number)

def try_except(key):
    try:
        return dict_to_use[key][0]
    except:
        return 0


def get_quantity(row):
    max_quant = max(row['BAL'], row['day_wise_requirement'])

    return row['RM packaging standard'] * ceil(max_quant / row['RM packaging standard'])

def get_plan(n, loading_file, j1_cric_file, j3_cric_file, Initial_time, critical_col=None):
    ## Read operation_SPM_List
    print("Reading Excel files")
    df_SPM = pd.read_excel(os.path.join(BASE_DIR,'productionPlanning/I4_part_SPM_operations.xlsx'))
    df = pd.read_excel(loading_file, sheet_name=int(n))
    df.rename(columns={'Total': 'Inventory', 'Grand Total': 'Grand req.'}, inplace=True)

    ## Reading critical sheets
    df_j1 = pd.read_excel(j1_cric_file, sheet_name=int(n))
    df_j3 = pd.read_excel(j3_cric_file, sheet_name=int(n))

    ## Reading rm_packaging_file
    df_rm = pd.read_excel(os.path.join(BASE_DIR,"productionPlanning/RM PACKAGING _.xlsx"), sheet_name=0)
    df_day_wise = pd.read_excel(os.path.join(BASE_DIR,"productionPlanning/RM PACKAGING _.xlsx"), sheet_name=1)

    if critical_col:
        cric_j1 = critical_col
        cric_j3 = critical_col
    else:
        cric_j1 = [i for i in df_j1.columns if str(i).strip().lower().startswith('critical')][0]
        cric_j3 = [i for i in df_j3.columns if str(i).strip().lower().startswith('critical')][0]
    print(cric_j1, "------------", cric_j3)

    rename_col = {"Bal": 'BAL', 'BAL.': 'BAL', 'Bal.': 'BAL', 'STATUS at 5.30pm': 'Status', 'STATUS': 'Status',
                  'Desc.': 'Part No.'}
    df_j3.rename(columns=rename_col, inplace=True)
    df_j1.rename(columns=rename_col, inplace=True)

    # Get next day requirements
    col_j1 = df_j1.columns[df_j1.columns.get_loc("BAL") + 1]
    col_j3 = df_j3.columns[df_j3.columns.get_loc("BAL") + 1]

    rename_col = {'Part Name': 'Part No.', 'Old Part No': 'Part No.', 'Part no': 'Part No.', 'Part no.': 'Part No.',
                  cric_j3: cric_j1, "Bal": 'BAL', 'Bal.': 'BAL', 'STATUS at 5.30pm': 'Status', col_j1: 'next_day',
                  col_j3: 'next_day', 'STATUS': 'Status'}
    df_j3.rename(columns=rename_col, inplace=True)
    df_j1.rename(columns=rename_col, inplace=True)

    common_parts = [("YCA-46515", "YCA-46516", "YCA-46515/16"), ("Y9T-67121", "Y9T-67421", "Y9T-67121/421"),
                    ("YSD-61316", "YSD-61326", "YSD-61316/326"), ("YJC-61224", "YJC-61225", "YJC-61224/25"),
                    ("YE3-58311", "YE3-58411", "YE3-58311/411")]

    df_j1 = df_j1[df_j1['Part No.'].isin(df_SPM['Part No.'].values.tolist())][
        [cric_j1, 'Part No.', 'BAL', 'Status', 'next_day']]
    df_j1 = df_j1.loc[:, ~df_j1.columns.duplicated(keep='last')]
    df_j3 = df_j3[df_j3['Part No.'].isin(df_SPM['Part No.'].values.tolist())][
        [cric_j1, 'Part No.', 'BAL', 'Status', 'next_day']]
    df_j3 = df_j3.loc[:, ~df_j3.columns.duplicated(keep='last')]

    print(df_j1)
    print('^^^^^^^^^^^^^^^^^^^^')
    print(df_j3)

    df_cric = pd.concat([df_j1, df_j3])
    aggregation_functions = {'BAL': 'sum', cric_j1: 'first', 'Status': 'first', 'next_day': 'first'}
    df_cric = df_cric.groupby(df_cric['Part No.']).aggregate(aggregation_functions)
    df_cric.reset_index(inplace=True)

    ## Extract Number from Status
    df_cric['Status'] = df_cric['Status'].apply(lambda x: extract_int(str(x)))
    df_cric['BAL'] = df_cric['BAL'] + df_cric['next_day'] - df_cric['Status']
    df_cric.drop(['Status', 'next_day'], axis=1, inplace=True)

    df_I4_merge = df[['CUS', 'Line', 'Part No.', 'Inventory', 'Today shortage', 'Grand req.']].merge(df_SPM, on=[
        'Part No.'], how='right')
    df_I4_merge = df_cric.merge(df_I4_merge, on=['Part No.'], how='right')

    cat_col = df_I4_merge.select_dtypes(include='object').columns.tolist()
    cat_col = {val: 'first' for val in cat_col}

    num_col = df_I4_merge.select_dtypes(exclude='object').columns.tolist()
    num_col = {val: 'sum' for val in num_col}

    cat_col.update(num_col)

    print(cat_col)

    cat_col['SPM'] = 'first'
    cat_col['operations'] = 'first'
    cat_col['BAL'] = 'max'
    df_to_use = df_I4_merge.groupby(df_I4_merge['Part No. To use']).aggregate(cat_col)

    df_to_use.reset_index(drop=True, inplace=True)
    df_to_use.drop(columns=['Part No.'], axis=1, inplace=True)
    df_to_use.rename(columns={'Part No. To use': 'Part No.'}, inplace=True)

    # Swap columns
    cols = list(df_to_use.columns)
    a, b = cols.index(cric_j1), cols.index('Part No.')
    cols[b], cols[a] = cols[a], cols[b]
    df_to_use = df_to_use[cols]
    #         return df_to_use
    df_to_use['Time to Produce minutes'] = round(df_to_use['BAL'] / df_to_use['SPM'], 0)
    df_to_use = df_to_use[df_to_use[cric_j1] != 'Line stopped']
    df_to_use = df_to_use[(df_to_use['BAL'] > 0)]
    df_to_use = df_to_use[(df_to_use['Time to Produce minutes'] > 60)]

    #         if df_to_use[~df_to_use[cric_j1].isnull()]['Time to Produce minutes'].sum()<1440:
    #             df_to_use = df_to_use[df_to_use['Time to Produce minutes'].cumsum() < 1440]
    #         else:
    #             df_to_use = df_to_use[~df_to_use[cric_j1].isnull()]
    df_to_use['Time to Produce'] = df_to_use['Time to Produce minutes'].apply(
        lambda x: time.strftime('%H:%M:%S', time.gmtime(x * 60)))
    df_to_use = df_to_use.sort_values(by=[cric_j1, 'BAL', 'SPM'], ascending=[True, False, True]).reset_index(
        drop=True)
    #         if df_to_use[~df_to_use[cric_j1].isnull()]['Time to Produce minutes'].sum()<1440:
    #             df_to_use = df_to_use[temp_df['Time to Produce minutes'].cumsum() < 1440]
    #         else:
    #             df_to_use = df_to_use[~df_to_use[cric_j1].isnull()]
    ## Perday Requirement and RM packing
    dict_to_use = df_day_wise.set_index('Part No.').T.to_dict('list')
    df_rm['day_wise_requirement'] = df_rm['Part No.'].apply(lambda x: try_except(str(x).split('-')[0]))

    df_to_use = df_to_use.merge(df_rm, on=['Part No.'], how='left')

    df_to_use['temp_BAL'] = df_to_use.apply(lambda x: get_quantity(x), axis=1)
    df_to_use['BAL'] = df_to_use['temp_BAL']

    output_df = df_to_use

    Die_Change_time = "00:10:00"
    output_df['Die Change Time'] = np.nan
    output_df['Plan Time'] = np.nan
    # output_df.sort_values('Time to Produce',inplace=True)
    # output_df = shuffle(output_df)
    preserve_production_time = output_df['Time to Produce'].values.tolist()
    output_df.reset_index(drop=True, inplace=True)
    for index, rows in output_df.iterrows():
        if index == 0:
            output_df.loc[index, 'Die Change Time'] = "00:00:00"
            timeList = [Initial_time, rows['Time to Produce'].split(',')[-1].strip()]
            output_df.loc[index, 'Time to Produce'] = addtime(timeList).split('day')[-1].strip()
            output_df.loc[index, 'Plan Time'] = Initial_time + " - " + addtime(timeList)
        else:
            output_df.loc[index, 'Die Change Time'] = Die_Change_time
            die_switch_time = \
            addtime([output_df.loc[index - 1, 'Time to Produce'].split(',')[-1].strip(), Die_Change_time]).split(
                'day')[-1].strip()
            timeList = [output_df.loc[index - 1, 'Time to Produce'].split(',')[-1].strip(), Die_Change_time,
                        rows['Time to Produce'].split(',')[-1].strip()]
            output_df.loc[index, 'Time to Produce'] = addtime(timeList).split('day')[-1].strip()
            output_df.loc[index, 'Plan Time'] = die_switch_time + " - " + addtime(timeList)
    output_df['Time to Produce'] = preserve_production_time
    return output_df

def get_result():
    data_dir = "productionPlanning/media/"
    for file in os.listdir(os.path.join(BASE_DIR,data_dir)):
        if "J1" in file:
            j1_cric_file = os.path.join(BASE_DIR,data_dir+file)
        elif "J3" in file:
            j3_cric_file = os.path.join(BASE_DIR,data_dir+file)
        else:
            loading_file = os.path.join(BASE_DIR,data_dir+file)
    print(j1_cric_file,"---",j3_cric_file,"----",loading_file)
    Initial_time = "08:00:00"

    output_df = get_plan(0, loading_file, j1_cric_file, j3_cric_file, Initial_time, 'CRITICAL 21.12')

    output_df.rename(columns={'BAL': 'Quantity'}, inplace=True)
    timeList = [output_df['Plan Time'].values.tolist()[-1].split('-')[-1].split(',')[-1].strip(), '00:10:00']
    Initial_time = addtime(timeList).split('day')[-1].strip()
    return output_df[['Part No.', 'Quantity', 'I-802', 'I-406', 'I-407', 'I-408', 'Plan Time']] .to_json()