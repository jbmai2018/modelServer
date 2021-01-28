import datetime
import time
import pandas as pd
import numpy as np
import math
import os
import productionPlanning.dbserver as db

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


def try_except(key):
    try:
        return dict_to_use[key][0]
    except:
        return 0


def get_quantity(row):
    max_quant = max(row['BAL.'], row['Vehicle plan/day'])

    return row['RM packaging standard'] * math.ceil(max_quant / row['RM packaging standard'])


def get_plan(Initial_time):
    critical = 'Critical'

    ## Read operation_SPM_List
    print("Reading Excel files")
    df_SPM = pd.read_excel(os.path.join(BASE_DIR,'productionPlanning/I4_part_SPM_operations.xlsx'))

    ## Reading rm_packaging_fike
    df_rm = pd.read_excel(os.path.join(BASE_DIR,'productionPlanning/RM PACKAGING _new.xlsx'))

    ## Reading critical sheets
    df_j1 = pd.read_json(db.get_data('critical_sheet_j1'),orient='records')
    df_j3 = pd.read_json(db.get_data('critical_sheet_j3'),orient='records')

    cric_j1 = [i for i in df_j1.columns if str(i).strip().lower().startswith('critical')][0]
    cric_j3 = [i for i in df_j3.columns if str(i).strip().lower().startswith('critical')][0]

    print("Renaming Columns {} and {} to {}".format(cric_j1, cric_j3, critical))

    rename_columns = {'Sap code': 'Material Code', 'Description': 'Part No.', cric_j1: critical, cric_j3: critical,
                      "Total": 'Inventory'}

    df_j1.rename(columns=rename_columns, inplace=True)
    df_j3.rename(columns=rename_columns, inplace=True)

    common_parts = [("YCA-46515", "YCA-46516", "YCA-46515/16"), ("Y9T-67121", "Y9T-67421", "Y9T-67121/421"),
                    ("YSD-61316", "YSD-61326", "YSD-61316/326"), ("YJC-61224", "YJC-61225", "YJC-61224/25"),
                    ("YE3-58311", "YE3-58411", "YE3-58311/411")]

    df_j1 = df_j1[df_j1['Material Code'].isin(df_SPM['Material Code'].values.tolist())][
        ['Material Code', 'Part No.', critical, 'Vehicle plan/day', 'BAL.', 'Inventory']]
    df_j3 = df_j3[df_j3['Material Code'].isin(df_SPM['Material Code'].values.tolist())][
        ['Material Code', 'Part No.', critical, 'Vehicle plan/day', 'BAL.', 'Inventory']]

    # Drop duplicate columns
    df_j1 = df_j1.loc[:, ~df_j1.columns.duplicated(keep='last')]
    df_j3 = df_j3.loc[:, ~df_j3.columns.duplicated(keep='last')]

    # Join Critical sheets
    df_cric = pd.concat([df_j1, df_j3])
    aggregation_functions = {'Part No.': 'first', critical: 'last', 'Vehicle plan/day': 'first', 'BAL.': 'sum',
                             'Inventory': 'sum'}
    df_cric = df_cric.groupby(df_cric['Material Code']).aggregate(aggregation_functions)
    df_cric.reset_index(inplace=True)
    df_cric.dropna(subset=['Vehicle plan/day'], axis=0, inplace=True)
    df_cric['days_inventory'] = df_cric['Inventory'] / df_cric['Vehicle plan/day']
    df_cric['days_inventory'] = df_cric['days_inventory'].apply(lambda x: math.floor(x))

    # Merge with SPM sheet
    df_I4_merge = df_cric.merge(df_SPM, on=['Material Code'], how='inner')

    # Calculate Net Balance
    df_I4_merge['BAL.'] = df_I4_merge['BAL.'] - df_I4_merge['Inventory']

    cat_col = df_I4_merge.select_dtypes(include='object').columns.tolist()
    cat_col = {val: 'first' for val in cat_col}

    num_col = df_I4_merge.select_dtypes(exclude='object').columns.tolist()
    num_col = {val: 'sum' for val in num_col}

    cat_col.update(num_col)

    print("Calculating Quantity to Produce")

    cat_col['Material Code'] = 'first'
    cat_col['Vehicle plan/day'] = 'first'
    cat_col['days_inventory'] = 'min'
    cat_col['Inventory'] = 'min'
    cat_col['SPM'] = 'first'
    cat_col['operations'] = 'first'
    cat_col['BAL.'] = 'max'
    df_to_use = df_I4_merge.groupby(df_I4_merge['Part No. To use']).aggregate(cat_col)

    df_to_use.reset_index(drop=True, inplace=True)
    df_to_use.drop(columns=['Part No._x', 'Part No._y'], axis=1, inplace=True)
    # df_to_use.rename(columns={'Part No. To use':'Part No.'},inplace=True)

    # Rearrange columns
    cols = list(df_to_use.columns)
    a, b = cols.index(critical), cols.index('Part No. To use')
    cols[b], cols[a] = cols[a], cols[b]
    df_to_use = df_to_use[cols]

    # Merge with RM Packaging
    df_to_use = df_to_use.merge(df_rm, on=['Material Code'], how='left')
    df_to_use.drop(columns=['Part No.'], axis=1, inplace=True)
    df_to_use.rename(columns={'Part No. To use': 'Part No.'}, inplace=True)

    # Calculate Quantity to produce
    df_to_use['temp_BAL'] = df_to_use.apply(lambda x: get_quantity(x), axis=1)
    df_to_use['BAL.'] = df_to_use['temp_BAL']

    print("Calculating Time to Produce")

    # Calculate Time to produce
    df_to_use['Time to Produce minutes'] = round(df_to_use['BAL.'] / df_to_use['SPM'], 0)
    df_to_use = df_to_use[df_to_use[critical] != 'Line stopped']
    df_to_use = df_to_use[(df_to_use['BAL.'] > 0)]
    df_to_use = df_to_use[(df_to_use['days_inventory'] < 2)]
    df_to_use = df_to_use[(df_to_use['Time to Produce minutes'] > 60)]
    df_to_use['Time to Produce'] = df_to_use['Time to Produce minutes'].apply(
        lambda x: time.strftime('%H:%M:%S', time.gmtime(x * 60)))
    df_to_use = df_to_use.sort_values(by=[critical, 'days_inventory'], ascending=[True, True]).reset_index(drop=True)

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
            addtime([output_df.loc[index - 1, 'Time to Produce'].split(',')[-1].strip(), Die_Change_time]).split('day')[
                -1].strip()
            timeList = [output_df.loc[index - 1, 'Time to Produce'].split(',')[-1].strip(), Die_Change_time,
                        rows['Time to Produce'].split(',')[-1].strip()]
            output_df.loc[index, 'Time to Produce'] = addtime(timeList).split('day')[-1].strip()
            output_df.loc[index, 'Plan Time'] = die_switch_time + " - " + addtime(timeList)
    output_df['Time to Produce'] = preserve_production_time

    output_df.rename(columns={'BAL.': 'Quantity'}, inplace=True)
    return output_df


def get_result():

    Initial_time = "08:00:00"

    output_df = get_plan(Initial_time)


    timeList = [output_df['Plan Time'].values.tolist()[-1].split('-')[-1].split(',')[-1].strip(), '00:10:00']
    Initial_time = addtime(timeList).split('day')[-1].strip()

    #store result in db
    predicted_plan = output_df[['Part No.', 'Quantity', 'I-802', 'I-406', 'I-407', 'I-408', 'Plan Time']].to_json(orient='records')
    db.update_predicted_plan(predicted_plan)
    return predicted_plan
