import numpy as np, pandas as pd
import sys, os, shutil
from datetime import datetime
import matplotlib.pyplot as plt
import loss_functions as lf


def write_all_predictions(all_predictions, save_path):
    all_predictions.to_csv(f"{save_path}all_predictions.csv", index=False)

def aggregate_predictions(all_predictions):
    avg_pred = all_predictions.groupby(['queueid', 'date'], as_index=False)[['act_value', 'pred_value']].mean() 
    return avg_pred


def save_predictions(test_data, save_path, filter_series=None):   
    test_data.to_csv(f"{save_path}test_predictions.csv", index=False)

    if filter_series is not None: 
        test_data2 = test_data[test_data['seriesid'].isin(filter_series)]
        test_data2.to_csv(f"{save_path}filtered_test_predictions.csv", index=False)



def save_actual_vs_pred_chart(test_data, save_path):   

    daily_vols = test_data.groupby(['date'], as_index=False)[['act_value', 'pred_value']].sum()
    daily_vols.rename(columns={'act_value_sum': 'act_value', 'pred_value_sum': 'pred_value'}, inplace=True)

    if not os.path.exists(save_path): os.makedirs(save_path)

    file_path_and_name = f'{save_path}act_vs_fcst_values.png'
    # print(file_path_and_name)

    fig = plt.figure(figsize=(12, 6 ))
    plt.tight_layout()
    plt.plot(daily_vols['date'], daily_vols['act_value'], label="act_value", linewidth=5, color='green')
    plt.plot(daily_vols['date'], daily_vols['pred_value'], label="pred_value", color='red', linewidth=2)
    plt.ylim(bottom = 0.)
    plt.legend()
    plt.savefig(file_path_and_name)
    plt.close(fig)



def write_wape_by_group(model_name, dataset_name, test_data, save_path):
    df = test_data[['seriesid', 'act_value', 'pred_value']].copy()
    df['abs_error'] = np.abs(df['act_value'] - df['pred_value'])
    df2 = df.groupby(by=['seriesid'], as_index=False)[['act_value', 'pred_value', 'abs_error']].sum()
    df2['wape'] = np.nan
    idx = df2['act_value'] > 0
    df2.loc[idx, 'wape'] = df2.loc[idx, 'abs_error'] / df2.loc[idx, 'act_value']
    df2.insert(0, 'model_name', model_name)
    df2.insert(1, 'dataset_name', dataset_name)
    df2.sort_values(by='wape', ascending=False, inplace=True)
    df2.to_excel(f"{save_path}wape_by_series.xlsx", index=False) 


def write_perform_metrics_by_queue(model_name, dataset_name, test_data, save_path):    
    unique_qs = test_data['seriesid'].unique().tolist()

    results = []
    for q in unique_qs:
        data = test_data[test_data.seriesid == q]
        results_df = get_performance_metrics_df(model_name, dataset_name, data)  
        results_df.insert(3, 'seriesid', q)
        results.append(results_df)  
    
    results = pd.concat(results)
    results.to_excel(f"{save_path}perf_metrics_by_queue.xlsx", index=False) 


def get_performance_metrics_df(model_name, dataset_name, test_data):
    act_vals = test_data['act_value'].values
    pred_vals = test_data['pred_value'].values

    mse = lf.get_mse(act_vals, pred_vals)
    mape = lf.get_mape(act_vals, pred_vals)
    smape = lf.get_smape(act_vals, pred_vals)
    wape = lf.get_wape(act_vals, pred_vals)
    r_squared = lf.get_r_squared(act_vals, pred_vals)

    perf_metrics = [ [
                model_name, 
                dataset_name,
                datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
                np.round(mse, 3), 
                np.round(np.sqrt(mse), 3),
                np.round(mape, 3),
                np.round(smape, 3),
                np.round(wape, 3),
                np.round(r_squared, 3) 
                 ]
    ]

    columns=[
        'Model_Name',
        'Dataset_Name',
        'Time_Ran',
        'MSE',
        'RMSE',
        'MAPE',
        'sMAPE',
        'WAPE',
        'R-squared'
    ]
    # convert to dataframe
    perf_metrics = pd.DataFrame(perf_metrics, columns=columns)    
    return perf_metrics


def write_performance_metrics(model_name, dataset_name, test_data, save_path): 
    perf_metrics = get_performance_metrics_df(model_name, dataset_name, test_data)    
    print(perf_metrics)
    perf_metrics.to_excel(f"{save_path}perf_metrics.xlsx", index=False) 