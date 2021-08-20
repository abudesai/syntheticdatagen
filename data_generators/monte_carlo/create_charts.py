
import numpy as np, pandas as pd
import random
import matplotlib.pyplot as plt
import math
import variables as vars
from datetime import datetime, timedelta



def plot_daily_level_data(max_num_series=5, max_days = 1000, node_type='org_id', file_dir = vars.output_dir):
    
    daily_data = pd.read_csv(f'{file_dir+vars.daily_history_file_prefix}.csv', parse_dates=['date'])

    groupby_col = [node_type, 'date'] 
    act_cols = [c for c in daily_data.columns if c.startswith('Act')]
    val_col = act_cols[0]
    agg_data = daily_data.groupby(groupby_col, as_index=False)[[val_col]].sum()
    agg_data = agg_data.sort_values(by=[node_type, 'date'])

    unique_series = agg_data[node_type].unique()
    if max_num_series < len(unique_series):
        unique_series = random.sample(list(unique_series), max_num_series)

    max_date = agg_data['date'].max()
    unique_dates = agg_data['date'].unique()
    unique_dates = [d for d in unique_dates if d >= max_date + timedelta(days=-max_days)]

    agg_data = agg_data[ (agg_data[node_type].isin(unique_series)) 
        & (agg_data['date'].isin(unique_dates) ) ]


    num_rows = len(unique_series)
    fig, axs = plt.subplots(nrows=num_rows, ncols=1, figsize = (14, 1+2*num_rows), 
        sharex=False, sharey=False)
    
    for i, s in enumerate(unique_series):
        df = agg_data[ agg_data[node_type] == s]

        
        ax = axs[i] if num_rows > 1 else axs

        ax.plot(df['date'], df[val_col])  

        ax.set_title(s)    
        # for tick in ax.get_xticklabels(): tick.set_rotation(45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))

        y_max = df[val_col].max()
        ax.set_ylim(bottom=0., top = y_max*1.1)

    plt.subplots_adjust(top = 0.90, bottom=0.1, hspace=.4, wspace=0.4)
    plt.show()
    # fig.tight_layout()
    fig.savefig(f'{file_dir + vars.daily_history_file_prefix}_chart.png')
    return 




if __name__ == '__main__':
    plot_daily_level_data(max_num_series=5, max_days = 365)