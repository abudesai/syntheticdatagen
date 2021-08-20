import numpy as np, pandas as pd
from datetime import datetime, date, time, timedelta
import sys
from calendar import monthrange
import random
import utils


def generate_time_scale(data_gen_params, random_seed):
    random.seed(int(random_seed))
    history_dates_df = get_history_dates(data_gen_params['main_params'])    
    history_dates_df = add_seasonality_features(history_dates_df, data_gen_params)    
    history_dates_df, special_day_ids = add_special_days(history_dates_df, data_gen_params)    
    history_dates_df = add_day_num_for_trend(history_dates_df)    
    
    return history_dates_df, special_day_ids


def add_day_num_for_trend(history_dates_df):
    history_dates_df['cumu_day_num'] = np.arange(len(history_dates_df))
    return history_dates_df


def add_special_days(history_dates_df, data_gen_params):
    special_days = data_gen_params['special_days']
    special_days_dates = data_gen_params['special_days_dates']
    special_days_dates.rename(
        columns={ 'Special_Day_Name': 'special_day', 'Special_Day_Date': 'date', }, 
        inplace=True
    )

    history_dates_df = history_dates_df.merge(special_days_dates, how='left', on='date')
    history_dates_df['is_holiday'] = 0
    special_day_ids = {}
    day_id = 1
    for _, row in special_days.iterrows():
        special_day = row['Special_Day_Name']
        days_before = row['Num_Days_Before']
        days_after = row['Num_Days_After']

        if special_day_ids.get(special_day) is None: special_day_ids[special_day] = day_id
        day_id += 1

        # flag if holiday
        idx = history_dates_df['special_day'] == special_day
        history_dates_df.loc[idx, 'is_holiday'] = row['Is_Holiday']

        # mark the days before and after the special day. we will apply synthetic multipliers to them
        special_day_rows =  history_dates_df.index[idx].tolist()
        for special_day_row in special_day_rows: 
            for d in range(days_before):
                if special_day_row - (d + 1) < 0: continue
                day_name = f'{special_day} - {(d + 1)}'
                history_dates_df.loc[special_day_row - (d + 1), 'special_day'] =  day_name
                if special_day_ids.get(day_name) is None: special_day_ids[day_name] = day_id
                day_id += 1

            for d in range(days_after):
                if special_day_row + (d + 1) > len(history_dates_df): continue
                day_name = f'{special_day} + {(d + 1)}'  
                history_dates_df.loc[special_day_row + (d + 1), 'special_day'] = day_name  
                if special_day_ids.get(day_name) is None: special_day_ids[day_name] = day_id
                day_id += 1   
    
    special_day_ids = [ {'special_day': k, "day_id": v} for k,v in special_day_ids.items()]

    # # add regular day 
    # special_day_ids.insert(0, {'special_day': 'none', "day_id": 0})
    special_day_ids = pd.DataFrame(special_day_ids)

    # mark all empty cells as regular days
    history_dates_df.fillna(value = {'special_day': 'none'}, inplace=True)
    return history_dates_df, special_day_ids


def mapped_day_of_month(d, threshold_day):
    '''
    we will use updated mapping of day in month. 
    Up to threshold_day, the day number is unchanged. 
    After threshold_day, we equate days to count from backwards from end of month, and equate to a 31 day month. 
    For example, day 30 in April will map to 31.  Similarly, day 28 in non-leap February will also map to 31. 
    Why do we do this? 
    We want to represent the actual seasonality in month. 
    The model should be able to recognize that the last day in March is 31, and last day in April is 30. 
    Without this, the model can't represent the fact that day 30 is the second to last day in March, but last day in April.
    In many retail environments, there is a big difference in customer transactions on specific days at the 
    beginning and end of month. Last day is not the same as second to last day.
    '''
    total_days_in_month = monthrange(d.year, d.month)[1]
    day_num = int(d.strftime("%d"))
    if day_num <= threshold_day: return day_num
    return int(31 - (total_days_in_month - day_num))


def add_seasonality_features(history_dates_df, data_gen_params):
    seasonality_rows = data_gen_params['regular_effects']['Effect_Type'] == 'Seasonality'
    seasonalities = data_gen_params['regular_effects'].loc[seasonality_rows]
    threshold_day = data_gen_params['main_params']['Day_in_month_threshold_day']
    # print(threshold_day); sys.exit()

    for i, row in seasonalities.iterrows():
        if row['Effect_Name'] == 'Month_of_the_year':
            history_dates_df['month'] = history_dates_df['date'].apply(lambda d: d.month)
        elif row['Effect_Name'] == 'Day_of_the_week':
            history_dates_df['weekday'] = history_dates_df['date'].apply(lambda d: (d.weekday() + 1) )
        elif row['Effect_Name'] == 'Day_of_the_month':
            history_dates_df['adj_day_in_month'] = history_dates_df['date'].apply(
                lambda d: mapped_day_of_month(d, threshold_day))
        elif row['Effect_Name'] == 'Time_of_the_day':
            # we are only creating daily level time-scale... time-of-day is handled elsewhere
            # why? because we don't want to create all-possible intra-daily time intervals in each day. 
            # we will later create time-intervals specific to open hours of each location.
            pass        
        else:
            raise Exception(f"Sorry, couldn't recognize {row['Effect_Name'] } as Seasonality_Type")

    return history_dates_df



def get_history_dates(main_params):
    # determine the history start date; chosen unif. randomly from given earliest start to latest start
    earliest_start_date = utils.parse_date_from_string(main_params['Earliest_history_start_date'])
    latest_start_date = utils.parse_date_from_string(main_params['Latest_history_start_date'] )
    num_days_in_range = (latest_start_date - earliest_start_date).days + 1

    history_start_date = earliest_start_date + timedelta(days=random.randint(0, num_days_in_range-1)) 
    # print(earliest_start_date, latest_start_date, history_start_date)

    # determine the history length; chosen unif. randomly from allowable history length range in config params
    num_days_in_history = random.randint(int(main_params['Lowest_num_days_in_history']), 
            int(main_params['Highest_num_days_in_history']))
    # print('num_days_in_history', num_days_in_history, history_start_date)

    # create the historical dates df
    history_dates = [history_start_date + timedelta(days=x) for x in range(num_days_in_history)]
    history_dates_df = pd.DataFrame(data = history_dates, columns = ['date'])

    history_dates_df['year'] = history_dates_df['date'].apply(lambda d: d.year)
    history_dates_df['month'] = history_dates_df['date'].apply(lambda d: d.month)
    # print(history_dates_df.shape, history_dates_df.head())
    # sys.exit()

    return history_dates_df


def get_history_months(history_dates_df):
    dates_df = history_dates_df.copy()
    if not 'year' in dates_df.columns: 
        dates_df['year'] = history_dates_df['date'].apply(lambda d: d.year)
    if not 'month' in dates_df.columns: 
        dates_df['month'] = history_dates_df['date'].apply(lambda d: d.month)    

    dates_df['yrmt'] = dates_df['year'] * 100 + dates_df['month']
    months_df = dates_df[['yrmt', 'year', 'month']].drop_duplicates().reset_index(drop=True)
    months_df.sort_values(by='yrmt', inplace=True)
    # months_df.drop(columns=['yrmt'], inplace=True)
    months_df['cumu_month_num'] = np.arange(len(months_df))
    return months_df


def get_num_time_of_day_intervals(time_granularity_in_minutes):
    return int(24 * 60 / int(time_granularity_in_minutes))

def get_num_special_days(history_dates):
    special_days = history_dates['special_day'].drop_duplicates().tolist()
    num_special_days = len(special_days) - 1 if 'none' in special_days else len(special_days) 
    return num_special_days


def time_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta 


def get_daily_time_intervals(time_granularity_in_minutes):
    today = date.today()
    day_start_time = datetime.strptime('00:00', '%H:%M').time() 
    day_end_time = datetime.strptime('23:59', '%H:%M').time() 

    start_date_time = datetime(
        today.year, today.month, today.day, 
        day_start_time.hour, 
        day_start_time.minute)

    end_date_time = datetime(
        today.year, today.month, today.day, 
        day_end_time.hour, 
        day_end_time.minute)
    # print(start_date_time, end_date_time )

    hh_gen = time_range(
            start_date_time, end_date_time,
            timedelta(minutes = time_granularity_in_minutes))

    time_intervals = [t.time() for t in hh_gen] #; print(time_intervals)
    time_intervals = pd.DataFrame(time_intervals, columns=['time_interval'])
    time_intervals.insert(0, 'time_int_num', np.arange(time_intervals.shape[0]))
    # print(time_intervals.head(100)) ; sys.exit()

    return time_intervals