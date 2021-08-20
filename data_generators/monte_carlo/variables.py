
bucket = 'verint-sagemaker-data-gen-project'
data_dir = 'abu_synthetic'

full_dir_path = f"s3://{bucket}/{data_dir}"

# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# file directories
input_dir = './config/'

output_dir = './outputs/'
output_dir_multi = './outputs_multi/'

split_files_output_dir = './outputs/split_files/'
parquet_files_output_dir = './outputs/parquet_files/'

monthly_history_file_prefix = 'history_monthly'
daily_history_file_prefix = 'history_daily'
processed_daily_history_file_prefix = 'processed_history_daily'
intra_daily_history_file_prefix = 'history_intra_daily'

irreg_effects_history_file_prefix = 'irreg_effects'

internal_intra_daily_parquet_file_name = 'Internal_intra_daily_history'
internal_daily_parquet_file_name ='Internal_daily_history'

# -------------------------------------------------------------------------------
# field labels 
org_id = 'org_id'
act_id = 'act_id'
date_field = 'date'
value = 'value'


# -------------------------------------------------------------------------------


data_gen_file_name = 'data_gen_params_v2.xlsx'


config_params = [
        {
            'name': 'main_params',
            'orientation': 'records',
        },
        {
            'name': 'hierarchy_params',
            'orientation': 'records',
        },
        {
            'name': 'noise',
            'orientation': 'records',
        },
        {
            'name': 'magnitude',
            'orientation': 'records',
        },
        {
            'name': 'weekly_open_hours',
            'orientation': 'records',
        },
        {
            'name': 'special_days',
            'orientation': 'records',
        },
        {
            'name': 'special_days_dates',
            'orientation': 'records',
        },
        {
            'name': 'regular_effects',
            'orientation': 'records',
        },
        {
            'name': 'irregular_effects',
            'orientation': 'records',
        },
    ]

config_file_name = 'config.json'


ts_regular_effect_types = {
    'Level': {'len': 1}, 
    'Trend_Linear': {'len': 1}, 
    'Trend_Multiplicative': {'len': 1}, 
    'Month_of_the_year': {'len': 12}, 
    'Day_of_the_week': {'len': 7}, 
    'Day_of_the_month': {'len': 31}, 
    'Time_of_the_day': {'len': None}, 
    'Special_Days': {'len': 1}, 
}

# -------------------------------------------------------------------------------

write_batch_size = 1



DEBUG = True