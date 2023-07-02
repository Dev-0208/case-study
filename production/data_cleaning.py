"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning
)
from scripts import binned_selling_price, process_sales_data


@register_processor("data-cleaning", "sales_data")
def clean_sales_table(context, params):
    """Clean the ``PRODUCT`` data table.

    The table contains information on the inventory being sold. This
    includes information on inventory id, properties of the item and
    so on.
    """

    input_dataset = "raw/sales_data"
    output_dataset = "cleaned/sales_data"

    # load dataset
    sales_data_df = load_dataset(context, input_dataset)

    sales_data_clean = (
        sales_data_df
        .copy()
        .to_datetime("system_calendar_key_N", format="%Y%m%d")
        .rename(columns={'system_calendar_key_N': 'date'})
        .replace({'': np.NaN})
        .clean_names(case_type='snake')
)
    

    # save the dataset
    save_dataset(context, sales_data_clean, output_dataset)

    return sales_data_clean




@register_processor("data-cleaning", "social_media_data")
def clean_social_media_data_table(context, params):
    """Clean the ``ORDER`` data table.

    The table containts the sales data and has information on the invoice,
    the item purchased, the price etc.
    """

    input_dataset = "raw/social_media_data"
    output_dataset = "cleaned/social_media_data"

    # load dataset
    social_media_data = load_dataset(context, input_dataset)

    social_media_data_clean = (
        social_media_data.copy()
        .assign(published_date=pd.to_datetime(social_media_data['published_date']))
        .replace({'': np.NaN})
        .drop_duplicates(subset=['Theme Id', 'published_date', 'total_post'], keep='first')
        .clean_names(case_type='snake')
        .assign(year=lambda x: x['published_date'].dt.year,
            week=lambda x: x['published_date'].dt.isocalendar().week,
            month=lambda x: x['published_date'].dt.month)
        .rename(columns={'published_date': 'date'})
        .reset_index(drop=True)
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
        .drop(columns=['date'])
        .reset_index(drop=True)
        .groupby(['theme_id', 'year', 'week'])
        ['total_post']
        .sum()
        .reset_index()
)


    # save dataset
    save_dataset(context, social_media_data_clean, output_dataset)
    return social_media_data_clean


@register_processor("data-cleaning", "google_search_data")
def clean_google_search_data_table(context, params):
    """Clean the ``ORDER`` data table.

    The table containts the sales data and has information on the invoice,
    the item purchased, the price etc.
    """

    input_dataset = "raw/google_search_data"
    output_dataset = "cleaned/google_search_data"

    # load dataset
    google_search_data = load_dataset(context, input_dataset)

    str_cols = ['platform']

    google_search_data['date']=pd.to_datetime(google_search_data['date'], format="%d-%m-%Y")
    
    google_search_data_clean=(   
       google_search_data

       .copy()
       .rename(columns= {'Claim_ID' : 'theme_id', 'week_number':'week', 'year_new' : 'year'})
       .assign(month=pd.to_datetime(google_search_data['date']).dt.month)
        #.groupby(['theme_id','platform', 'year', 'month', 'week'])
        #.agg(search_volume=('searchVolume', 'sum'))
       .transform_columns(str_cols, string_cleaning, elementwise=False)
       .replace({'': np.NaN})
    
       # clean column names (comment out this line while cleaning data above)
       .clean_names(case_type='snake')
       .reset_index(drop=True)
)
    google_search_data_clean = (
        google_search_data_clean
        .drop(google_search_data_clean[(google_search_data_clean['year'] == 2016) & (google_search_data_clean['week'] == 53)].index)
        .reset_index(drop=True)
        .drop(columns=['date'])
        .reset_index(drop=True)
        .drop_duplicates()
        .reset_index(drop=True)
        .groupby(['theme_id', 'year', 'week', 'platform'])
        ['search_volume']
        .sum()
        .reset_index()
)


    # save dataset
    save_dataset(context, google_search_data_clean, output_dataset)
    return google_search_data_clean


@register_processor("data-cleaning", "product_manufacturer_list")
def clean_product_manufacturer_list_table(context, params):
    """Clean the ``ORDER`` data table.

    The table containts the sales data and has information on the invoice,
    the item purchased, the price etc.
    """

    input_dataset = "raw/product_manufacturer_list"
    output_dataset = "cleaned/product_manufacturer_list"

    # load dataset
    product_manufacturer_list = load_dataset(context, input_dataset)

    str_cols = list(
    set(product_manufacturer_list.select_dtypes("object").columns.to_list())
    - set(
        ['PRODUCT_ID', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4',
       'Unnamed: 5', 'Unnamed: 6']
    )
)

    cols_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4',
       'Unnamed: 5', 'Unnamed: 6']
                
    product_manufacturer_list_clean = (
        product_manufacturer_list
   
        .copy()
    
        .transform_columns(str_cols, string_cleaning, elementwise=False)
    
        .replace({'': np.NaN})
    
        # drop unnecessary cols : nothing to do here
        .drop(columns = cols_to_drop, axis = 1)
    
        # ensure that the key column does not have duplicate records
        .remove_duplicate_rows(col_names=['PRODUCT_ID', 'Vendor'], keep_first=True)
    
        # clean column names (comment out this line while cleaning data above)
        .clean_names(case_type='snake')
)



    # save dataset
    save_dataset(context, product_manufacturer_list_clean, output_dataset)
    return product_manufacturer_list_clean


@register_processor("data-cleaning", "Theme_list")
def clean_Theme_list_table(context, params):
    """Clean the ``ORDER`` data table.

    The table containts the sales data and has information on the invoice,
    the item purchased, the price etc.
    """

    input_dataset = "raw/Theme_list"
    output_dataset = "cleaned/Theme_list"

    # load dataset
    Theme_list = load_dataset(context, input_dataset)

    str_cols = list(
    set(Theme_list.select_dtypes("object").columns.to_list())
    - set(
        ['CLAIM_ID']
    )
)
                
    Theme_list_clean = (
        Theme_list

        .copy()

        .transform_columns(str_cols, string_cleaning, elementwise=False)
    
        .replace({'': np.NaN})
    
        .rename(columns = {'CLAIM_ID' : 'theme_id'})
    
        .rename(columns = {'Claim Name' : 'theme_name'})
    
        .clean_names(case_type='snake')
)
    # save dataset
    save_dataset(context, Theme_list_clean, output_dataset)
    return Theme_list_clean

@register_processor("data-cleaning", "Theme_product_list")
def clean_Theme_product_list_table(context, params):
    """Clean the ``ORDER`` data table.

    The table containts the sales data and has information on the invoice,
    the item purchased, the price etc.
    """

    input_dataset = "raw/Theme_product_list"
    output_dataset = "cleaned/Theme_product_list"

    # load dataset
    Theme_product_list = load_dataset(context, input_dataset)

    Theme_product_list_clean = (
        Theme_product_list
    
        .replace({'': np.NaN})
    
        .rename(columns = {'CLAIM_ID' : 'theme_id'})
    
        .clean_names(case_type='snake')
)

    # save dataset
    save_dataset(context, Theme_product_list_clean, output_dataset)
    return Theme_product_list_clean


@register_processor("data-cleaning", "sales")
def clean_sales_table(context, params):
    """Clean the ``SALES`` data table.

    The table is a summary table obtained by doing a ``inner`` join of the
    ``PRODUCT`` and ``ORDERS`` tables.
    """
    input_sales_ds = "cleaned/sales_data"
    input_social_ds = "cleaned/social_media_data"
    input_google_ds = "cleaned/google_search_data"
    input_product_ds = "cleaned/product_manufacturer_list"
    input_theme_list_ds = "cleaned/Theme_list"
    input_theme_product_ds = "cleaned/Theme_product_list"
    output_dataset = "cleaned/sales"

    # load datasets
    sales_df = load_dataset(context, input_sales_ds)
    social_df = load_dataset(context, input_social_ds)
    google_df = load_dataset(context, input_google_ds)
    product_df = load_dataset(context, input_product_ds)
    theme_df = load_dataset(context, input_theme_list_ds)
    theme_product_df = load_dataset(context, input_theme_product_ds)

    sales_df = pd.merge(sales_df, product_df, how='inner', on='product_id')

    sales_df = sales_df.assign(
    year=sales_df['date'].dt.year,
    #month=sales_df['system_calendar_key_N'].dt.month,
    week=sales_df['date'].dt.isocalendar().week
    #quarter=sales_df['system_calendar_key_N'].dt.quarter
)

    multiple_themes_df = (
         theme_product_df.groupby('product_id')['theme_id']
        .nunique()
        .reset_index(name='theme_count')
        .query('theme_count > 1')
        .merge(theme_product_df, on='product_id')
        .drop_duplicates(subset ='product_id', keep='last').reset_index(drop=True)
)

    single_theme_df = (
        theme_product_df.groupby('product_id')['theme_id']
        .nunique()
        .reset_index(name='theme_count')
        .query('theme_count == 1')
        .merge(theme_product_df, on='product_id')
)
    
    single_product_theme = pd.concat([multiple_themes_df, single_theme_df], ignore_index= True)
    single_product_theme = single_product_theme.drop(columns = 'theme_count')

    sales_df1 = pd.merge(sales_df, single_product_theme, how='inner', on='product_id')

    df1= (pd.merge(social_df, google_df, 
                   on=['theme_id','year','week'],how='inner'))
    
    theme_id = 151
    columns_to_drop = [('sales_units_value', 'B',), 
                       ('sales_units_value', 'F',), 
                     ('sales_units_value', 'Others',),('sales_units_value', 'Private Label',)]
    column_rename = ['year', 'month', 'week', 'client_price_per_pound','competitor_B_price_per_pound',
                          'competitor_F_price_per_pound', 'competitor_others_price_per_pound', 'competitor_private_label_price_per_pound', 
                          'sales_units_value_A','seasonal_A','seasonal_B','seasonal_F','seasonal_others','seasonal_private']

    sodium = process_sales_data(theme_id, columns_to_drop, column_rename, df1, sales_df1)

    sodium.rename(columns={'search_volume':'google_platform'},inplace=True)
    search_shift=1
    social_shift=7
    sodium['total_post'] = sodium['total_post'].shift(social_shift)
    sodium['google_platform'] = sodium['google_platform'].shift(search_shift)
    df = sodium.iloc[social_shift:, :].reset_index(drop=True).copy()
    df.dropna(inplace=True, axis=1)

    sales = df.copy()




    save_dataset(context, sales, output_dataset)
    return sales


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``SALES`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/sales"
    output_train_features = "train/sales/features"
    output_train_target = "train/sales/target"
    output_test_features = "test/sales/features"
    output_test_target = "test/sales/target"
    
    # load dataset
    sales_df_processed = load_dataset(context, input_dataset)

    
    # split the data
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    sales_df_train, sales_df_test = custom_train_test_split(
        sales_df_processed, splitter, by=binned_selling_price
    )

    # split train dataset into features and target
    target_col = params["target"]
    train_X, train_y = (
        sales_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        sales_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
