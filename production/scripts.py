"""Module for listing down additional custom functions required for production."""

import pandas as pd

def binned_selling_price(df):
    """Bin the selling price column using quantiles."""
    return pd.qcut(df["sales_units_value_A"], q=5)


def process_sales_data(theme_id, columns_to_drop, column_rename, df1, sales_df1):
    theme_sodium_sales = sales_df1[sales_df1['theme_id'] == 151].reset_index(drop=True)
    theme_sodium_sales['lbs_per_dollar'] = theme_sodium_sales['sales_lbs_value'] / theme_sodium_sales['sales_dollars_value']
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    ts = theme_sodium_sales['sales_dollars_value']
    result = seasonal_decompose(ts, model='additive', period=12)
    seasonality = result.seasonal
    theme_sodium_sales = pd.concat([theme_sodium_sales, seasonality.rename('seasonal')], axis=1)
    theme_sodium_sales.drop(columns=['sales_dollars_value', 'sales_lbs_value'], inplace=True)
    
    grouped_sales = theme_sodium_sales.groupby(['vendor', 'year',  'week']).agg({
        'lbs_per_dollar': 'sum',
        'sales_units_value': 'sum',
        'seasonal': 'sum'
    }).reset_index()
    
    pivot_table = pd.pivot_table(grouped_sales, values=['lbs_per_dollar', 'sales_units_value', 'seasonal'],
                                 index=['year',  'week'], columns='vendor').reset_index()
    
    pivot_table.drop(columns=columns_to_drop, inplace=True)
    df_pivot_table = pd.DataFrame(pivot_table.to_records())
    df_pivot_table.columns = ['index', 'year',  'week', 'client_price_per_pound',
                              'competitor_B_price_per_pound', 'competitor_F_price_per_pound',
                              'competitor_others_price_per_pound', 'competitor_private_label_price_per_pound',
                              'sales_units_value_A', 'seasonal_A', 'seasonal_B', 'seasonal_F',
                              'seasonal_others', 'seasonal_private']
    
    df_pivot_table.drop(columns='index', inplace=True)
    
    theme_sodium_df1 = df1[df1['theme_id'] == 151].drop(columns=['theme_id']).reset_index(drop=True)
    dummy_df_sodium = pd.get_dummies(theme_sodium_df1['platform'])
    updated_df_sodium = pd.concat([theme_sodium_df1, dummy_df_sodium], axis=1)
    updated_df_sodium.drop(columns='platform', inplace=True)
    
    df = pd.merge(updated_df_sodium, df_pivot_table, on=['year', 'week'], how='inner').reset_index(drop=True)
    df.drop(columns=['google'], inplace=True, axis=1)
    
    return df
