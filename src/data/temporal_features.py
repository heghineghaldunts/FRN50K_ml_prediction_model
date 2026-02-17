import pandas as pd
import numpy as np

class TemporalFeatureEngineer:

    def __init__(self, lag_periods=[1, 7, 168]):
        self.lag_periods = lag_periods

    def create_temporal_features(self, df, datetime_col='dt'):
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['month'] = df[datetime_col].dt.month

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df

    def create_lag_features(self, df, target_col='sale_amount'):
        df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
        for lag in self.lag_periods:
            df[f'{target_col}_lag_{lag}'] = df.groupby(['store_id','product_id'])[target_col].shift(lag)
        return df

    def engineer_all_features(self, df):
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        return df
