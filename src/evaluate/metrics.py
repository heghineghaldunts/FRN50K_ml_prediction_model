import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ForecastingMetrics:
    """
    Comprehensive evaluation metrics for demand forecasting models.
    
    Core metrics:
    - MAE, RMSE, MAPE, sMAPE, Bias, Bias %
    - MASE (if training data is available)
    
    Business-relevant metrics:
    - High vs low volume products
    - Peak vs off-peak hours
    - Stockout vs in-stock periods
    
    Residual analysis helpers:
    - Add residuals, absolute errors, and percentage errors to DataFrame
    """
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error - average absolute difference between predictions and actual values.
        
        Teaching insight: MAE is easy to interpret (same units as target variable)
        and robust to outliers. Good for understanding typical prediction errors.
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error - penalizes large errors more than small ones.
        
        Teaching insight: RMSE is more sensitive to outliers than MAE.
        Use when large errors are particularly problematic for your business.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Mean Absolute Percentage Error - percentage-based error metric.
        
        Teaching insight: MAPE is scale-independent, making it good for comparing
        across different products or stores. However, it can be problematic when
        actual values are close to zero (division by zero issue).
        """
        # Add small epsilon to avoid division by zero
        denominator = np.maximum(np.abs(y_true), epsilon)
        return np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """
        Symmetric MAPE - addresses some issues with traditional MAPE.
        
        Teaching insight: sMAPE treats over-forecasting and under-forecasting
        more symmetrically than regular MAPE.
        """
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(numerator / denominator) * 100
    
    @staticmethod
    def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Bias - measures systematic over or under-prediction.
        
        Teaching insight: Bias is crucial in retail forecasting. Consistent
        over-forecasting leads to excess inventory; under-forecasting leads to stockouts.
        """
        return np.mean(y_pred - y_true)
    
    @staticmethod
    def bias_percentage(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Bias as a percentage of actual values."""
        mean_actual = np.mean(y_true) + epsilon
        return (np.mean(y_pred - y_true) / mean_actual) * 100
    
    @staticmethod
    def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 1) -> float:
        """
        Mean Absolute Scaled Error - compares model performance to naive forecast.
        
        Teaching insight: MASE is scale-independent and compares your model
        to a simple baseline. Values < 1 mean your model beats the naive forecast.
        """
        if len(y_train) <= seasonal_period:
            # Fallback to simple naive forecast if insufficient training data
            naive_forecast = np.full_like(y_true, np.mean(y_train))
            mae_naive = ForecastingMetrics.mean_absolute_error(y_true, naive_forecast)
        else:
            # Use seasonal naive forecast
            mae_naive = np.mean(np.abs(np.diff(y_train[-seasonal_period:])))
        
        mae_model = ForecastingMetrics.mean_absolute_error(y_true, y_pred)
        
        return mae_model / (mae_naive + 1e-8)  # Avoid division by zero
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics at once.
        
        This is the main function students will use to evaluate their models.
        """
        metrics = {
            'MAE': ForecastingMetrics.mean_absolute_error(y_true, y_pred),
            'RMSE': ForecastingMetrics.root_mean_squared_error(y_true, y_pred),
            'MAPE': ForecastingMetrics.mean_absolute_percentage_error(y_true, y_pred),
            'sMAPE': ForecastingMetrics.symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'Bias': ForecastingMetrics.bias(y_true, y_pred),
            'Bias_Percentage': ForecastingMetrics.bias_percentage(y_true, y_pred)
        }
        
        # Add MASE if training data is available
        if y_train is not None:
            metrics['MASE'] = ForecastingMetrics.mean_absolute_scaled_error(y_true, y_pred, y_train)
        
        return metrics
    
    @staticmethod
    def evaluate_by_group(df: pd.DataFrame, y_true_col: str, y_pred_col: str, 
                         group_cols: List[str], y_train: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Evaluate model performance by different groupings (store, product, etc.).
        
        Teaching value: Understanding where your model performs well or poorly
        is crucial for model improvement and business insights.
        """
        results = []
        
        for group_values, group_data in df.groupby(group_cols):
            y_true_group = group_data[y_true_col].values
            y_pred_group = group_data[y_pred_col].values
            
            # Calculate metrics for this group
            group_metrics = ForecastingMetrics.calculate_all_metrics(y_true_group, y_pred_group, y_train)
            
            # Add group identifiers
            group_result = dict(zip(group_cols, group_values if isinstance(group_values, tuple) else [group_values]))
            group_result.update(group_metrics)
            group_result['n_observations'] = len(group_data)
            
            results.append(group_result)
        
        return pd.DataFrame(results)
    
    # Example helpers for business use-cases
    @staticmethod
    def high_low_volume(df: pd.DataFrame, volume_col: str, threshold: float):
        """Classify products as high- or low-volume based on a threshold"""
        if threshold is None:
            threshold = df[volume_col].median()
        df['volume_category'] = df[volume_col].apply(lambda x: 'high' if x >= threshold else 'low')
        return df
    
    @staticmethod
    def evaluate_by_time_period(df: pd.DataFrame, y_true_col: str, y_pred_col: str, hour_col: str, peak_hours: List[int], y_train: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Evaluate performance during peak vs off-peak hours"""
        df['time_period'] = df[hour_col].apply(lambda h: 'peak' if h in peak_hours else 'off_peak')
        return ForecastingMetrics.evaluate_by_group(df, y_true_col, y_pred_col, ['time_period'], y_train)

    @staticmethod
    def evaluate_by_stock_status(df: pd.DataFrame, y_true_col: str, y_pred_col: str, stock_col: str, y_train: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Evaluate performance during stockout vs in-stock periods"""
        df['stock_status'] = df[stock_col].apply(lambda x: 'in_stock' if x > 0 else 'stockout')
        return ForecastingMetrics.evaluate_by_group(df, y_true_col, y_pred_col, ['stock_status'], y_train)

    # Residual / Error Analysis
    @staticmethod
    def add_residuals(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
        df = df.copy()
        df['residual'] = df[y_pred_col] - df[y_true_col]
        df['abs_error'] = np.abs(df['residual'])
        df['pct_error'] = np.abs(df['residual'] / np.maximum(np.abs(df[y_true_col]), 1e-8)) * 100
        return df

    @staticmethod
    def compare_models(df: pd.DataFrame, y_true_col: str, y_pred_cols: List[str], y_train: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Compare multiple models side by side"""
        comparison = []
        for col in y_pred_cols:
            metrics = ForecastingMetrics.calculate_all_metrics(df[y_true_col].values, df[col].values, y_train)
            metrics['model'] = col
            comparison.append(metrics)
        return pd.DataFrame(comparison).sort_values('MAE')
    
    def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Analyze residuals for systematic bias or errors.
        """
        residuals = y_pred - y_true
        return pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'residual': residuals
        })
    
    @staticmethod
    def evaluate_volume_groups(df: pd.DataFrame, y_true_col: str, y_pred_col: str, volume_threshold: float):
        """Separate high-volume vs low-volume products"""
        df['volume_group'] = df[y_true_col].apply(lambda x: 'high' if x >= volume_threshold else 'low')
        results = {}
        for group in ['high', 'low']:
            group_data = df[df['volume_group'] == group]
            results[group] = {
                'MAE': ForecastingMetrics.MAE(group_data[y_true_col].values, group_data[y_pred_col].values),
                'Bias': ForecastingMetrics.Bias(group_data[y_true_col].values, group_data[y_pred_col].values)
            }
        return results

    @staticmethod
    def evaluate_peak_hours(df: pd.DataFrame, y_true_col: str, y_pred_col: str, peak_hours: List[int]):
        """Evaluate model during peak vs off-peak hours"""
        df['hour_group'] = df['hour'].apply(lambda x: 'peak' if x in peak_hours else 'offpeak')
        results = {}
        for group in ['peak', 'offpeak']:
            group_data = df[df['hour_group'] == group]
            results[group] = {
                'MAE': ForecastingMetrics.MAE(group_data[y_true_col].values, group_data[y_pred_col].values),
                'Bias': ForecastingMetrics.Bias(group_data[y_true_col].values, group_data[y_pred_col].values)
            }
        return results

    @staticmethod
    def evaluate_stockout(df: pd.DataFrame, y_true_col: str, y_pred_col: str, stock_col: str):
        """Evaluate performance on stockout vs in-stock periods"""
        df['stock_group'] = df[stock_col].apply(lambda x: 'stockout' if x == 0 else 'in-stock')
        results = {}
        for group in ['stockout', 'in-stock']:
            group_data = df[df['stock_group'] == group]
            results[group] = {
                'MAE': ForecastingMetrics.MAE(group_data[y_true_col].values, group_data[y_pred_col].values),
                'Bias': ForecastingMetrics.Bias(group_data[y_true_col].values, group_data[y_pred_col].values)
            }
        return results

    
    