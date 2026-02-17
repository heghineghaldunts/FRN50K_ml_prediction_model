# src/models/random_forest_baseline.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import Dict
from ..base_model import BaseForecastingModel
import logging

logger = logging.getLogger(__name__)

class RandomForestForecastingModel(BaseForecastingModel):
    """
    Random Forest Regression Baseline for demand forecasting.
    
    Features:
    - Missing value handling
    - Optional feature scaling
    - Validation metrics
    - Non-negative predictions
    - Compatible with BaseForecastingModel interface
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("RandomForestBaseline", config)
        
        self.n_estimators = config.get('n_estimators', 100) if config else 100
        self.max_depth = config.get('max_depth', 15) if config else 15
        self.scaler_type = config.get('scaler', 'standard') if config else 'standard'
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler() if self.scaler_type == 'standard' else None

    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Handle missing values and optional scaling."""
        X_numeric = X.select_dtypes(include=[np.number])
        self.feature_names = X_numeric.columns.tolist()
        
        # Impute missing values
        if fit:
            X_proc = pd.DataFrame(
                self.imputer.fit_transform(X_numeric),
                columns=self.feature_names,
                index=X_numeric.index
            )
        else:
            X_proc = pd.DataFrame(
                self.imputer.transform(X_numeric),
                columns=self.feature_names,
                index=X_numeric.index
            )
        
        # Optional scaling
        if self.scaler:
            if fit:
                X_scaled = self.scaler.fit_transform(X_proc)
            else:
                X_scaled = self.scaler.transform(X_proc)
        else:
            X_scaled = X_proc.values
        
        return X_scaled

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RandomForestBaselineModel':
        """Fit Random Forest with preprocessing."""
        self.validate_input(X, y)
        
        X_proc = self._preprocess(X, fit=True)
        self.model.fit(X_proc, y)
        self.is_fitted = True
        
        # Training R²
        train_r2 = self.model.score(X_proc, y)
        self.training_history['train_r2'] = train_r2
        self.training_history['n_features'] = X_proc.shape[1]
        self.training_history['n_samples'] = X_proc.shape[0]
        
        logger.info(f"Random Forest training R²: {train_r2:.4f}")
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate non-negative predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_proc = self._preprocess(X, fit=False)
        preds = self.model.predict(X_proc)
        return np.maximum(preds, 0)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances if available."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}
