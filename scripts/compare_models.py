
"""
Compare multiple model results from YAML files.

Usage:
    python compare_models.py
"""

import yaml
import numpy as np
from pathlib import Path
import pandas as pd

# List of YAML result files
yaml_files = [
    r'C:\Users\ghald\Documents\FRN50K_ml_prediction_model\models\trained\baseline_ensemble_20260216_235315_20260216_235412_results.yaml',
    r'C:\Users\ghald\Documents\FRN50K_ml_prediction_model\models\trained\baseline_gradient_boosting_20260217_003034_20260217_003107_results.yaml',
    r'C:\Users\ghald\Documents\FRN50K_ml_prediction_model\models\trained\baseline_lasso_20260216_164431_20260216_164510_results.yaml',
    r'C:\Users\ghald\Documents\FRN50K_ml_prediction_model\models\trained\baseline_xgboost_20260216_162414_20260216_162504_results.yaml'
]

# Output file for comparison
output_file = Path(r'C:\Users\ghald\Documents\FRN50K_ml_prediction_model\models\trained\all_models_comparison.yaml')

def decode_scalar(x):
    """Convert numpy scalars to float for YAML compatibility."""
    if isinstance(x, np.generic):
        return float(x)
    return x

def main():
    comparison = []
    for file in yaml_files:
        with open(file, "rb") as f:
            data = yaml.load(f, Loader=yaml.UnsafeLoader)
        
        # Decode metrics
        train_metrics = {k: decode_scalar(v) for k, v in data['train_metrics'].items()}
        eval_metrics = {k: decode_scalar(v) for k, v in data['eval_metrics'].items()}

        # Feature importance
        feature_importance = {}
        if data.get('feature_importance'):
            feature_importance = {k: decode_scalar(v) for k, v in data['feature_importance'].items()}
        
        # Append to comparison list
        comparison.append({
            'model': data['model_name'],
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'top_features': feature_importance
        })

    # Save combined YAML
    with open(output_file, 'w') as f:
        yaml.dump(comparison, f, default_flow_style=False)
    
    # Also create a summary DataFrame for evaluation metrics
    df_eval = pd.DataFrame([
        {
            'model': item['model'],
            **item['eval_metrics']
        } for item in comparison
    ])
    
    print("Model Evaluation Comparison")
    print(df_eval.sort_values('MAE'))


if __name__ == "__main__":
    main()
