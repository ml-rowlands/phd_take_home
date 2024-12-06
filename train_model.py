import argparse
import logging
from joblib import dump
from helper_funcs import *
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(dataset_path, output_path):
    # Load dataset using Polars
    try:
        df = pl.read_csv(dataset_path)
        logging.info(f"Dataset loaded successfully from {dataset_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {dataset_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        raise

    # Preprocess dataset using preprocess_tax_df
    X, y = preprocess_tax_df(df)
    logging.info("Dataset preprocessing completed")

    # Define preprocessing pipeline for numerical and categorical features
    categorical_features = [col for col in X.columns if X[col].dtype == pl.String]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        'imputer', SimpleImputer(strategy='constant', fill_value='missing'),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features)
        ]
    )

    # Define models and hyperparameters
    models = [
        ("RandomForest", RandomForestClassifier(), {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        })
        # Add more models if needed
    ]

    # Evaluate models using evaluate_models_with_thresholds
    results = evaluate_models_with_thresholds(
        models=models,
        X=X.to_pandas(),  # Convert Polars DataFrame to Pandas for scikit-learn
        y=y,
        preprocessor=preprocessor,
        sampling_strategies=[None]
    )

    # Find the best model based on profit
    best_model_name = max(results, key=lambda k: results[k]["total_profit"])
    best_model = results[best_model_name]["model"]

    # Save the best model
    dump(best_model, output_path)
    logging.info(f"Best model '{best_model_name}' saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize a machine learning model.")
    parser.add_argument("--dataset", required=True, help="Path to the input dataset (CSV format).")
    parser.add_argument("--output", default="best_model_pipeline.joblib", help="Path to save the trained model.")

    args = parser.parse_args()
    main(args.dataset, args.output)
