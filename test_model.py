import argparse
import logging
import pandas as pd
from joblib import load
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
)
from helper_funcs import *

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(dataset_path, model_path, output_probs_path, mode, random_state=42):
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        logging.info(f"Dataset loaded successfully from {dataset_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {dataset_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

    if mode == "test":
        # Split dataset into training and testing subsets for labeled test data
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
            logging.info("Dataset split into training and testing subsets")
        except Exception as e:
            logging.error(f"Error during train-test split: {e}")
            raise

        # Preprocess the test data
        try:
            X_test, y_test = preprocess_tax_df(test_df)
            logging.info("Test data preprocessing completed")
        except Exception as e:
            logging.error(f"Error during test data preprocessing: {e}")
            raise

    elif mode == "predict":
        # Use the entire dataset for predictions
        try:
            X_test, y_test = preprocess_tax_df(df)  # y_test will be None
            logging.info("Data preprocessing completed for real data")
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise
    else:
        logging.error("Invalid mode selected. Use 'test' or 'predict'.")
        raise ValueError("Invalid mode selected. Use 'test' or 'predict'.")

    # Load the trained model
    try:
        model = load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    # Predict probabilities and labels using the best threshold
    try:
        probabilities = model.predict_proba(X_test)[:, 1]  # Positive class probabilities
        best_threshold = getattr(model, "best_threshold", 0.5)  # Default to 0.5 if not set
        predictions = (probabilities >= best_threshold).astype(int)
        logging.info(f"Predictions made using best threshold: {best_threshold}")
    except AttributeError:
        logging.error("Model does not contain best_threshold attribute. Ensure it's saved correctly during training.")
        raise
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

    # Handle evaluation or prediction output
    if mode == "test":
        # Evaluate the model
        try:
            logging.info("Evaluating model performance")
            print("Classification Report:\n", classification_report(y_test, predictions))
            print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
            print("F1 Score:", f1_score(y_test, predictions))
            print("Average Precision Score:", average_precision_score(y_test, probabilities))
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise
    else:  # mode == "predict"
        logging.info("Real data prediction mode. No evaluation metrics generated.")

    # Save probabilities and predictions to a CSV file
    try:
        output_data = {"predicted_probabilities": probabilities, "predictions": predictions}
        if mode == "test":
            output_data["actual"] = y_test  # Add actual labels if available

        probabilities_df = pd.DataFrame(output_data)
        probabilities_df.to_csv(output_probs_path, index=False)
        logging.info(f"Predicted probabilities saved to {output_probs_path}")
    except Exception as e:
        logging.error(f"Error saving probabilities to CSV: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or predict using a trained machine learning model.")
    parser.add_argument("--dataset", required=True, help="Path to the input dataset (CSV format).")
    parser.add_argument("--model", required=True, help="Path to the trained model file.")
    parser.add_argument("--output_probs", default="predicted_probabilities.csv", help="Path to save predicted probabilities.")
    parser.add_argument("--mode", choices=["test", "predict"], required=True, help="Mode: 'test' for evaluation or 'predict' for real data.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for train-test split (used in 'test' mode).")

    args = parser.parse_args()
    main(args.dataset, args.model, args.output_probs, args.mode, args.random_state)
