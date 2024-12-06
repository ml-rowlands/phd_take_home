import polars as pl
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, roc_auc_score, classification_report, confusion_matrix, make_scorer, f1_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import numpy as np 
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline


##################################################################Preprocessing Functions##################################################################

def preprocess_tax_df(df):
    """Preprocesses the tax dataset and returns the features and target variable. Drops column c10 which is a recoding of the target variable. 
    Recodes column n4 to be catergorical with 999 (missing) being 0 and 1 otherwise. Bins the 'i4' column into 4 bins. Drops columns i1, i2, i5, due to high correlation with i4.

    Parameters:
       - df (polars): Dataset to preprocess.
        
    Returns:
       - X (polars): Preprocessed features.
       - y (polars): Target variable.
    """
    
    #Drop c10 as it is a recoding of the target variable
    df = df.drop('c10')
    
    # Subset to categorical cols
    cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
    
    # Fill categorical cols missing vals with "missing"
    df = df.with_columns(
        [pl.col(col).fill_null('missing').alias(col) for col in cat_cols]
    )
    
    # Replace specific missing value indicators with 'missing'
    df = df.with_columns([
        pl.when(pl.col('b1') == '-1').then(pl.lit('missing')).otherwise(pl.col('b1')).alias('b1'),
        pl.when(pl.col('c3') == 'unknown').then(pl.lit('missing')).otherwise(pl.col('c3')).alias('c3'),
        pl.when(pl.col('employment') == 'unknown').then(pl.lit('missing')).otherwise(pl.col('employment')).alias('employment')])
   
    
    #Recode column n4 to be catergorical with 999 (missing) being 0 and 1 otherwise
    df = df.with_columns(pl.when(
    pl.col('n4') == 999).then(0).otherwise(1).alias('n4_recoded'))
    
    
    # Bin the 'i4' column into 4 bins
    df = df.with_columns(
        pl.when(pl.col("i4") <= 1.5)
        .then(pl.lit("low"))
        .when(pl.col("i4").is_between(1.5, 3.9))  
        .then(pl.lit("medium"))
        .when(pl.col("i4").is_between(3.9, 4.2))
        .then(pl.lit("high_1"))
        .otherwise(pl.lit("high_2"))
        .alias("i4_binned")
    )
    
    lb = LabelBinarizer()
    y = lb.fit_transform(df['successful_sell'].to_numpy()).ravel() 

    X = df.drop(['i1', 'i2' , 'i4', 'i5', 'n4', 'successful_sell'])
    
    return X, y  

    
    
    
  
############################################################# Model Evaluation Functions ####################################################################  

def custom_cost(y_true, y_pred, tp_cost=1, tn_cost=0, fp_cost=-0.1, fn_cost=-1):
    """
    Calculate the custom cost based on the confusion matrix.

    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.
    - tp_cost, tn_cost, fp_cost, fn_cost: Costs associated with each outcome.

    Returns:
    - Total cost.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate total cost
    total_cost = (tp * tp_cost) + (tn * tn_cost) + (fp * fp_cost) + (fn * fn_cost)
    return total_cost




def evaluate_models_with_thresholds(
    models, X, y, preprocessor, n_splits=5, random_state=42,
    cost_params=None, thresholds=np.linspace(0.1, 0.9, 9), sampling_strategies=None
):
    """
    Evaluate multiple models with hyperparameter tuning using cross-validation,
    optimize classification thresholds for custom cost function, and calculate costs.

    Returns:
    - results: dict, evaluation metrics, costs, optimal thresholds, and predictions for each model
    """
    
    # Set default values of cost parameters and sampling strategies
    if cost_params is None:
        cost_params = {"tp_cost": 100, "tn_cost": 0, "fp_cost": -5, "fn_cost": 0}
        
    if sampling_strategies is None:
        
        sampling_strategies = [
            None,
            RandomOverSampler(random_state=random_state),
            RandomUnderSampler(random_state=random_state)
        ]

    # Initialize cross-validation strategy and results dictionary
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    # Iterate over models and sampling strategies with hyperparameter tuning using GridSearchCV
    for model_name, model, param_grid in models:
        
        for sampler in sampling_strategies:
            sampler_name = sampler.__class__.__name__ if sampler else "NoSampler"
            pipeline = ImbPipeline([
                ("preprocessor", preprocessor),
                ("sampler", sampler if sampler else "passthrough"),
                ("model", model)
            ])

            grid_search = GridSearchCV(
                pipeline,
                param_grid={"model__" + key: value for key, value in param_grid.items()},
                cv=cv,
                scoring='average_precision',  # Use average precision for hyperparameter tuning
                n_jobs=-1,
                return_train_score=True
            )
            
            # Fit the GridSearchCV object
            grid_search.fit(X, y)

            # Pull best model from GridSearchCV using average precision as metric
            best_model = grid_search.best_estimator_

            # Calculate cross-validation score variation for the best model
            best_index = grid_search.best_index_
            cv_scores = grid_search.cv_results_['mean_test_score']
            cv_std = grid_search.cv_results_['std_test_score'][best_index]

            # Get cross-validated probabilities
            probs = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

            # Optimize threshold based on custom cost
            best_threshold = None
            best_cost = float('-inf')
            best_preds = None

            
            for threshold in thresholds:
                
                # If the probability is greater than the threshold, predict as positive
                thresholded_preds = (probs >= threshold).astype(int)
                cost = custom_cost(y, thresholded_preds, **cost_params)
                
                # If threshold increases profit, update best cost, threshold, and predictions
                if cost > best_cost:
                    best_cost = cost
                    best_threshold = threshold
                    best_preds = thresholded_preds

            # Pull feature importance from the best model if available
            feature_importance = None
            if hasattr(best_model.named_steps["model"], "feature_importances_"):
                feature_importance = best_model.named_steps["model"].feature_importances_
            elif hasattr(best_model.named_steps["model"], "coef_"):
                feature_importance = np.abs(best_model.named_steps["model"].coef_).flatten()

            # Confusion matrix
            confusion_mat = confusion_matrix(y, best_preds)

            # F1 score
            f1 = f1_score(y, best_preds)

            # Average precision
            average_precision = average_precision_score(y, probs)

            # Store results
            results[f"{model_name}_{sampler_name}"] = {
                "best_params": grid_search.best_params_,
                "cv_score_variation": cv_std,  # Standard deviation of CV scores
                "best_threshold": best_threshold,
                "total_profit": best_cost,
                "average_profit_per_sale_attempt": best_cost / (confusion_mat[1,1] + confusion_mat[0,1]),
                "feature_importance": feature_importance,
                "model": best_model,
                "confusion_matrix": confusion_mat,
                "F1": f1,
                "average_precision_score": average_precision,
                "probs": probs,  # Store probabilities
                "best_preds": best_preds,  # Store thresholded predictions
            }

    return results





def evaluate_models_across_ratios(models, X, y, preprocessor, ratios, n_splits=5, random_state=42, thresholds=np.linspace(0.1, 0.9, 9), sampling_strategies=None):
    """
    Evaluate models with different cost-to-revenue ratios.

    Parameters:
    - models: list of tuples (model_name, model_instance, param_grid)
    - X: Pandas DataFrame containing the X training data
    - y: Series containing the target variable
    - preprocessor: ColumnTransformer for preprocessing features
    - ratios: list of tuples (tp_cost, fp_cost, fn_cost), representing different cost-to-revenue ratios
    - n_splits: int, number of folds for cross-validation (default: 5)
    - random_state: int, random state for reproducibility (default: 42)
    - thresholds: array-like, thresholds to evaluate (default: np.linspace(0.1, 0.9, 9))
    - sampling_strategies: list of samplers to evaluate (default: None)

    Returns:
    - results: dict, performance metrics for each model across ratios
    """
    results_by_ratio = {}

    for tp_cost, fp_cost, fn_cost in ratios:
        cost_params = {"tp_cost": tp_cost, "tn_cost": 0, "fp_cost": fp_cost, "fn_cost": fn_cost}
        print(f"Evaluating models for cost parameters: TP={tp_cost}, FP={fp_cost}, FN={fn_cost}")

        results = evaluate_models_with_thresholds(
            models=models,
            X=X,
            y=y,
            preprocessor=preprocessor,
            n_splits=n_splits,
            random_state=random_state,
            cost_params=cost_params,
            thresholds=thresholds,
            sampling_strategies=sampling_strategies
        )

        results_by_ratio[(tp_cost, fp_cost, fn_cost)] = results

    return results_by_ratio





def model_diagnostics(results):
    """Print model diagnostics using results dictionary.

    Args:
        results (dict): Dictionary of model results.
    """
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["total_profit"], reverse=True)
    
    for model_name, metrics in sorted_results:
        
        print(f"Model: {model_name}")
        print(f"Best Parameters: {metrics['best_params']}")
        print(f"Total Profit: {metrics['total_profit']:.2f}")
        print(f"Average Profit per Sale Attempt: {metrics['average_profit_per_sale_attempt']:.2f}")
        print(f"Best Threshold: {metrics['best_threshold']:.2f}")
        print(f"Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"F1 Score: {metrics['F1']:.2f}")
        print(f"Average Precision Score: {metrics['average_precision_score']:.2f}")
        print(f"CV Average Precision Variation (std): {metrics['cv_score_variation']:.3f}")
        print("-" * 50)






def get_transformed_feature_names(preprocessor, feature_names, numeric_features, categorical_features):
    """
    Retrieve transformed feature names after preprocessing.

    Parameters:
    - preprocessor: The ColumnTransformer used for preprocessing.
    - feature_names: List of original feature names (before preprocessing).
    - numeric_features: List of numeric feature names.
    - categorical_features: List of categorical feature names.

    Returns:
    - List of transformed feature names.
    """
    transformed_names = []
    
    # Extract transformers
    transformers = preprocessor.named_transformers_
    
    # Add numeric features (unchanged after scaling)
    if "num" in transformers:
        transformed_names.extend(numeric_features)
    
    # Add expanded categorical features (after OneHotEncoding)
    if "cat" in transformers:
        encoder = transformers["cat"]
        if hasattr(encoder, "get_feature_names_out"):
            cat_names = encoder.get_feature_names_out(categorical_features)
            transformed_names.extend(cat_names)
        else:
            transformed_names.extend(categorical_features)
    
    return transformed_names





def plot_feature_importances(results, feature_names):
    """
    Plot feature importances for all models in the results with feature names, spacing, and ordering.

    Parameters:
    - results: dict, results containing feature importances for each model
    - feature_names: list, names of the features
    """
    # Iterate over models in results
    for model_name, metrics in results.items():
        if "feature_importance" in metrics and metrics["feature_importance"] is not None:
            importances = metrics["feature_importance"]
            
            # Normalize importances to sum to 1 for comparison
            normalized_importances = importances / np.sum(importances)
            
            # Sort features by importance
            sorted_indices = np.argsort(normalized_importances)[::-1]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]
            sorted_importances = normalized_importances[sorted_indices]
            
            # Create bar plot
            plt.figure(figsize=(12, max(6, len(sorted_feature_names) * 0.25)))  # Adjust height based on number of features
            plt.barh(sorted_feature_names, sorted_importances, color="skyblue")
            plt.title(f"Feature Importances for {model_name}")
            plt.xlabel("Normalized Importance")
            plt.ylabel("Features")
            plt.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()





def get_top_features(results, model_name, feature_names, top_n=10):
    """
    Extract the top N most important features from a model's feature importance.

    Parameters:
    - results: dict, results dictionary containing model metrics and feature importances
    - model_name: str, the name of the model (e.g., "Random Forest")
    - feature_names: list, original feature names
    - top_n: int, number of top features to select (default: 10)

    Returns:
    - top_features: list, names of the top N most important features
    """
    # Get the feature importances for the specified model
    feature_importances = results[model_name]["feature_importance"]

    if feature_importances is None:
        raise ValueError(f"Feature importance not available for model: {model_name}")

    # Create a DataFrame to sort features by importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    })

    # Sort by importance in descending order and pick the top N features
    top_features = (
        importance_df.sort_values(by="Importance", ascending=False)
        .head(top_n)["Feature"]
        .tolist()
    )

    return top_features





def visualize_prediction_confidence(results, y):
    """
    Visualize prediction confidence (probabilities) for correct and incorrect predictions.

    Parameters:
    - results: dict, results from evaluate_models_with_thresholds.
    - y: Series or array of true labels.
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, metrics in results.items():
        probs = metrics["probs"]
        preds = metrics["best_preds"]
        best_threshold = metrics["best_threshold"]

        correct_indices = (preds == y)
        incorrect_indices = ~correct_indices

        plt.hist(probs[correct_indices], bins=20, alpha=0.5, label=f"{model_name} Correct", density=True)
        plt.hist(probs[incorrect_indices], bins=20, alpha=0.5, label=f"{model_name} Incorrect", density=True)
        
        ax = plt.gca()
        ax.axvline(best_threshold, color="red", linestyle="--", label="Best Threshold")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Prediction Confidence for Correct and Incorrect Predictions")
    plt.legend()
    plt.show()
    
    
def visualize_error_overlap_by_class(misclassified_df, models_to_compare, true_labels, target_class):
    """
    Visualize the overlap of misclassified instances for a specific true label.

    Args:
        misclassified_df (pd.DataFrame): DataFrame containing misclassified instances for models.
        models_to_compare (list): List of model names to compare.
        true_labels (np.ndarray or pd.Series): True labels for the target variable.
        target_class (int): The class label to filter on (e.g., 0 or 1).

    Returns:
        None
    """
    # Ensure true_labels is a pandas Series
    if not isinstance(true_labels, pd.Series):
        true_labels = pd.Series(true_labels, name="True_Label")

    # Filter by the target class
    filtered_indices = true_labels[true_labels == target_class].index
    filtered_misclassified = misclassified_df.loc[filtered_indices]

    # Prepare sets for Venn diagram
    sets = [
        set(filtered_misclassified[filtered_misclassified[model] == 1].index)
        for model in models_to_compare
    ]

    # Plot Venn diagram
    plt.figure(figsize=(8, 8))
    venn = venn3(
        subsets=sets,
        set_labels=models_to_compare
    )
    plt.title(f"Overlap of Misclassified Instances for True Class {target_class}")
    plt.show()



def error_analysis(results, X, y):
    """
    Perform error analysis across models.

    Parameters:
    - results: dict, results from evaluate_models_with_thresholds.
    - X: Pandas DataFrame or NumPy array of input features.
    - y: Pandas Series, array, or list of true labels.

    Returns:
    - misclassified_df: DataFrame showing misclassified instances for each model.
    """
    misclassified = {}
    
    # Ensure y is a pandas Series for consistency
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="True_Label")

    # Iterate through models to collect misclassified data
    for model_name, metrics in results.items():
        best_preds = metrics["best_preds"]  # Use stored hard predictions
        misclassified[model_name] = (best_preds != y).astype(int)  # 1 if misclassified, 0 otherwise

    # Convert misclassified dictionary to DataFrame
    misclassified_df = pd.DataFrame(misclassified)

    # Add true labels
    misclassified_df["True_Label"] = y.values

    return misclassified_df



def visualize_error_overlap(misclassified_df, models):
    """
    Visualize overlap of misclassified instances across models using a Venn diagram.

    Parameters:
    - misclassified_df: DataFrame, output of error_analysis.
    - models: list of model names to include in the Venn diagram.
    """
    if len(models) != 3:
        print("Venn diagram supports exactly 3 models. Adjust model selection.")
        return

    sets = [
        set(misclassified_df.index[misclassified_df[models[0]] == 1]),
        set(misclassified_df.index[misclassified_df[models[1]] == 1]),
        set(misclassified_df.index[misclassified_df[models[2]] == 1]),
    ]

    venn = venn3(sets, set_labels=models)
    plt.title("Overlap of Misclassified Instances")
    plt.show()