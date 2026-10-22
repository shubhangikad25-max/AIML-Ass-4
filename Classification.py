
# Practical: Comparative Analysis of Classification Algorithms
# Algorithms: SVM, Decision Tree, KNN, and Naive Bayes

# Import pandas for storing and displaying results in a table
import pandas as pd

# Import the Breast Cancer dataset
from sklearn.datasets import load_breast_cancer

# Import the function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Import StandardScaler to normalize feature values
from sklearn.preprocessing import StandardScaler

# Import make_pipeline to combine scaling and classification
from sklearn.pipeline import make_pipeline

# Import the four classification algorithms
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Import performance evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# STEP 1: Load the dataset
# --------------------------------------------------

# Load the built-in Breast Cancer Wisconsin dataset
data = load_breast_cancer()

# X contains input features and y contains target labels
X = data.data
y = data.target

# Display basic information about the dataset
print("Dataset loaded successfully.")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", data.target_names)

# --------------------------------------------------
# STEP 2: Split the dataset
# --------------------------------------------------

# Use 80% of data for training and 20% for testing
# random_state ensures reproducible results
# stratify preserves class proportions in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display the number of training and testing samples
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# STEP 3: Initialize classification models
# --------------------------------------------------

# StandardScaler is used for SVM and KNN because
# these algorithms are sensitive to feature scales
models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    ),

    # Limit tree depth to help reduce overfitting
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    # Use 5 nearest neighbors for classification
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),

    # Gaussian Naive Bayes for continuous features
    "Naive Bayes": GaussianNB()
}

# Create a list to store model evaluation results
results = []

# --------------------------------------------------
# STEP 4: Train and evaluate each model
# --------------------------------------------------

# Iterate through all four classification algorithms
for name, model in models.items():

    # Train the model using training data
    model.fit(X_train, y_train)

    # Predict class labels for the test data
    predictions = model.predict(X_test)

    # Predict class probabilities for calculating log loss
    probabilities = model.predict_proba(X_test)

    # Calculate the proportion of correct predictions
    accuracy = accuracy_score(y_test, predictions)

    # Calculate weighted precision across both classes
    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Calculate weighted recall across both classes
    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Calculate weighted F1-score
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Calculate log loss; lower values are better
    loss = log_loss(y_test, probabilities)

    # Store all metrics for the current model
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Log Loss": loss
    })

    # Display the model name
    print("\n" + "=" * 45)
    print("Algorithm:", name)
    print("=" * 45)

    # Display precision, recall, F1-score, and other
    # class-wise evaluation details
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=data.target_names,
            zero_division=0
        )
    )

    # Display the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# STEP 5: Compare the classification models
# --------------------------------------------------

# Convert the results list into a Pandas DataFrame
results_df = pd.DataFrame(results)

# Display the comparison table rounded to four decimals
print("\nCOMPARATIVE PERFORMANCE ANALYSIS")
print(results_df.round(4).to_string(index=False))

# --------------------------------------------------
# STEP 6: Find the best model for each metric
# --------------------------------------------------

print("\nBEST MODEL FOR EACH METRIC")

# Higher values are better for these metrics
for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]:
    best = results_df.loc[results_df[metric].idxmax()]

    print(
        f"{metric}: {best['Algorithm']} "
        f"({best[metric]:.4f})"
    )

# The lowest log loss indicates the best probability
# predictions according to this metric
best_loss = results_df.loc[
    results_df["Log Loss"].idxmin()
]

print(
    f"Lowest Log Loss: {best_loss['Algorithm']} "
    f"({best_loss['Log Loss']:.4f})"
)

# --------------------------------------------------
# STEP 7: Save the results
# --------------------------------------------------

# Export the comparative table to a CSV file
results_df.to_csv(
    "comparison_results.csv",
    index=False
)

print("\nResults saved to comparison_results.csv")
print("Classification analysis completed successfully.")
