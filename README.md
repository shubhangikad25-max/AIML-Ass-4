# Comparative Analysis of Classification Algorithms

## 📌 Project Overview

This project compares four supervised machine learning classification algorithms using the Iris Flower Dataset. The objective is to train the models, evaluate their performance using different metrics, and identify the strengths and limitations of each algorithm.

The algorithms implemented are:
- Support Vector Machine (SVM)
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Gaussian Naïve Bayes

## 🎯 Objectives

- Implement four classification algorithms using Python.
- Train and test each model on the same dataset.
- Evaluate performance using standard classification metrics.
- Compare the results to understand the effectiveness of different algorithms.
- Save the model comparison results in a CSV file.

## 📊 Dataset Description

The project uses the Iris Flower Dataset available through Scikit-learn.

- **Total samples:** 150
- **Number of features:** 4
- **Number of classes:** 3
- **Training data:** 80% (120 samples)
- **Testing data:** 20% (30 samples)

### Input Features

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

### Target Classes

- Iris Setosa
- Iris Versicolor
- Iris Virginica

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Machine Learning
- Classification Algorithms

## 🤖 Algorithms Implemented

### 1. Support Vector Machine (SVM)
Classifies samples by finding an effective decision boundary. The RBF kernel is used to handle nonlinear relationships.

### 2. Decision Tree
Classifies samples using feature-based decision rules. The maximum tree depth is limited to help reduce overfitting.

### 3. K-Nearest Neighbors (KNN)
Predicts the class of a sample using the majority class among its five nearest neighbors.

### 4. Gaussian Naïve Bayes
Uses Bayes' theorem and Gaussian probability distributions to predict the flower species.

## 📈 Evaluation Metrics

The algorithms are evaluated using:

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures the correctness of positive class predictions.
- **Recall:** Measures how many actual class samples are correctly identified.
- **F1-Score:** Combines precision and recall.
- **Log Loss:** Evaluates the quality of predicted class probabilities.
- **Confusion Matrix:** Displays correct and incorrect predictions for each class.

## ⚙️ Project Workflow

1. Load the Iris dataset.
2. Separate features and target labels.
3. Split the dataset into training and testing sets.
4. Apply feature scaling for SVM and KNN.
5. Train all four classification models.
6. Generate predictions on test data.
7. Calculate evaluation metrics.
8. Compare model performance.
9. Export results to a CSV file.

## 📂 Project Structure

```text
classification-algorithms/
│
├── classification.py
├── comparison_results.csv
└── README.md
```

`comparison_results.csv` is generated when the Python program runs successfully.

## 🚀 Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd classification-algorithms
```

### Step 3: Install Dependencies

```bash
pip install pandas scikit-learn
```

### Step 4: Run the Program

```bash
python classification.py
```

## 📋 Output

The program displays:

- Dataset information
- Training and testing sample counts
- Classification reports
- Confusion matrices
- Comparative performance table
- Best-performing model for each evaluation metric

The final comparison is saved in `comparison_results.csv`.

## 🧪 Experimental Setup

- Train-test split: 80:20
- Random state: 42
- SVM kernel: RBF
- Decision Tree maximum depth: 5
- KNN neighbors: 5
- Naïve Bayes variant: Gaussian Naïve Bayes
- Precision, recall, and F1-score averaging: Weighted

## 🏁 Conclusion

This project demonstrates the implementation and comparison of four supervised machine learning classification algorithms on the Iris Flower Dataset. The performance metrics help evaluate the models from different perspectives and understand their relative strengths and limitations.

The best-performing algorithm is determined from the actual experimental results.

## 👩‍💻 Author

**Shubhangi Kad**

## 📄 License

This project is intended for educational and academic purposes.
