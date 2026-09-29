import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Manual Train-Test Split Function
# ---------------------------------------------------------
# This function divides the dataset into training and testing
# sets without using sklearn's train_test_split function.
# Stratification is used to maintain the class distribution.
def manual_train_test_split(X, y, test_size=0.3, random_state=42, stratify=None):
    rng = np.random.RandomState(random_state)

    if stratify is not None:
        train_idx, test_idx = [], []
        classes = np.unique(stratify)

        # Split each class separately to preserve class proportions
        for cls in classes:
            cls_idx = np.where(np.array(stratify) == cls)[0]
            rng.shuffle(cls_idx)

            n_test = max(1, int(len(cls_idx) * test_size))

            test_idx.extend(cls_idx[:n_test])
            train_idx.extend(cls_idx[n_test:])

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    else:
        # Randomly split the complete dataset when stratification is not used
        n = len(y)
        indices = np.arange(n)
        rng.shuffle(indices)

        n_test = int(n * test_size)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    # Select the corresponding training and testing samples
    X_train = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
    X_test = X[test_idx] if isinstance(X, np.ndarray) else X.iloc[test_idx]

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Manual Standardization
# ---------------------------------------------------------
# KNN is distance-based, so features should be on a similar
# scale. Standardization converts the features using:
#
#       X_scaled = (X - mean) / standard deviation
def manual_standard_scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Prevent division by zero for constant features
    std[std == 0] = 1

    return (X - mean) / std


# ---------------------------------------------------------
# Manual Evaluation Metrics
# ---------------------------------------------------------
# Accuracy measures the overall percentage of correctly
# classified samples.
def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


# Precision measures how many predicted positive samples
# are actually positive.
def precision_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


# Recall measures how many actual positive samples
# were correctly identified by the model.
def recall_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


# F1-score is the harmonic mean of precision and recall.
def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------
# Manual ROC Curve
# ---------------------------------------------------------
# This function calculates False Positive Rate (FPR) and
# True Positive Rate (TPR) for different prediction thresholds.
def roc_curve(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Sort predictions from highest probability to lowest
    desc_indices = np.argsort(y_scores)[::-1]

    y_scores = y_scores[desc_indices]
    y_true = y_true[desc_indices]

    # Find positions where the prediction score changes
    distinct_indices = np.where(np.diff(y_scores))[0]

    threshold_indices = np.concatenate(
        [distinct_indices, [len(y_true) - 1]]
    )

    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    # Calculate cumulative true positives and false positives
    tps = np.cumsum(y_true)[threshold_indices]
    fps = (threshold_indices + 1) - tps

    # Calculate TPR and FPR
    tpr = (
        np.concatenate([[0], tps / total_pos])
        if total_pos > 0
        else np.concatenate([[0], np.zeros_like(tps)])
    )

    fpr = (
        np.concatenate([[0], fps / total_neg])
        if total_neg > 0
        else np.concatenate([[0], np.zeros_like(fps)])
    )

    thresholds = y_scores[threshold_indices]

    return fpr, tpr, thresholds


# Calculate Area Under the ROC Curve using the trapezoidal rule
def roc_auc_score(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return np.trapezoid(tpr, fpr)


# ---------------------------------------------------------
# Load Marketing Campaign Dataset
# ---------------------------------------------------------
# The dataset is loaded from the Excel file and the required
# worksheet is selected.
df = pd.read_excel(
    "Lab Session Data (1).xlsx",
    sheet_name="marketing_campaign"
)


# ---------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------

# Remove duplicate records to avoid repeated observations
df = df.drop_duplicates()

# Convert Response into numeric format
# Invalid values are converted into missing values
df["Response"] = pd.to_numeric(
    df["Response"],
    errors="coerce"
)

# Remove rows where the target variable is missing
df = df.dropna(subset=["Response"])


# ---------------------------------------------------------
# Date Feature Extraction
# ---------------------------------------------------------
# Convert customer registration date into datetime format.
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    errors="coerce",
    dayfirst=True
)

# Extract useful numerical information from the date
df["Customer_Year"] = df["Dt_Customer"].dt.year
df["Customer_Month"] = df["Dt_Customer"].dt.month
df["Customer_Day"] = df["Dt_Customer"].dt.day

# The original date column is no longer required
df = df.drop(columns=["Dt_Customer"])


# ID is only an identifier and does not contribute useful
# predictive information for the KNN model.
df = df.drop(columns=["ID"])


# ---------------------------------------------------------
# Separate Features and Target
# ---------------------------------------------------------
# Response is the target variable.
# All remaining columns are used as input features.
X = df.drop(columns=["Response"])
y = df["Response"].astype(int)


# ---------------------------------------------------------
# Convert Categorical Features
# ---------------------------------------------------------
# One-hot encoding converts categorical variables into
# numerical columns that can be processed by KNN.
X = pd.get_dummies(X, drop_first=True)

# Convert all feature values to numeric form
X = X.apply(pd.to_numeric, errors="coerce")

# Replace missing values using the mean of each column
X = X.fillna(X.mean())

# Replace any remaining missing values with zero
X = X.fillna(0)


# ---------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------
# Standardization is important for KNN because the algorithm
# calculates distances between data points.
X = manual_standard_scale(
    X.values.astype(np.float64)
)


# ---------------------------------------------------------
# Split Dataset
# ---------------------------------------------------------
# 70% of the data is used for training and 30% for testing.
# Stratification maintains the distribution of Response classes.
X_train, X_test, y_train, y_test = manual_train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# Euclidean Distance
# ---------------------------------------------------------
# KNN determines the nearest samples using Euclidean distance.
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# ---------------------------------------------------------
# Bubble Sort
# ---------------------------------------------------------
# Sorts the calculated distances from smallest to largest.
def bubble_sort(distances):
    n = len(distances)

    for i in range(n):
        for j in range(0, n-i-1):

            if distances[j][0] > distances[j+1][0]:
                distances[j], distances[j+1] = (
                    distances[j+1],
                    distances[j]
                )

    return distances


# ---------------------------------------------------------
# Selection Sort
# ---------------------------------------------------------
# Finds the smallest distance and places it in the correct
# position during each iteration.
def selection_sort(distances):
    n = len(distances)

    for i in range(n):
        min_index = i

        for j in range(i+1, n):
            if distances[j][0] < distances[min_index][0]:
                min_index = j

        distances[i], distances[min_index] = (
            distances[min_index],
            distances[i]
        )

    return distances


# ---------------------------------------------------------
# Insertion Sort
# ---------------------------------------------------------
# Builds the sorted distance list one element at a time.
def insertion_sort(distances):
    for i in range(1, len(distances)):

        key = distances[i]
        j = i - 1

        while j >= 0 and distances[j][0] > key[0]:
            distances[j+1] = distances[j]
            j -= 1

        distances[j+1] = key

    return distances


# ---------------------------------------------------------
# Find K Nearest Neighbors
# ---------------------------------------------------------
# Calculates the distance between a test sample and every
# training sample, sorts the distances and returns the k
# closest samples.
def get_neighbors(
    X_train,
    y_train,
    test_point,
    k,
    sorting="insertion"
):
    distances = []

    for i in range(len(X_train)):

        distance = euclidean_distance(
            test_point,
            X_train[i]
        )

        distances.append(
            (distance, y_train.iloc[i])
        )

    # Select the required sorting algorithm
    if sorting == "bubble":
        distances = bubble_sort(distances)

    elif sorting == "selection":
        distances = selection_sort(distances)

    else:
        distances = insertion_sort(distances)

    return distances[:k]


# ---------------------------------------------------------
# Majority Voting
# ---------------------------------------------------------
# In normal KNN, the class with the highest number of votes
# among the k nearest neighbors is selected.
def majority_vote(neighbors):
    votes = {}

    for distance, label in neighbors:
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)


# ---------------------------------------------------------
# Distance Weighted Voting
# ---------------------------------------------------------
# Closer neighbors receive higher weights, allowing nearby
# samples to have more influence on the final prediction.
def weighted_vote(neighbors):
    weights = {}

    for distance, label in neighbors:

        # A very small value avoids division by zero
        weight = 1 / (distance + 1e-10)

        weights[label] = weights.get(label, 0) + weight

    return max(weights, key=weights.get)


# ---------------------------------------------------------
# Custom KNN Classifier
# ---------------------------------------------------------
# This class implements the basic KNN algorithm manually
# instead of using sklearn's KNeighborsClassifier.
class CustomKNN:

    def __init__(
        self,
        k=3,
        sorting="insertion",
        weighted=False
    ):
        self.k = k
        self.sorting = sorting
        self.weighted = weighted

    # Store the training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.reset_index(drop=True)

        return self

    # Predict the class for each test sample
    def predict(self, X):
        predictions = []

        for point in X:

            neighbors = get_neighbors(
                self.X_train,
                self.y_train,
                point,
                self.k,
                self.sorting
            )

            # Choose between normal and weighted voting
            if self.weighted:
                predictions.append(
                    weighted_vote(neighbors)
                )
            else:
                predictions.append(
                    majority_vote(neighbors)
                )

        return np.array(predictions)

    # Estimate the probability of belonging to class 1
    def predict_proba(self, X):
        probabilities = []

        for point in X:

            neighbors = get_neighbors(
                self.X_train,
                self.y_train,
                point,
                self.k,
                self.sorting
            )

            # Probability is based on the proportion of
            # positive-class neighbors.
            probability = sum(
                label == 1
                for distance, label in neighbors
            ) / self.k

            probabilities.append(probability)

        return np.array(probabilities)

    # Calculate classification accuracy
    def score(self, X, y):
        return accuracy_score(
            y,
            self.predict(X)
        )


# ---------------------------------------------------------
# Train the KNN Model
# ---------------------------------------------------------
# A value of k=3 is used for the initial KNN model.
model = CustomKNN(
    k=3,
    sorting="insertion"
)

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# Generate Predictions
# ---------------------------------------------------------
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------
# Calculate the main classification performance metrics.
print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(y_test, y_pred)
)

print(
    "Recall:",
    recall_score(y_test, y_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred)
)


# ---------------------------------------------------------
# AUC and ROC Curve
# ---------------------------------------------------------
# Obtain probability scores for the positive class.
y_prob = model.predict_proba(X_test)

# Calculate the Area Under the ROC Curve.
auc = roc_auc_score(
    y_test,
    y_prob
)

print("AUC:", auc)

# Generate points required for plotting the ROC curve.
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)


# Plot ROC Curve
plt.plot(
    fpr,
    tpr,
    label="AUC = {:.3f}".format(auc)
)

# Diagonal line represents random classification.
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ---------------------------------------------------------
# Effect of Different K Values
# ---------------------------------------------------------
# Test different values of k to understand how the number
# of neighbors affects classification accuracy.
k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:

    model = CustomKNN(k=k)

    model.fit(
        X_train,
        y_train
    )

    accuracies.append(
        model.score(
            X_test,
            y_test
        )
    )


# Plot Accuracy vs K
plt.plot(
    k_values,
    accuracies,
    marker="o"
)

plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.show()


# ---------------------------------------------------------
# Normal KNN vs Weighted KNN
# ---------------------------------------------------------
# Compare standard majority voting with distance-weighted
# voting for different values of k.
normal_accuracy = []
weighted_accuracy = []

for k in k_values:

    # Standard KNN using majority voting
    model1 = CustomKNN(
        k=k,
        weighted=False
    )

    model1.fit(
        X_train,
        y_train
    )

    normal_accuracy.append(
        model1.score(
            X_test,
            y_test
        )
    )

    # Weighted KNN where closer neighbors have more influence
    model2 = CustomKNN(
        k=k,
        weighted=True
    )

    model2.fit(
        X_train,
        y_train
    )

    weighted_accuracy.append(
        model2.score(
            X_test,
            y_test
        )
    )


# ---------------------------------------------------------
# Plot Comparison
# ---------------------------------------------------------
plt.plot(
    k_values,
    normal_accuracy,
    marker="o",
    label="Normal"
)

plt.plot(
    k_values,
    weighted_accuracy,
    marker="o",
    label="Weighted"
)

plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()
plt.show()