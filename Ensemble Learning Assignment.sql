# Ensemble Learning Assignment 

## Question 1 - Can we use Bagging for regression problems?

Answer:
Yes, Bagging can be used for regression problems. In regression tasks, Bagging combines predictions from multiple regression models by averaging their outputs. This helps reduce variance and improves model stability and accuracy. A common example is the Bagging Regressor.

---

## Question 2 - What is the difference between multiple model training and single model training?

Answer:
Single model training uses only one machine learning algorithm to make predictions, while multiple model training combines predictions from several models. Ensemble methods use multiple models to improve performance, accuracy, and robustness.

---

## Question 3 - Explain the concept of feature randomness in Random Forest.

Answer:
Feature randomness in Random Forest means that at each split in a decision tree, only a random subset of features is considered. This increases diversity among trees and reduces overfitting.

---

## Question 4 - What is OOB (Out-of-Bag) Score?

Answer:
OOB Score is a validation technique used in Bagging and Random Forest. Some training samples are not selected during bootstrap sampling. These unused samples are called Out-of-Bag samples and are used to evaluate model performance without needing a separate validation dataset.

---

## Question 5 - How can you measure the importance of features in a Random Forest model?

Answer:
Feature importance in Random Forest can be measured using the decrease in impurity or Gini importance. Features contributing more to reducing impurity receive higher importance scores.

---

## Question 6 - Explain the working principle of a Bagging Classifier.

Answer:
A Bagging Classifier creates multiple bootstrap samples from the training dataset and trains separate classifiers on each sample. Final predictions are made using majority voting among all classifiers.

---

## Question 7 - How do you evaluate a Bagging Classifier’s performance?

Answer:
A Bagging Classifier can be evaluated using metrics such as accuracy, precision, recall, F1-score, ROC-AUC score, and confusion matrix.

---

## Question 8 - How does a Bagging Regressor work?

Answer:
A Bagging Regressor trains multiple regression models on different bootstrap samples of the dataset. The final prediction is obtained by averaging the predictions from all models.

---

## Question 9 - What is the main advantage of ensemble techniques?

Answer:
The main advantage of ensemble techniques is improved prediction accuracy and robustness by combining multiple models.

---

## Question 10 - What is the main challenge of ensemble methods?

Answer:
The main challenge of ensemble methods is increased computational complexity and reduced interpretability.

---

## Question 11 - Explain the key idea behind ensemble techniques.

Answer:
The key idea behind ensemble techniques is combining multiple weak or simple models to create a stronger and more accurate predictive model.

---

## Question 12 - What is a Random Forest Classifier?

Answer:
A Random Forest Classifier is an ensemble learning algorithm that combines multiple decision trees trained on random subsets of data and features to perform classification tasks.

---

## Question 13 - What are the main types of ensemble techniques?

Answer:
The main types of ensemble techniques are:

1. Bagging
2. Boosting
3. Stacking
4. Voting

---

## Question 14 - What is ensemble learning in machine learning?

Answer:
Ensemble learning is a machine learning technique where multiple models are combined to improve overall performance and accuracy.

---

## Question 15 - When should we avoid using ensemble methods?

Answer:
Ensemble methods should be avoided when computational resources are limited, interpretability is important, or the dataset is very small.

---

## Question 16 - How does Bagging help in reducing overfitting?

Answer:
Bagging reduces overfitting by training multiple models on different random subsets of data and averaging their predictions, which lowers variance.

---

## Question 17 - Why is Random Forest better than a single Decision Tree?

Answer:
Random Forest is better because it reduces overfitting, improves accuracy, and provides more stable predictions by combining multiple decision trees.

---

## Question 18 - What is the role of bootstrap sampling in Bagging?

Answer:
Bootstrap sampling creates multiple random subsets of the original dataset with replacement. These subsets are used to train different models in Bagging.

---

## Question 19 - What are some real-world applications of ensemble techniques?

Answer:
Real-world applications include fraud detection, medical diagnosis, recommendation systems, stock market prediction, spam detection, and customer churn prediction.

---

## Question 20 - What is the difference between Bagging and Boosting?

Answer:
Bagging trains models independently and combines predictions to reduce variance, while Boosting trains models sequentially where each model corrects errors made by previous models.

---

# Practical Questions

## Question 21 - Train a Bagging Classifier using Decision Trees on a sample dataset and print model accuracy.

```python
# Question 21 - Train a Bagging Classifier using Decision Trees on a sample dataset and print model accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base model
base_model = DecisionTreeClassifier()

# Bagging classifier
bagging_model = BaggingClassifier(
    estimator=base_model,
    n_estimators=50,
    random_state=42
)

# Train model
bagging_model.fit(X_train, y_train)

# Predictions
predictions = bagging_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
```

---

## Question 22 - Train a Bagging Regressor using Decision Trees and evaluate using Mean Squared Error (MSE).

```python
# Question 22 - Train a Bagging Regressor using Decision Trees and evaluate using Mean Squared Error (MSE)

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base model
base_regressor = DecisionTreeRegressor()

# Bagging regressor
bagging_regressor = BaggingRegressor(
    estimator=base_regressor,
    n_estimators=50,
    random_state=42
)

# Train model
bagging_regressor.fit(X_train, y_train)

# Predictions
predictions = bagging_regressor.predict(X_test)

# MSE
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
```

---

## Question 23 - Train a Random Forest Classifier on the Breast Cancer dataset and print feature importance scores.

```python
# Question 23 - Train a Random Forest Classifier on the Breast Cancer dataset and print feature importance scores

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
rf_model.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': rf_model.feature_importances_
})

print(importance.sort_values(by='Importance', ascending=False))
```

---

## Question 24 - Train a Random Forest Regressor and compare its performance with a single Decision Tree.

```python
# Question 24 - Train a Random Forest Regressor and compare its performance with a single Decision Tree

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
single_tree = DecisionTreeRegressor(random_state=42)
single_tree.fit(X_train, y_train)
tree_predictions = single_tree.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_predictions)

# Random Forest
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)

print("Decision Tree MSE:", tree_mse)
print("Random Forest MSE:", rf_mse)
```

---

## Question 25 - Compute the Out-of-Bag (OOB) Score for a Random Forest Classifier.

```python
# Question 25 - Compute the Out-of-Bag (OOB) Score for a Random Forest Classifier

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Random Forest with OOB score
rf_model = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)

# Train model
rf_model.fit(X, y)

# OOB Score
print("OOB Score:", rf_model.oob_score_)
```

---

## Question 26 - Train a Bagging Classifier using SVM as a base estimator and print accuracy.

```python
# Question 26 - Train a Bagging Classifier using SVM as a base estimator and print accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base estimator
svm_model = SVC(probability=True)

# Bagging classifier
bagging_model = BaggingClassifier(
    estimator=svm_model,
    n_estimators=10,
    random_state=42
)

# Train model
bagging_model.fit(X_train, y_train)

# Predictions
predictions = bagging_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
```

---

## Question 27 - Train a Random Forest Classifier with different numbers of trees and compare accuracy.

```python
# Question 27 - Train a Random Forest Classifier with different numbers of trees and compare accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different tree counts
tree_counts = [10, 50, 100, 200]

for trees in tree_counts:
    model = RandomForestClassifier(n_estimators=trees, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Trees: {trees}, Accuracy: {accuracy}")
```

---

## Question 28 - Train a Bagging Classifier using Logistic Regression as a base estimator and print AUC score.

```python
# Question 28 - Train a Bagging Classifier using Logistic Regression as a base estimator and print AUC score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base estimator
log_model = LogisticRegression(max_iter=10000)

# Bagging classifier
bagging_model = BaggingClassifier(
    estimator=log_model,
    n_estimators=20,
    random_state=42
)

# Train model
bagging_model.fit(X_train, y_train)

# Prediction probabilities
probabilities = bagging_model.predict_proba(X_test)[:, 1]

# AUC Score
auc = roc_auc_score(y_test, probabilities)
print("AUC Score:", auc)
```

---

## Question 29 - Train a Random Forest Regressor and analyze feature importance scores.

```python
# Question 29 - Train a Random Forest Regressor and analyze feature importance scores

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
rf_model.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    'Feature': housing.feature_names,
    'Importance': rf_model.feature_importances_
})

print(importance.sort_values(by='Importance', ascending=False))
```

---

## Question 30 - Train an ensemble model using both Bagging and Random Forest and compare accuracy.

```python
# Question 30 - Train an ensemble model using both Bagging and Random Forest and compare accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Bagging model
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Bagging Accuracy:", bagging_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
```

---

# Additional Practical Questions

## Question 31 - Train a Random Forest Classifier and tune hyperparameters using GridSearchCV.

```python
# Question 31 - Train a Random Forest Classifier and tune hyperparameters using GridSearchCV

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf_model = RandomForestClassifier(random_state=42)

# Parameters
parameters = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10]
}

# GridSearchCV
grid_search = GridSearchCV(rf_model, parameters, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

---

## Question 32 - Train a Bagging Regressor with different numbers of base estimators and compare performance.

```python
# Question 32 - Train a Bagging Regressor with different numbers of base estimators and compare performance

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

estimators = [10, 50, 100]

for n in estimators:
    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=n,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f"Estimators: {n}, MSE: {mse}")
```

---

## Question 33 - Train a Random Forest Classifier and analyze misclassified samples.

```python
# Question 33 - Train a Random Forest Classifier and analyze misclassified samples

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
predictions = rf_model.predict(X_test)

# Misclassified samples
misclassified = pd.DataFrame(X_test)
misclassified['Actual'] = y_test
misclassified['Predicted'] = predictions

misclassified = misclassified[misclassified['Actual'] != misclassified['Predicted']]

print(misclassified)
```

---

## Question 34 - Train a Bagging Classifier and compare its performance with a single Decision Tree Classifier.

```python
# Question 34 - Train a Bagging Classifier and compare its performance with a single Decision Tree Classifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
tree_predictions = single_tree.predict(X_test)

# Bagging
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_predictions))
print("Bagging Accuracy:", accuracy_score(y_test, bagging_predictions))
```

---

## Question 35 - Train a Random Forest Classifier and visualize the confusion matrix.

```python
# Question 35 - Train a Random Forest Classifier and visualize the confusion matrix

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
predictions = rf_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

# Visualization
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()
```

---

## Question 36 - Train a Stacking Classifier using Decision Trees, SVM, and Logistic Regression, and compare accuracy.

```python
# Question 36 - Train a Stacking Classifier using Decision Trees, SVM, and Logistic Regression, and compare accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base models
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True)),
]

# Stacking model
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# Train model
stack_model.fit(X_train, y_train)

# Predictions
predictions = stack_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Stacking Accuracy:", accuracy)
```

---

## Question 37 - Train a Random Forest Classifier and print the top 5 most important features.

```python
# Question 37 - Train a Random Forest Classifier and print the top 5 most important features

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Feature importance
importance = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': rf_model.feature_importances_
})

# Top 5 features
top_features = importance.sort_values(by='Importance', ascending=False).head(5)

print(top_features)
```

---

## Question 38 - Train a Bagging Classifier and evaluate performance using Precision, Recall, and F1-score.

```python
# Question 38 - Train a Bagging Classifier and evaluate performance using Precision, Recall, and F1-score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

# Train model
bagging_model.fit(X_train, y_train)

# Predictions
predictions = bagging_model.predict(X_test)

# Metrics
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
```

---

## Question 39 - Train a Random Forest Classifier and analyze the effect of max_depth on accuracy.

```python
# Question 39 - Train a Random Forest Classifier and analyze the effect of max_depth on accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different max_depth values
max_depth_values = [2, 4, 6, 8, None]

for depth in max_depth_values:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Max Depth: {depth}, Accuracy: {accuracy}")
```

---

## Question 40 - Train a Bagging Regressor using different base estimators (DecisionTree and KNeighbors) and compare performance.

```python
# Question 40 - Train a Bagging Regressor using different base estimators (DecisionTree and KNeighbors) and compare performance

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree base estimator
dt_model = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=50,
    random_state=42
)

dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# KNeighbors base estimator
knn_model = BaggingRegressor(
    estimator=KNeighborsRegressor(),
    n_estimators=50,
    random_state=42
)

knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

# MSE
print("Decision Tree MSE:", mean_squared_error(y_test, dt_predictions))
print("KNeighbors MSE:", mean_squared_error(y_test, knn_predictions))
```

---

## Question 41 - Train a Random Forest Classifier and evaluate its performance using ROC-AUC Score.

```python
# Question 41 - Train a Random Forest Classifier and evaluate its performance using ROC-AUC Score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Probabilities
probabilities = rf_model.predict_proba(X_test)[:, 1]

# ROC-AUC
auc = roc_auc_score(y_test, probabilities)
print("ROC-AUC Score:", auc)
```

---

## Question 42 - Train a Bagging Classifier and evaluate its performance using cross-validation.

```python
# Question 42 - Train a Bagging Classifier and evaluate its performance using cross-validation

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Model
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

# Cross-validation
scores = cross_val_score(bagging_model, X, y, cv=5)

print("Cross-validation Scores:", scores)
print("Average Score:", scores.mean())
```

---

## Question 43 - Train a Random Forest Classifier and plot the Precision-Recall curve.

```python
# Question 43 - Train a Random Forest Classifier and plot the Precision-Recall curve

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Load dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Probabilities
probabilities = rf_model.predict_proba(X_test)[:, 1]

# Precision-Recall values
precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

# Plot
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
```

---

## Question 44 - Train a Stacking Classifier with Random Forest and Logistic Regression and compare accuracy.

```python
# Question 44 - Train a Stacking Classifier with Random Forest and Logistic Regression and compare accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

# Stacking classifier
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

# Train model
stack_model.fit(X_train, y_train)

# Predictions
predictions = stack_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Stacking Accuracy:", accuracy)
```

---

## Question 45 - Train a Bagging Regressor with different levels of bootstrap samples and compare performance.

```python
# Question 45 - Train a Bagging Regressor with different levels of bootstrap samples and compare performance

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different bootstrap sample levels
sample_levels = [0.5, 0.7, 1.0]

for sample in sample_levels:
    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=50,
        max_samples=sample,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    print(f"Bootstrap Sample Level: {sample}, MSE: {mse}")
```

---

# End of Assignment
