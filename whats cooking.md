
# Whats cooking

The script sets up a machine learning pipeline to classify cuisines from the "What's Cooking?" dataset using a Random Forest classifier. It reads the dataset, performs one-hot encoding on the 'cuisine' column, and splits the data into features and target variables. The dataset is then divided into training and testing sets. The Random Forest classifier is trained on the training data, and predictions are made on the test data. Finally, the script evaluates the classifier's performance by calculating the accuracy. However, it misses handling the 'ingredients' column, which should be vectorized for better feature representation.


## About project
1: Objective: Classify cuisine types based on ingredients using machine learning.

2: Data Source: Uses the "What's Cooking?" dataset in JSON format.

3: Feature Engineering: One-hot encodes the 'cuisine' column to convert categorical data into numerical format.

4: Data Concatenation: Merges one-hot encoded columns back with the original DataFrame.

5: Column Dropping: Removes the original 'cuisine' column and non-feature columns like 'id' and 'ingredients'.

6: Feature and Target Split: Separates the dataset into feature variables (X) and target variables (y).

7: Data Splitting: Divides the dataset into training and testing sets with an 80-20 split.

8: Model Selection: Uses a Random Forest classifier for the classification task.

9: Model Training: Trains the classifier on the training data.

10: Model Evaluation: Predicts on the test data and calculates the accuracy to evaluate the model's performance.






## Installation
1: Pandas: For data manipulation and analysis.

2: Scikit-learn: For machine learning algorithms and model evaluation.
## Result
The output will be like this:

    Accuracy: 1.0