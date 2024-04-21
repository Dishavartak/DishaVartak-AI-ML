import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
url = "D:/Disha/Diploma/AI ML/Loan Prediction/train_u6lujuX_CVtuZ9i.csv"
loan_df = pd.read_csv(url)

missing_values = loan_df.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)

# Impute missing values for numerical columns with the mean
numerical_columns = loan_df.select_dtypes(include=['int64', 'float64']).columns
loan_df[numerical_columns] = loan_df[numerical_columns].fillna(loan_df[numerical_columns].mean())

# Impute missing values for categorical columns with the mode
categorical_columns = loan_df.select_dtypes(include=['object']).columns
loan_df[categorical_columns] = loan_df[categorical_columns].fillna(loan_df[categorical_columns].mode().iloc[0])

# Verify if there are any missing values left
missing_values = loan_df.isnull().sum()
print("Missing values in the dataset after imputation:")
print(missing_values)

# Drop Loan_ID column as it is not useful for prediction
loan_df.drop('Loan_ID', axis=1, inplace=True)

# Convert categorical variables to numerical using LabelEncoder
le = LabelEncoder()
for col in loan_df.select_dtypes(include='object'):
    loan_df[col] = le.fit_transform(loan_df[col])

# Split the dataset into features and target variable
X = loan_df.drop('Loan_Status', axis=1)
y = loan_df['Loan_Status']

# Oversampling using RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("Original class distribution:", Counter(y))
print("Resampled class distribution:", Counter(y_resampled))

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# Initialize and train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
