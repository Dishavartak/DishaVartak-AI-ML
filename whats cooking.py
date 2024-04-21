import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_json("D:/Disha/Diploma/AI ML/whats-cooking/train.json/train.json")

# Perform one-hot encoding on the 'cuisine' variable
one_hot_encoded = pd.get_dummies(df['cuisine'], prefix='Cuisine')

# Concatenate the one-hot encoded columns with the original DataFrame
df_encoded = pd.concat([df, one_hot_encoded], axis=1)

# Drop the original 'cuisine' column
df_encoded.drop('cuisine', axis=1, inplace=True)

# Split the dataset into features (X) and target variable (y)
X = df_encoded.drop(['id', 'ingredients'], axis=1)
y = df_encoded[['Cuisine_greek', 'Cuisine_southern_us', 'Cuisine_filipino', 'Cuisine_indian',
                'Cuisine_jamaican', 'Cuisine_spanish', 'Cuisine_italian', 'Cuisine_mexican',
                'Cuisine_chinese', 'Cuisine_british', 'Cuisine_thai', 'Cuisine_vietnamese']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a classification algorithm (Random Forest Classifier)
clf = RandomForestClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)