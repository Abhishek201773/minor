# -*- coding: utf-8 -*-
"""NewMP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bgJRVzIJyUi7Mf3Qx_3ZgZxGHgJJRic5
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import random
from random import choice, sample
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

random_seed = 42  # You can use any integer value as the seed
np.random.seed(random_seed)
random.seed(random_seed)

# Number of samples
num_samples = 3000

# Define categorical variables
core_branches = ['AIML', 'DS', 'CS', 'Cyber Security']
programming_languages = ['Python', 'Java', 'C++', 'Kotlin']
technical_skills = ['Machine Learning', 'Data Analytics', 'DBMS', 'DSA', 'Web development', 'Android Development']

# Generate random data for input features
data = {
    'Communication Skills': np.random.randint(5, 11, size=num_samples),  # Scale: 5 to 10
    'Core Branch': [choice(core_branches) for _ in range(num_samples)],
    'Programming Language': [', '.join(sample(programming_languages, np.random.randint(1, len(programming_languages) + 1))) for _ in range(num_samples)],
    'Other Technical Skills': [', '.join(sample(technical_skills, np.random.randint(1, len(technical_skills) + 1))) for _ in range(num_samples)],
    'Certification Course': np.random.randint(0, 2, size=num_samples),  # Binary: 0 or 1
    'Backlog': np.random.randint(0, 2, size=num_samples),  # Binary: 0 or 1
    '10th Score (%)': np.random.uniform(60, 100, size=num_samples),  # Scale: 50 to 100
    '12th Score (%)': np.random.uniform(60, 100, size=num_samples),
    'Attendance (%)': np.random.uniform(60, 100, size=num_samples),
    'Company': [choice(['Company A', 'Company B', 'Company C']) for _ in range(num_samples)]  # Company as a feature
}

# Generate random eligibility labels (1 for eligible, 0 for not eligible) based on new conditions
eligibility = []
for i in range(num_samples):
    if (data['Core Branch'][i] == 'AIML' and data['Communication Skills'][i] > 7 and 'Python' in data['Programming Language'][i] and 'Machine Learning' in data['Other Technical Skills'][i]) or \
       (data['Core Branch'][i] == 'DS' and data['Communication Skills'][i] > 8 and 'Python' in data['Programming Language'][i] and 'Data Analytics' in data['Other Technical Skills'][i]) or \
       (data['Communication Skills'][i] > 6 and any(language in data['Programming Language'][i] for language in ['C++', 'Java']) and 'DSA' in data['Other Technical Skills'][i]) or \
       (data['Communication Skills'][i] > 8 and 'Kotlin' in data['Programming Language'][i] and any(language in data['Programming Language'][i] for language in ['C++', 'Java']) and 'DSA' in data['Other Technical Skills'][i]) or \
       (data['Communication Skills'][i] > 6 and ('Web development' in data['Other Technical Skills'][i] and 'DBMS' in data['Other Technical Skills'][i])) or \
       (any(skill in data['Other Technical Skills'][i] for skill in ['DSA', 'DBMS'])) or \
       (data['Core Branch'][i] == 'Cyber Security'):
        eligibility.append(0)  # Eligible
    else:
        eligibility.append(1)  # Not eligible

# Create a DataFrame
df = pd.DataFrame(data)
df['Eligibility'] = eligibility  # Target column indicating eligibility

# Save the dataset to a CSV file
df.to_csv('student_dataset.csv', index=False)

df

df['Eligibility'].value_counts()

# Importing necessary libraries
from sklearn.model_selection import train_test_split

# Load your dataset from the CSV file
df = pd.read_csv('student_dataset.csv')

# Separate features (X) and target variable (y)
X = df.drop('Eligibility', axis=1)  # Features
y = df['Eligibility']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting sets to confirm the split

# Importing necessary libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Perform one-hot encoding on categorical features
X_encoded = pd.get_dummies(X, columns=['Core Branch', 'Programming Language', 'Other Technical Skills', 'Company'])

# Split the encoded dataset into training and testing sets (80% train, 20% test)
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)

# Train the classifier on the training data
xgb_classifier.fit(X_train_encoded, y_train_encoded)

def predict_eligibility(communication_skills, core_branch, programming_language, technical_skills, certification_course, backlog,
                        tenth_score, twelfth_score, attendance_percentage, company):
    # Create a DataFrame for the input features
    input_data = {
        'Communication Skills': [communication_skills],
        'Certification Course': [certification_course],
        'Backlog': [backlog],
        '10th Score (%)': [tenth_score],
        '12th Score (%)': [twelfth_score],
        'Attendance (%)': [attendance_percentage],
        'Core Branch': [core_branch],
        'Programming Language': [programming_language],
        'Other Technical Skills': [technical_skills],
        'Company': [company]
    }

    input_df = pd.DataFrame(input_data)

    # Perform one-hot encoding on the input features
    input_df_encoded = pd.get_dummies(input_df, columns=['Core Branch', 'Programming Language', 'Other Technical Skills', 'Company'])

    # Ensure the input features have the same columns as the training data
    missing_features = list(set(X_train_encoded.columns) - set(input_df_encoded.columns))
    for feature in missing_features:
        input_df_encoded[feature] = 0

    # Ensure that only the relevant columns are considered
    input_df_encoded = input_df_encoded[X_train_encoded.columns]

    eligibility_prediction = xgb_classifier.predict(input_df_encoded)[0]

    if eligibility_prediction == 1:
        eligibility_message = "Congratulations! You are eligible for the program."
    else:
        eligibility_message = "Sorry, you are not eligible for the program."

    return eligibility_message

import pickle

pickle.dump(xgb_classifier,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

# Sample test cases
test_cases = [
    (8, 'AIML', 'Python', 'Machine Learning', 1, 0, 92, 88, 95, 'Company A'),
    (9, 'DS', 'Python', 'Data Analytics', 1, 0, 94, 89, 92, 'Company B'),
    (7, 'CS', 'Java', 'DSA, Web development', 1, 0, 85, 80, 88, 'Company C'),
    (8, 'Cyber Security', 'Python', 'Web development, Android Development', 1, 0, 88, 84, 90, 'Company A')

]

for test_case in test_cases:
    communication_skills, core_branch, programming_language, technical_skills, certification_course, backlog, tenth_score, twelfth_score, attendance_percentage, company = test_case
    eligibility_prediction = predict_eligibility(communication_skills, core_branch, programming_language, technical_skills,
                                                 certification_course, backlog, tenth_score, twelfth_score, attendance_percentage,
                                                 company)
    print(f"Input Features: {test_case}")
    print(f"Eligibility Criteria: {eligibility_prediction}")

