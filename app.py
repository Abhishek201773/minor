import random
from random import choice, sample
import warnings
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import xgboost as xgb


warnings.filterwarnings("ignore")


flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/predict",methods = ["POST"])
def predict():
    
    df=pd.read_csv('student_dataset.csv')
    X = df.drop('Eligibility', axis=1)  # Features
    y = df['Eligibility']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_encoded = pd.get_dummies(X, columns=['Core Branch', 'Programming Language', 'Other Technical Skills', 'Company'])
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    xgb_classifier = xgb.XGBClassifier(random_state=42)
    xgb_classifier.fit(X_train_encoded, y_train_encoded)


    
    A="Congratulations! You are eligible for the program"
    B="Sorry, you are not eligible for the program."
    Ans="  "

    def predict_eligibility(communication_skills, core_branch, programming_language, technical_skills, certification_course, backlog,tenth_score, twelfth_score, attendance_percentage, company):
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
        input_df_encoded = pd.get_dummies(input_df, columns=['Core Branch', 'Programming Language', 'Other Technical Skills', 'Company'])
        missing_features = list(set(X_train_encoded.columns) - set(input_df_encoded.columns))
        for feature in missing_features:
             input_df_encoded[feature] = 0
             input_df_encoded = input_df_encoded[X_train_encoded.columns]
             eligibility_prediction = xgb_classifier.predict(input_df_encoded)[0]
             if eligibility_prediction == 1:
                  eligibility_message = "Congratulations! You are eligible for the program."
                  Ans+=eligibility_message
             else:
                 eligibility_message = "Sorry, you are not eligible for the program."
                 Ans+=eligibility_message
                 return eligibility_message
            
             
             
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
            

    return render_template("index.html", prediction_text = "PLACEMENT RESULT: {}".format(Ans))










if __name__ == "__main__":
    flask_app.run(debug=True)