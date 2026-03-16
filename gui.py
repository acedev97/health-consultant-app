import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

body_parts = {
    "head": ["headache", "dizziness", "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", "loss_of_smell", "blurred_and_distorted_vision", "visual_disturbances", "pain_behind_the_eyes", "sinus_pressure", "congestion", "runny_nose", "throat_irritation", "redness_of_eyes", "patches_in_throat", "slurred_speech", "stiff_neck", "neck_pain"],
    "chest": ["chest_pain", "breathlessness", "fast_heart_rate", "palpitations", "cough", "phlegm", "blood_in_sputum", "rusty_sputum", "mucoid_sputum"],
    "stomach": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "indigestion", "nausea", "loss_of_appetite", "abdominal_pain", "constipation", "diarrhoea", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "passage_of_gases", "internal_itching", "belly_pain", "distention_of_abdomen", "stomach_bleeding", "swelling_of_stomach"],
    "skin": ["itching", "skin_rash", "nodal_skin_eruptions", "yellowish_skin", "dischromic _patches", "pus_filled_pimples", "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze", "bruising", "puffy_face_and_eyes", "drying_and_tingling_lips", "red_spots_over_body"],
    "limbs": ["joint_pain", "weakness_in_limbs", "knee_pain", "hip_joint_pain", "muscle_weakness", "swelling_joints", "movement_stiffness", "swollen_legs", "swollen_blood_vessels", "swollen_extremeties", "cramps", "painful_walking", "prominent_veins_on_calf"],
    "urinary": ["burning_micturition", "spotting_ urination", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine"],
    "general": ["continuous_sneezing", "shivering", "chills", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "high_fever", "sunken_eyes", "sweating", "dehydration", "mild_fever", "acute_liver_failure", "swelled_lymph_nodes", "malaise", "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium", "abnormal_menstruation", "watering_from_eyes", "increased_appetite", "polyuria", "family_history", "lack_of_concentration", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "history_of_alcohol_consumption", "fluid_overload", "excessive_hunger", "extra_marital_contacts", "muscle_wasting", "dark_urine", "yellow_urine", "yellowing_of_eyes", "enlarged_thyroid", "brittle_nails", "obesity"]
}

training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

model = SVC()
model.fit(x_train, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

serverityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {}

for index, symptoms in enumerate(x):
  symptoms_dict[symptoms] = index
  
def calc_condition(exp, days):
  sum = 0
  for item in exp:
    sum = sum + serverityDictionary[item]
  if ((sum * days) / (len(exp) + 1) > 13):
    return "You should take the consultation from doctor."
  else:
    return "It might not be that bad but you should take precautions."
  
def getDescription():
  global description_list
  with open('MasterData/symptom_Description.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      _description = {row[0]: row[1]}
      description_list.update(_description)
      
def getSeverityDict():
  global serverityDictionary
  with open('MasterData/symptom_severity.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    try:
      for row in csv_reader:
        _diction = {row[0]: int(row[1])}
        serverityDictionary.update(_diction)
    except:
      pass
      
def getprecautionDict():
  global precautionDictionary
  with open('MasterData/symptom_precaution.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
      precautionDictionary.update(_prec)
      
def getInfo():
    return "------------------------- HealthCare ChatBot --------------------------\nYour Name?"


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    disease = le.inverse_transform(node)
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names):
    tree_ = tree.tree


def predict_disease(symptoms):
  # Read in symptom severity, description, and precaution data
  getSeverityDict()
  getDescription()
  getprecautionDict()
  
  # Convert user input of symptoms to a list
  if symptoms is not None:
    symptoms_list = symptoms.split(',')
  else:
      symptoms_list = []

  # Convert symptom names to numbers
  symptom_indices = [symptoms_dict[symptom.strip()] for symptom in symptoms_list if symptom.strip() in symptoms_dict]

  # Create a vector with 0s and 1s indicating whether each symptom is present or not
  input_vector = np.zeros(len(symptoms_dict))
  input_vector[symptom_indices] = 1

  # Make prediction using decision tree
  prediction = print_disease(clf.predict([input_vector]))

  # Get descriptions and precautions for predicted disease
  descriptions = description_list.get(prediction[0], "No description available")
  precautions = precautionDictionary.get(prediction[0], ["No precautions available"])

  # Return prediction, descriptions, and precautions
  return prediction[0], descriptions, precautions


if __name__ == '__main__':
    st.title("HealthCare ChatBot")
    st.write("Welcome to the HealthCare ChatBot!")

    name = st.text_input("Your Name:")
    
    if name:
        st.write(f"Hello {name}!")
        
        body_part = st.selectbox("Which part of your body is sick?", list(body_parts.keys()))
        
        if body_part:
            potential_symptoms = body_parts[body_part]
            st.write(f"Potential symptoms for {body_part}:")
            selected_symptoms = st.multiselect("Select symptoms:", potential_symptoms)
            
            if selected_symptoms:
                days = st.number_input("How long have you experienced these symptoms? (in days)", min_value=1, step=1)
                
                if st.button("Predict Disease"):
                    symptoms_str = ','.join(selected_symptoms)
                    prediction, descriptions, precautions = predict_disease(symptoms_str)
                    st.subheader(f"Predicted Disease: {prediction}")
                    st.write(f"**Description:** {descriptions}")
                    st.write("**Precautions:**")
                    for prec in precautions:
                        if prec:
                            st.write(f"- {prec}")
                    # Additional advice
                    condition = calc_condition(selected_symptoms, days)
                    st.write(f"**Additional advice:** {condition}")