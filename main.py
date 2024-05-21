import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import streamlit as st

# Load data
df = pd.read_csv("college project final.csv")

# Label encoding
lt = LabelEncoder()
df['GENDER'] = lt.fit_transform(df['GENDER'])
df['1.Experiencing ongoing stress symptoms?'] = lt.fit_transform(df['1.Experiencing ongoing stress symptoms?'])
df['2.Feeling down or sad?'] = lt.fit_transform(df['2.Feeling down or sad?'])
df['3.Dealing with constant anxiety?'] = lt.fit_transform(df['3.Dealing with constant anxiety?'])
df['4.Experiencing social isolation lately?'] = lt.fit_transform(df['4.Experiencing social isolation lately?'])
df['5.Caught in overthinking patterns?'] = lt.fit_transform(df['5.Caught in overthinking patterns?'])
df['6.Having trouble sleeping well?'] = lt.fit_transform(df['6.Having trouble sleeping well?'])
df['7.Experiencing changes in weight?'] = lt.fit_transform(df['7.Experiencing changes in weight?'])
df['8.Dealing with frequent panic attacks?'] = lt.fit_transform(df['8.Dealing with frequent panic attacks?'])
df['9.Frequent headaches or migraines?'] = lt.fit_transform(df['9.Frequent headaches or migraines?'])
df['10.Noticing changes in mood?'] = lt.fit_transform(df['10.Noticing changes in mood?'])

# Fill missing values
df.fillna({'11.Do you find it difficult to concentrate or make decisions?':'0'}, inplace=True)
df.fillna({'12.Do you face nightmare frequently?':'0'}, inplace=True)
df.fillna({'13.Continuous fall in academic marks?':'0'}, inplace=True)
df.fillna({'AGE ':'0'}, inplace=True)

# Select features and target
x = df.iloc[:, [4, 5, 7, 8, 9, 10, 11]].values
y = df.iloc[:, [6]].values

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train logistic regression model
lo = LogisticRegression()
lo.fit(x_train, y_train)

# Predictions and evaluation
pred = lo.predict(x_test)
re = confusion_matrix(y_test, pred)
print(re)
print("Accuracy:", accuracy_score(y_test, pred))

# Streamlit frontend
st.title("Mental Health Prediction")

# Define label encoder for user input with all possible categories
lt_gender = LabelEncoder()
lt_gender.fit(['Male', 'Female'])

lt_binary = LabelEncoder()
lt_binary.fit(['Yes', 'No'])

gender = st.selectbox('Gender', ('Please select', 'Male', 'Female'))
experiencing_stress = st.selectbox('Experiencing ongoing stress symptoms?', ('Please select', 'Yes', 'No'))
feeling_down = st.selectbox('Feeling down or sad?', ('Please select', 'Yes', 'No'))
social_isolation = st.selectbox('Experiencing social isolation lately?', ('Please select', 'Yes', 'No'))
overthinking = st.selectbox('Caught in overthinking patterns?', ('Please select', 'Yes', 'No'))
trouble_sleeping = st.selectbox('Having trouble sleeping well?', ('Please select', 'Yes', 'No'))
changes_in_weight = st.selectbox('Experiencing changes in weight?', ('Please select', 'Yes', 'No'))

# Submit button
if st.button('Submit'):
    if 'Please select' in [gender, experiencing_stress, feeling_down, social_isolation, overthinking, trouble_sleeping, changes_in_weight]:
        st.write("Please fill in all fields.")
    else:
        # Encode user input
        gender_encoded = lt_gender.transform([gender])[0]
        experiencing_stress_encoded = lt_binary.transform([experiencing_stress])[0]
        feeling_down_encoded = lt_binary.transform([feeling_down])[0]
        social_isolation_encoded = lt_binary.transform([social_isolation])[0]
        overthinking_encoded = lt_binary.transform([overthinking])[0]
        trouble_sleeping_encoded = lt_binary.transform([trouble_sleeping])[0]
        changes_in_weight_encoded = lt_binary.transform([changes_in_weight])[0]

        user_input = np.array([[experiencing_stress_encoded, feeling_down_encoded, social_isolation_encoded, overthinking_encoded, trouble_sleeping_encoded, changes_in_weight_encoded, gender_encoded]])
        prediction = lo.predict(user_input)

        if prediction[0] == 1:
            st.write("You may be experiencing some mental health issues. Please consider seeking help from a professional.")
        else:
            st.write("You seem to be mentally fit. Keep maintaining a healthy lifestyle and reach out if you ever need help.")
