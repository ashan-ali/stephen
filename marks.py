from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
#-------------------------------------------------------------------------------------
data = pd.read_csv("Student_Marks.csv")
x = data.loc[:, ['time_study']]
y = data.loc[:, ['Marks']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 23)

model = LinearRegression()
model.fit(x_train, y_train)
#Streamlit interface......
st.title("Exam Score Predictor")
hour = st.number_input("Enter Hours Studies:")
if st.button("Predict Score"):
    predicted_scores = model.predict([[hour]])[0]
    st.success(f"Predicted score is ==>{predicted_scores}")
st.write ("###Sample Training Data")
st.dataframe(data)
