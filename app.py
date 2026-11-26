import streamlit as st
import pandas as pd
import tensorflow
from tensorflow.keras.models import load_model
import pickle

st.title("Titanic Survival Prediction using Deep Learning")
pclass = st.selectbox("Select passenger's class",[1,2,3])
sex = st.selectbox("Select gender",['male','female'])
age = st.slider("What is the passenger's age?",1,100)
sibsp = st.slider("Select total number of Siblings & Spouce",0,8)
parch = st.slider("Select total number of Parent & Child",0,6)
fare = st.number_input("Enter fare")
embarked = st.selectbox("From where was the journey started? { S-Southampton, C- Chebourg, Q- Queenstown }",['S','C','Q'])

user = {'Pclass':pclass, 'Sex':sex, 'Age':age, 'SibSp':sibsp, 'Parch':parch, 'Fare':fare,'Embarked':embarked}
user = pd.DataFrame([user])
if st.button("View Passenger's details"):
    st.write(user)

with open('label_encoder.pkl','rb') as file:
    label = pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

model = load_model('model.h5')

user['Sex'] = label.transform(user['Sex'])

emb = onehot.transform(user[['Embarked']])
emb = pd.DataFrame(emb,columns=onehot.get_feature_names_out())
user = user.drop(columns=['Embarked'])
user = pd.concat([user,emb], axis=1)

user[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = scaler.transform(user[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])

output = model.predict(user)
if st.button('Predict'):
    st.write("Chance of passenger's survival:",output[0][0])
    if output>0.5:
        st.write("The passenger will survive.")
    else:
        st.write("The passenger will not survive.")