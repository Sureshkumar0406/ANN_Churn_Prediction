import streamlit as st 
import pandas as pd
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

# Load the ANN model
model = tf.keras.models.load_model('model1.h5')

## load the encoder and scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Streamlit App

st.title("Customer CHurn Prediction")

#User Input

credit_score = st.number_input('CreditScore') 
Geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_) 
Age = st.number_input('Age',18,92) 
Tenure = st.slider('Tenure',0,10)  
Balance = st.number_input('Balance')
No_of_products = st.slider('NumOfProducts',1,4)
Has_cr_card = st.selectbox('HasCrCard',[0,1])
Active_Member = st.selectbox('IsActiveMember',[0,1])  
Salary = st.number_input('EstimatedSalary')




#Prepare the INput data

input_data = pd.DataFrame({
CreditScore: credit_score,
Geography: Geography,
Gender: label_encoder_gender.transform([Gender])[0],
Age: Age,
Tenure: Tenure,
Balance: Balance,
No_of_products: No_of_products,
Has_cr_card: Has_cr_card,
Active_Member: Active_Member,
Salary: Salary
  
})

#One hot encoded Geography
geo_encoder=onehot_encoder_geo.transform(data1[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoder,columns =onehot_encoder_geo.get_feature_names_out(['Geography']) )

#Combine onehot encoded with input_data
input_data = pd.concat([input_data,reset_index( drop = True), geo_encoded_df],axis =1)

#Scale the input_data

scaled_input_data = scaler.transform(input_data)


#Prediction churn

prediction = model.predict(scaled_input_data)
prediction_proba = prediction[0][0]


if prediction_proba > 0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')