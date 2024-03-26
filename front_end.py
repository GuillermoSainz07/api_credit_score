import streamlit 

import json
import requests

import streamlit as st
import joblib
import pickle
from data_science_process.contant import columns_to_predict
import pandas as pd
import numpy as np

st.header('Credit Score with Machine Learning ☄️​​💴​')
st.write("")


st.write(f'Select the feature inputo to make a prediction with Extra Trees Algorithm')

col1, col2 = st.columns(2)


with col1:
    month_salary = st.number_input("Enter your credit limit:")
    bank_accounts = st.number_input("Num of bank accounts:", step=1)
    num_credit_card = st.number_input("Num of credit cards:", step=1)
    interest_rate = st.number_input("Interest Rate:")
    delay_date = st.number_input('Delay Date:', step=1)
    credit_inquiries = st.number_input('Num Credit Inquiries:',step=1)
    credit_utilization = st.number_input('Credit Utilization Ratio:')
    emi_per_month = st.number_input('EMI Per Month:')
    age = st.number_input('Age:',step=1)
    annual_income = st.number_input('Anual Income:')
    num_loan = st.number_input('Num of Loan:',step=1)
    delay_payment = st.number_input('Num_of_Delayed_Payment:',step=1)
    changed_limit = st.number_input('Changed Limit:')
    outstanding_debt = st.number_input('Outstanding Debt:') 

with col2:
    amount_invested = st.number_input('Amount_invested_monthly:')
    month_balance = st.number_input('Monthly_Balance:')
    occupation = st.selectbox("Select you occupation:",
                              options=['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
                                        'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
                                        'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'])
    history_age = st.number_input('History Age:', step=1)
    paymen_min = st.selectbox('Payment_of_Min_Amount',
                              options=['No', 'Yes'])
    payment_behaviour = st.selectbox('Payment_Behaviour',
                                     options=['High_spent_Small_value_payments',
                                              'Low_spent_Large_value_payments',
                                                'Low_spent_Medium_value_payments',
                                                'Low_spent_Small_value_payments',
                                                'High_spent_Medium_value_payments',
                                                'High_spent_Large_value_payments'])
    payday_loan = st.selectbox('Payday Loan',
                               options=['Yes','No'])
    personal_loan = st.selectbox('Personal Loan',
                               options=['Yes','No'])
    mortgage_loan = st.selectbox('Mortage Loan',
                               options=['Yes','No'])
    student_loan = st.selectbox('Student Loan',
                               options=['Yes','No'])
    auto_loan = st.selectbox('Auto Loan',
                               options=['Yes','No'])
    credit_builder_loan = st.selectbox('Credit Builder Loan',
                               options=['Yes','No'])
    home_equity_loan = st.selectbox('Home Equity Loan',
                               options=['Yes','No'])
    debt_cons_loan = st.selectbox('Debt Cons Loan',
                               options=['Yes','No'])

tonum_dict = {'Yes':1,
            'No':0}


inputs = [month_salary, bank_accounts,num_credit_card, interest_rate,
          delay_date,credit_inquiries,credit_utilization,
          emi_per_month,age,annual_income,num_loan,delay_payment,changed_limit,
          outstanding_debt,amount_invested,month_balance,occupation,history_age,
          paymen_min,payment_behaviour,tonum_dict[payday_loan],tonum_dict[personal_loan],
          tonum_dict[mortgage_loan],tonum_dict[student_loan],tonum_dict[auto_loan],
          tonum_dict[credit_builder_loan],tonum_dict[home_equity_loan],tonum_dict[debt_cons_loan]]

inputs = np.array(inputs).reshape(1,-1)
inputs = pd.DataFrame(inputs, columns=columns_to_predict, index=[0])

### With api
#if st.button("Make predictions ☄️"):
    #res = requests.post(url="http://127.0.0.1:8000/predict_model", data=json.dumps(inputs))
    #st.subheader(f"Response from ML API☄️: {res.json()['Prediction']}")

model = joblib.load('ml_model/model.pkl')

to_str_pred = {0:'Poor',
               1:'Standard',
               2:'Good'}

if st.button("Make predictions ☄️"):

    prediction = model.predict(inputs)[0]
    st.subheader(f"Credit Score Predicted☄️: {to_str_pred[prediction]}")


st.subheader('Download the model below ⬇️​')
st.download_button(
    f"Download ✅​",
    data=pickle.dumps(model),
    file_name="et_model.pkl"
)