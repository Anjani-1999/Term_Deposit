import numpy as np
import pickle
import pandas as pd
import streamlit as st

gfc_model = open("Bankdeposit_gfc.pkl","rb")
gfc_mod = pickle.load(gfc_model)
st.write("""
# Bank Term Deposit

- The goal is to predict whether a client will subscribe a term deposit (Target Variable ) with the help of a given set of independent variables.

## This app helps the bank marketting team to identify potential customer who would really like to subscribe for a fixed deposit in their bank

Data obtained from the (https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

""")




from PIL import Image
image = Image.open('bank1.jpg')
st.sidebar.image(image,
         use_column_width=True)
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">',unsafe_allow_html=True)
st.write("campaign: This is the campaign conducted by bank marketting team  ")

def run():
    #st.sidebar.info("This app is created using pycaret and streamlit")
    #salary = st.sidebar.number_input("Salary")
    balance = st.number_input("Bank Balance")
    fifth,sixth=st.beta_columns(2)
    day = fifth.number_input("Date of campaign ", 1, 31)

    frst,sec=st.beta_columns(2)
    third,fourth=st.beta_columns(2)
    campaign = frst.number_input("No.of contacts performed during this campaign for this client")
    pdays = sec.number_input("No.of days that passed by after the client was last contacted from a previous campaign")
    previous = third.number_input("No.of contacts performed in previous campaign for this client")
    duration = fourth.number_input("Duration of customer with Bank Marketting Team(in minutes")
    a,b=st.beta_columns(2)
    age_group = a.selectbox("Age Group", [20, 30, 40, 50])
    month_int = sixth.selectbox("Month", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    marital = b.selectbox("Marital status", ["Single", "Married", "Divorcerd"])
    if marital == "Married":
        marital_int = 1
    elif marital == "Single":
        marital_int = 2
    else:
        marital_int = 3
    p,q=st.beta_columns(2)
    education = p.selectbox("Education", ["Primary", "Secondary", "Teritiary"])
    if education == "Primary":
        education_int = 1
    elif education == "Secondary":
        education_int = 2
    else:
        education_int = 3
    job = q.selectbox("Job",
                       ["management", "technician", "entrepreneur", "blue-collar", "retired", "admin.", "services",
                        "self-employed", "unemployed", "student", "housemaid", "other"])
    if job == "management":
        Job_int = 1
    elif job == "technician":
        Job_int = 2
    elif job == "entrepreneur":
        Job_int = 3
    elif job == "blue-collar":
        Job_int = 4
    elif job == "retired":
        Job_int = 5
    elif job == "admin.":
        Job_int = 6
    elif job == "services":
        Job_int = 7
    elif job == "self-employed":
        Job_int = 8
    elif job == "unemployed":
        Job_int = 9
    elif job == "student":
        Job_int = 10
    elif job == "housemaid":
        Job_int = 11
    else:
        Job_int = 12
    r,s=st.beta_columns(2)

    poutcome = r.selectbox("Outcome of the previous marketing campaign", ["Success", "Failure", "Unknown"])
    if poutcome == "Success":
        poutcome_int = 2
    elif poutcome == "Failure":
        poutcome_int = 1
    else:
        poutcome_int = 3
    housing_loan = s.selectbox("Housing Loan", ["Yes", "No"])
    if housing_loan == "Yes":
        housing_binary = 1
    else:
        housing_binary = 0

    default = st.selectbox("Default Credit", ["Yes", "No"])
    if default == "Yes":
        default_binary = 1
    else:
        default_binary = 0
    personal_loan = st.selectbox("Personal Loan", ["Yes", "No"])
    if personal_loan == "Yes":
        loan_binary = 1
    else:
        loan_binary = 0


    result = ""

    if st.button("Predict"):
        result = gfc_mod.predict([[balance, day, duration, campaign, pdays, previous, age_group, month_int,
                                 marital_int,Job_int, education_int, poutcome_int, housing_binary, default_binary, loan_binary
                                 ]])
    #st.success(result)
    #st.write(result)
    #term_dep=np.array(["This person is not interested to subscribe for Term Deposit","This person is interested to subscribe for Term Deposit"])
    #st.write(term_dep[result[0]])
    if result==0:
        st.success("This person is not interested to subscribe for Term Deposit")
    else:
        st.success("This person is interested to subscribe for Term Deposit")
    st.subheader("Prob")
    st.write(gfc_mod.predict_proba([[balance, day, duration, campaign, pdays, previous, age_group, month_int,
                                 marital_int,Job_int, education_int, poutcome_int, housing_binary, default_binary, loan_binary
                                 ]]))

if __name__ == "__main__":
    run()
