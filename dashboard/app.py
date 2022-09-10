from cProfile import label
from tkinter import PAGES
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
# 
import sys
# sys.path.insert(0,'./scripts/')

#
import seaborn as sns
import matplotlib.pyplot as plt

import datetime

# displays user overview analysis
def customer_purchase_behavior():
    st.title("Customer Purchase Behavior")

def home():
    st.markdown("<h1 style='padding:2rem;text-align:left; \
            background-color:blue;color:black;font-size:1.8rem;\
            border-radius:0.2rem;'>Rosmann Pharmaceuticals Sales Prediction</h1>", 
            unsafe_allow_html=True)
    st.write("---")
    st.write(
    """
   The finance team at Rossmann Pharmaceuticals wants to forecast 
   sales in all their stores across several cities six weeks ahead of time.
    """
    )

    st.subheader("The dataset")
    st.write(" the dataset should be of the following form:")
    st.write(pd.read_csv("./data/cleaned_train.csv"))

    st.subheader("Sales trend")

def sales_prediction_ml():
    st.title("Sales Prediction using Machine Learning(Random Forest)")
    number = st.number_input('Enter a StoreID')
    st.write('The enterd StoreID is ', number)

    d = st.date_input("Choose a Date of Prediction", datetime.date(2019, 7, 6))
    st.write('The selected date is:', d)

    prediction_day = st.radio(
     "Choose a prediction date",
     ('Monday', 'Tuesday', 'Wednesday','Thursday','Friday'))

    if prediction_day == 'Monday':
        st.write('You selected Monday.')
    elif prediction_day == 'Tuesday':
        st.write('You selected Tuesday.')
    elif prediction_day == 'Wednesday':
        st.write('You selected Wednesday.')
    elif prediction_day == 'Thursday':
        st.write('You selected Thursday.')
    else:
        st.write("You selected Friday.")

    promotion_day = st.radio(
     "Promotion day ?",('Yes', 'No'))

    if promotion_day == 'Yes':
        st.write('You selected Yes.')
    else:
        st.write("You selected No.")

    school_holiday = st.radio(
     "School Holiday",('Yes', 'No'))

    if school_holiday == 'Yes':
        st.write('You selected Yes.')
    else:
        st.write("You selected No.")

    state_holiday = st.radio(
     "State Holiday",('None', 'Christmas','Easter Holiday','Public Holiday'))

    if state_holiday == 'None':
        st.write('You selected None.')
    elif state_holiday == 'Christmas':
        st.write('You selected Christmas.')
    elif state_holiday == 'Easter Holiday':
        st.write('You selected Easter Holiday.')
    else:
        st.write("You selected Public Holiday.")

    if st.button('Predict'):
     st.write('The prediction goes here')

    st.write('The predicted sales is: $')

def sales_prediction_dl():
    st.title("Sales Prediction using Deep Learning(LSTM)")
    

pages = {
    "Home":home,
    "Customer Behavior Analysis": customer_purchase_behavior,
    "Sales Prediction using Random Forest": sales_prediction_ml,
    "Sales Prediction using LSTM": sales_prediction_dl,
}


options = st.sidebar.selectbox("Rossman Pharmaceutical Sales Prediction",list(pages.keys()))
pages[options]()


