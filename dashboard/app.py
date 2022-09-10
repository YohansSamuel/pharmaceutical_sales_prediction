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


# displays user overview analysis
def customer_purchase_behavior():
    st.title("Customer Purchase Behavior Analysis")
# 
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
    st.header('Sample Data from the experience dataset')

def sales_prediction_dl():
    st.title("Sales Prediction using Deep Learning(LSTM)")
    st.header('Sample Data from the experience dataset')
    

pages = {
    "Welcome Page":home,
    "Customer Behavior Analysis": customer_purchase_behavior,
    "Sales Prediction using Random Forest": sales_prediction_ml,
    "Sales Prediction using LSTM": sales_prediction_dl,
}


options = st.sidebar.selectbox("Rossman Pharmaceutical Sales Prediction",list(pages.keys()))
pages[options]()


