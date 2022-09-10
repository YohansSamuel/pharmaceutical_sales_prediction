from cProfile import label
from tkinter import PAGES
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
# 
import sys
sys.path.insert(0,'./scripts/')
from user_overview import UserOverview
import user_engagement_dashboard
from data_plots import plot_hist,plot_mult_hist

#
import seaborn as sns
import matplotlib.pyplot as plt


# displays user overview analysis
def user_overview_analysis():
    st.title("User Overview Analysis")
    
    # original dataset
    st.markdown(f'<h1 style="color:green;font-size:24px;">{"Sample Data from the original dataset"}</h1>', unsafe_allow_html=True)
    df = pd.read_csv("./data/telecom_data_source.csv",index_col=0)
    st.write(df.head(10))

    # cleaned dataset
    st.markdown(f'<h1 style="color:green;font-size:24px;">{"Sample Data from the cleaned dataset"}</h1>', unsafe_allow_html=True)
    df = pd.read_csv("./data/cleaned_telecom_data_source.csv",index_col=0)
    st.write(df.head(10))
    
    # display top 10 handsets used by users
    st.header('Display top 10 handsets used by customers')
    user_ov = UserOverview(df)
    top_10_handsets = user_ov.get_top_handsets(10)
    st.write(top_10_handsets)

    fig = go.Figure(go.Pie(labels = top_10_handsets.index,values = top_10_handsets.values))
    st.header("Top 10 Handset Type used by customers")
    st.plotly_chart(fig)

    # display top 3 manufacturers
    st.header('Display top 3 handset manufacturers')
    top_3_manufacturers = user_ov.get_top_manufacturers(3)
    st.write(top_3_manufacturers)

    fig = go.Figure(go.Pie(labels = top_3_manufacturers.index,
    values = top_3_manufacturers.values
    ))
    st.header("Top 3 Handset Manufacturers")
    st.plotly_chart(fig)
# 
def user_engagement_analysis():
    st.title("User Engagement Analysis")
    df = pd.read_csv('./data/user_engagement.csv',index_col=0)
    user_engagement_df = df[['Cluster','Session Frequency','Duration','Total Data Usage']]
    user_engagement_df = user_engagement_df.groupby('Cluster').agg({'Session Frequency':'count',
    'Duration':'sum','Total Data Usage':'sum'})

    col = st.sidebar.selectbox(
        "Select top 10 from", (["Session Frequency", "Duration", "Total Data Usage"]))
    if col == "Sessions_Frequency":
        sessions = df.nlargest(10, "Session Frequency")['Session Frequency']
        return plot_hist(sessions)
    elif col == "Duration":
        duration = df.nlargest(10, "Duration")['Duration']

        return plot_mult_hist([duration], 1, 1, "User Engagement Duration", ['Duration (sec)'])

    else:
        total_data_volume = df.nlargest(
            10, "Total Data Usage")['Total Data Usage']
        
        return plot_mult_hist([total_data_volume], 1, 1, "User Engagement Total Data Usage", ['Total Data Usage (kbps)'])

# 
def user_experience_analysis():
    st.title("User Experience Analysis")
    st.header('Sample Data from the experience dataset')
    df = pd.read_csv('./data/user_experience.csv',index_col=0)
    st.write(df.head(10))

    st.header('Total Average RTT in ms')
    st.bar_chart(df['Total Avg RTT (ms)'])

    st.header('Total Average TCP in Bytes')
    st.bar_chart(df['Total Avg TCP (Bytes)'])


pages = {
    "Customer Behavior Analysis": user_overview_analysis,
    "Sales Prediction using Random Forest": user_engagement_analysis,
    "Sales Prediction using LSTM": user_experience_analysis,
}


options = st.sidebar.selectbox("Rossman Pharmaceutical Sales Prediction",list(pages.keys()))
pages[options]()


