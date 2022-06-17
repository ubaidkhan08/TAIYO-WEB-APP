import streamlit as st
from joblib import dump, load
import numpy as np
import pandas as pd


def main():
    st.title("Natural Gas Prediction Model for Taiyo")
    html_temp = """
    <div style="background-color:teal ;padding:20px">
    </div>
    """

    st.subheader('Crude Oil Price')
    COP = st.number_input('(in Dollars)')


    st.subheader('US Dollar Index')
    UDSI = st.number_input('Enter')


    st.subheader('Texas Temperature')
    TT = st.slider('(In Farnheit)', 0.0, 110.0)


    st.subheader('California Temperature')
    CT = st.slider('(In Farnheit)', 0.0, 110.0)

    
    st.subheader('Natural Gas Production in the USA')
    NGP = st.number_input('(in Trillion cubic feet)')


    st.subheader('GDP of USA')
    GDP = st.number_input('(in Billion Dollars)')


    st.subheader('US Natural Gas Reserves')
    NGR = st.number_input('(in Trillion cubic feet)')

    st.subheader('US Natural Gas Consumption')
    NGC = st.number_input('(in Trillion cubic feet)')

    st.subheader('US Natural Gas Imports')
    NGI = st.number_input('(in Trillion cubic feet)')
    
    if st.button('Predict Natural Gas Price'):
        output= classify(COP,USDI,TT,CT,NGP,GDP,NGR,NGC,NGI)
        #st.success()
        st.success(output)

        
def classify(COP,USDI,TT,CT,NGP,GDP,NGR,NGC,NGI):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    inputs = [[COP,USDI,TT,CT,NGP,GDP,NGR,NGC,NGI]]

    from joblib import dump, load
    log_model = load('naturalgas.joblib')
    predictionn = round((log_model.predict(inputs)),3)
    return('Predicted Price:',predictionn)
    
if __name__=='__main__':
    main()
