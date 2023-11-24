import streamlit as st
import pandas as pd
import numpy as np

st.title("Emissions de CO2")

df2013 = pd.read_csv(CO2/gov2013.csv,encoding='latin1', sep=';')

st.dataframe(df2013.head())
