# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:52:27 2022

@author: u391435
"""


import streamlit as st
import json
import requests

st.title("Calculate")

age = st.text_input("Enter first number","0")
salary = st.text_input("Enter second number","5")

operation = st.selectbox("select operation:", ["Addition","Substraction"])

# press button then the results will display
if st.button("prediction"):
    #spinner to show loading/computing
  url = 'http://127.0.0.1:8000/model'

  request_data = json.dumps({'age':40,'salary':20000})
  response = requests.post(url,request_data)
  print(response.text)
