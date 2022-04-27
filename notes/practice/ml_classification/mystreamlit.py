# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:52:27 2022

@author: u391435
"""


import streamlit as st
import json
import requests

st.title("Calculate")

age = st.text_input("Enter age","20")
salary = st.text_input("Enter salary","20000")

# press button then the results will display
if st.button("prediction"):
  url = 'http://127.0.0.1:8000/model'

  request_data = json.dumps({'age':age,'salary':salary})
  response = requests.post(url,request_data)

  if response.text=="the prediction is [0]":
    st.write("will not purchase insurance")
  else:
    st.write("will purchase")  
