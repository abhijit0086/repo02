# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:29:27 2022

@author: u391435
"""

import json
import requests 

url = 'http://127.0.0.1:8000/model'

request_data = json.dumps({'age':40,'salary':20000})
response = requests.post(url,request_data)
print(response.text)



