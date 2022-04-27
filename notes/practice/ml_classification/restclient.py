# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:35:03 2022

@author: u391435
"""

import json
import requests 

url = 'http://localhost:8000/model'
request_data = json.dumps({'model':'knn'})
response = requests.post(url, request_data)

print(response.text)
