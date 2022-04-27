# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:18:24 2022

@author: u391435
"""

from flask import Flask, request
app = Flask(__name__)

@app.route('/model',methods=['POST'])
def hello_world():
    request_data = request.get_json(force=True)
    # print(request_data)
    model_name = request_data['model']
    return "you are requesting for a {0} model".format(model_name)

if __name__ == "__main__":
    app.run(port=8000, debug=True)
    