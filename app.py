#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import nbimporter
from FYP_csv_model import connect_and_predict


# In[ ]:


app = Flask(__name__)
@app.route('/data', methods=['GET'])
def get_data():
    result = connect_and_predict()
    print(result)

if __name__ == "__main__":
    app.run()

