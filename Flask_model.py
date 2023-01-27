from flask import Flask, request, jsonify
import pickle
import re
import os
import numpy as np
import json

# print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
# print("PATH:", os.environ.get('PATH'))

with open('hw1.pkl', 'rb') as pkl_file:
   regressor = pickle.load(pkl_file)
# print(type(regressor))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.data[3:]
    arg = json.loads(data)

    num = arg['numbers']
    # num = request.get_json()['numbers']
    
    return jsonify({
        'result': regressor.predict(np.array(num).reshape(1, -1))[0]
    })

if __name__ == '__main__':
    app.run('localhost', 5000, debug=True)