import os
import json

from flask import Flask, request, render_template, redirect, url_for
from waitress import serve

from predict import make_prediction

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def root():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company_name = request.form['param']
    score, proba = make_prediction(company_name)
    if score is not None and proba is not None:
        return json.dumps({"score": score, "proba": proba})
    else:
        return json.dumps({"score": None, "proba": None})


if __name__=="__main__":
    app.run()
    # serve(app, listen="127.0.0.1:5000", expose_tracebacks=True, log_socket_errors=True)