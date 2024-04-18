from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__) 

# Load the data
data = pd.read_csv('data.csv')

# Create your machine learning model
x = data.drop(['label'], axis=1)
y = data['label']
model = DecisionTreeClassifier()
model.fit(x, y)

@app.route('/')
def home():
    return render_template('login-in.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        
        return render_template('index.html', prediction=prediction[0])

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            return "User already exists! Please choose a different username."
        users[username] = password  # Storing in dictionary (replace with database)
        session['username'] = username  # Log in user
        return redirect('/')
    return render_template('sign-up.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return render_template('/dashboard')
    return render_template('/dashboard')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/index')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
