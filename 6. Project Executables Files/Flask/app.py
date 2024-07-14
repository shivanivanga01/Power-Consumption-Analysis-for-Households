from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('PCA_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("pca.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
        input_features = [float(x) for x in request.form.values()]
        features_name = ['Global_reactive_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        
        df = pd.DataFrame([input_features], columns=features_name)
        output = model.predict(df)
        
        return render_template('result.html', prediction_text=output)
    
if __name__ == "__main__":
    app.run(debug=False)
    
