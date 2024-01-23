from flask import Flask, current_app
from flask import request
import pandas as pd
import os
from pycaret.regression import *

app = Flask(__name__)

df_neighborhood = pd.read_excel('assets/neighborhood_median_sorocaba.xlsx')

def get_neighborhood_median(neighborhood:str):
    return df_neighborhood[df_neighborhood['neighborhood']==neighborhood]['neighborhood_median'].values[0]

@app.route('/')
def home():
    return current_app.send_static_file('index.html')

@app.route('/flask')
def flask():
    return "<p>Flask is running!</p>"

@app.route('/neighborhoods')
def neighborhoods():
    return {'neighborhoods':df_neighborhood['neighborhood'].to_list()}

@app.route('/neighborhoodMedian', methods=['GET'])
def neighborhood_median():
    return str(df_neighborhood[df_neighborhood['neighborhood']==request.args.get('neighborhood')]['neighborhood_median'].values[0])

@app.route('/models')
def models():
    return {'models':sorted(os.listdir('models/'))}

@app.route('/predict', methods=['POST'])
def predict():
    body = request.json
    model_name = body['model'].replace('.pkl','')
    data = body['data']
    data['neighborhood_median'] = get_neighborhood_median(data['neighborhood'])
    model = load_model('models/'+model_name)
    x_data = pd.DataFrame(data,index=[0])
    x_data['totalAreas'] = pd.to_numeric(x_data['totalAreas'])
    x_data['usableAreas'] = pd.to_numeric(x_data['usableAreas'])
    x_data['parkingSpaces'] = pd.to_numeric(x_data['parkingSpaces'])
    x_data['bathrooms'] = pd.to_numeric(x_data['bathrooms'])
    x_data['suites'] = pd.to_numeric(x_data['suites'])
    x_data['bedrooms'] = pd.to_numeric(x_data['bedrooms'])
    x_data['condominium'] = pd.to_numeric(x_data['condominium'])
    x_data['BACKYARD'] = pd.to_numeric(x_data['BACKYARD'])
    x_data['GYM'] = pd.to_numeric(x_data['GYM'])
    x_data['POOL'] = pd.to_numeric(x_data['POOL'])
    x_data['BARBECUE_GRILL'] = pd.to_numeric(x_data['BARBECUE_GRILL'])
    y_pred = predict_model(model,x_data)['prediction_label'].values[0]
    return f"R$ {round(y_pred,2):_}".replace('.',',').replace('_','.')

if __name__ == '__main__':
    app.run(debug=True)