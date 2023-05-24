from fastapi import FastAPI
import os
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import requests
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

app = FastAPI()
templates = Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")

def model0(df):
    # load_model
    filename0 = 'models/0_Diabetes_DT.pkl'
    model0 = pickle.load(open(filename0, 'rb'))

    # load normalization
    scalerfile0 = 'models/0_Diabetes_.sav'
    scalerModel0 = pickle.load(open(scalerfile0, 'rb'))

    # create dataframe
    df = df[['patientSexName', 'patientAge','vital_bpd','vital_bps', 'vital_bmi', 'vital_waist' ,
               'exercise', 'alcohol', 'smoking','narcotic']]

    # predict
    x_norm0 = scalerModel0.transform(df)
    pred0 = model0.predict_proba(x_norm0)

    # explaination
    model = model0.best_estimator_
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    shap.summary_plot(shap_values, df, show=False)
    plt.savefig('static/modelexplainer.png')

    return pred0[0][1]

def model3(df):
    # load_model
    filename3 = 'models/3_Diabetes_ADB.pkl'
    model3 = pickle.load(open(filename3, 'rb'))

    # load normalization
    scalerfile3 = 'models/3_Diabetes_.sav'
    scalerModel3 = pickle.load(open(scalerfile3, 'rb'))

    # create dataframe
    df = df[['patientSexName', 'patientAge', 'vital_bpd', 'vital_bps', 'vital_bmi', 'vital_waist',
               'exercise', 'alcohol', 'smoking','hypertension_disease','hyperlipidaemia_disease',
         'Fasting_glucose','HbA1c','eGFR','LDL_cholesterol','HDL_cholesterol','Cholesterol','Triglyceride']]

    # predict
    x_norm3 = scalerModel3.transform(df)
    pred3 = model3.predict_proba(x_norm3)

    #model = model3.best_estimator_
    explainer = shap.LinearExplainer(model, df)
    shap_values = explainer.shap_values(df)
    shap.summary_plot(shap_values, df, show=False)
    plt.savefig('static/modelexplainer.png')

    return pred3[0][1]

def model7(df):
    # load_model
    filename7 = 'models/7_Diabetes_LR.pkl'
    model7 = pickle.load(open(filename7, 'rb'))

    # load normalization
    scalerfile7 = 'models/7_Diabetes_.sav'
    scalerModel7 = pickle.load(open(scalerfile7, 'rb'))

    # create dataframe
    df = df[['patientSexName', 'patientAge', 'vital_bpd', 'vital_bps', 'vital_bmi', 'vital_waist',
               'vital_pulse', 'vital_temperature', 'vital_rr','hypertension_disease','hyperlipidaemia_disease',
         'kidney_stones_disease' , 'chronic_kidney_disease' ,'ischaemi_heart_disease', 'stroke_disease',
        'gout_disease', 'pterygium_disease', 'chronic_obstructive_pulmonary_disease',
    'thyrotoxicosis_disease','Fasting_glucose','Creatinine','HbA1c','eGFR','LDL_cholesterol',
             'HDL_cholesterol','Cholesterol','BUN','Triglyceride','Potassium','Sodium','Chloride',
          'CO2']]

    x_norm7 = scalerModel7.transform(df)
    pred7 = model7.predict_proba(x_norm7)

    # explaination
    model = model7.best_estimator_
    explainer = shap.LinearExplainer(model, df)
    shap_values = explainer.shap_values(df)
    shap.summary_plot(shap_values, df, show=False)
    plt.savefig('static/modelexplainer.png')

    return pred7[0][1]


@app.get('/')
def main(request: Request):
    return templates.TemplateResponse('index.html', {'request':request})


@app.get('/predict')
def predict(request:Request, patientSexName: float, patientAge: float,vital_bpd: float,vital_bps: float, vital_weight: float,vital_height: float, vital_waist: float,
                vital_pulse: float, vital_temperature: float, vital_rr: float, exercise: float, alcohol: float,
                smoking: float, narcotic: float, hypertension_disease: float, hyperlipidaemia_disease: float, kidney_stones_disease: float,
                chronic_kidney_disease: float, ischaemi_heart_disease: float, stroke_disease: float, gout_disease: float, pterygium_disease: float,
                chronic_obstructive_pulmonary_disease: float, thyrotoxicosis_disease: float, Fasting_glucose: float,
                Creatinine: float, HbA1c: float, eGFR: float, LDL_cholesterol: float, HDL_cholesterol: float, Cholesterol: float, BUN: float, Triglyceride: float
                , Potassium: float, Sodium: float, Chloride: float, CO2: float):


    if os.path.exists('static/modelexplainer.png'):
        os.unlink('static/modelexplainer.png')

    vital_bmi = vital_weight / ((vital_height / 100) ** 2)

    dic = {'patientSexName': patientSexName, 'patientAge': patientAge, 'vital_bpd': vital_bpd, 'vital_bps': vital_bps,
           'vital_bmi': vital_bmi, 'vital_waist': vital_waist, 'vital_pulse': vital_pulse,
           'vital_temperature': vital_temperature, 'vital_rr': vital_rr, 'exercise': exercise, 'alcohol': alcohol,
           'smoking': smoking, 'narcotic': narcotic,
           'hypertension_disease': hypertension_disease, 'hyperlipidaemia_disease': hyperlipidaemia_disease,
           'kidney_stones_disease': kidney_stones_disease,
           'chronic_kidney_disease': chronic_kidney_disease, 'ischaemi_heart_disease': ischaemi_heart_disease,
           'stroke_disease': stroke_disease, 'gout_disease': gout_disease,
           'pterygium_disease': pterygium_disease,
           'chronic_obstructive_pulmonary_disease': chronic_obstructive_pulmonary_disease,
           'thyrotoxicosis_disease': thyrotoxicosis_disease,
           'Fasting_glucose': Fasting_glucose, 'Creatinine': Creatinine, 'HbA1c': HbA1c, 'eGFR': eGFR,
           'LDL_cholesterol': LDL_cholesterol,
           'HDL_cholesterol': HDL_cholesterol, 'Cholesterol': Cholesterol, 'BUN': BUN, 'Triglyceride': Triglyceride,
           'Potassium': Potassium, 'Sodium': Sodium, 'Chloride': Chloride, 'CO2': CO2}

    temp_df = pd.DataFrame(dic, index=[0])

    df0 = temp_df[['patientAge', 'vital_bpd', 'vital_bps', 'vital_bmi', 'vital_waist']]
    df3 = temp_df[['Fasting_glucose', 'HbA1c', 'eGFR', 'LDL_cholesterol', 'HDL_cholesterol', 'Cholesterol', 'Triglyceride']]
    df7 = temp_df[['vital_pulse', 'vital_temperature', 'vital_rr', 'Fasting_glucose', 'Creatinine', 'HbA1c', 'eGFR',
                   'LDL_cholesterol', 'HDL_cholesterol', 'Cholesterol', 'BUN', 'Triglyceride', 'Potassium', 'Sodium', 'Chloride', 'CO2']]

    # find zero
    df0_cal = (df0 == 0).sum(axis=1)
    df3_cal = (df3 == 0).sum(axis=1)
    df7_cal = (df7 == 0).sum(axis=1)

    # df0 = p0 = model 1
    # df3 = p3 = model 2
    # df7 = p7 = model 3

    if df0_cal[0] > 0:
        response = "Please fill all the required feature"
        response2 = ""
    elif df7_cal[0] <= 3:
        p7 = model7(temp_df)
        response = "You are at approx {output:.2f}% risk of type 2 diabetes.".format(output=100 * p7)
        response2 = "The model {model} was used to predict your profile and the references are in the below.".format(model='3')

    elif df3_cal[0] <= 2:
        p3 = model3(temp_df)
        response = "You are at approx {output:.2f}% risk of type 2 diabetes.".format(output=100 * p3)
        response2 = "The model {model} was used to predict your profile and the references are in the below.".format(model='2')
    else:
        p0 = model0(temp_df)
        response = "You are at approx {output:.2f}% risk of type 2 diabetes.".format(output=100 * p0)
        response2 = "The model {model} was used to predict your profile and the references are in the below.".format(model='1')

    result = response
    result2 = response2

    imgpath = request.url_for('static', path='modelexplainer.png')

    return templates.TemplateResponse('index.html', context={'request':request, 'prediction_text':result, 'prediction_text2':result2, 'img_url':imgpath})

if __name__ == '__main__':
      uvicorn.run(app)