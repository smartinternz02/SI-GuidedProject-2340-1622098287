import numpy as np
from flask import Flask, request, render_template
import pickle
import joblib
model = pickle.load(open('randomforest.pkl','rb'))
ct = joblib.load("column")
sc = pickle.load(open('standscale.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("loll.html")
@app.route('/display',methods=['POST'])
def result():
    satisfaction_level=request.form['satisfaction_level']
    last_evaluation=request.form['last_evaluation']
    number_project=request.form['number_project']
    average_montly_hours=request.form['average_montly_hours']
    time_spend_company=request.form['time_spend_company']
    Work_accident=request.form['Work_accident']
    promotion_last_5years=request.form['promotion_last_5years']
    Department=request.form['Department']
    Salary=request.form['Salary']
    data=[[satisfaction_level,last_evaluation,number_project,average_montly_hours,time_spend_company,Work_accident,promotion_last_5years,Department,Salary]]
    pred=model.predict(sc.transform(ct.transform(data)))
    return render_template("loll.html",y=pred)
    
if __name__ == "__main__":
    app.run(debug=True)