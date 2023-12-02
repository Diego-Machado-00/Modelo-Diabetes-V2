from flask import Flask, render_template, request, jsonify
from pydantic import basemodel
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("models/model_perceptron.pkl", "rb"))

class DiabetesItem(BaseModel):
    Age : int
    Sex : int
    PhysActivity : int
    Fruits : int
    Veggies : int
    HvyAlcoholConsump : int
    Smoker : int
    HighBP :int 
    HighChol : int
    BMI : int
    GenHlth : int
    PhysHlth : int
    DiffWalk : int
    HeartDiseaseorAttack : int


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict(item : DiabetesItem):
    df  = pd.DataFrame([ item.dict().values()], columns=item.dict().keys())
    prediction = model.predict(df)
    return render_template("index.html", predictions = prediction)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)  # Get data posted as a json
    age = data['Age']
    sex = data['Sex']
    physActivity = data['PhysActivity']
    fruits = data['Fruits']
    veggies = data['Veggies']
    hvyAlcoholConsump = data['HvyAlcoholConsump']
    smoker = data['Smoker']
    highBP = data['HighBP']
    highChol = data['HighChol']
    bMI = data['BMI']
    genHlth = data['GenHlth']
    physHlthad = data['PhysHlth']
    diffWalk = data['DiffWalk']
    heartDiseaseorAttack = data['HeartDiseaseorAttack']
    prediction = model.predict([[age, sex, physActivity, fruits, veggies, hvyAlcoholConsump, smoker, highBP, highChol, bMI, genHlth, physHlthad, diffWalk, heartDiseaseorAttack]])
    return jsonify({'prediction': prediction}) 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)