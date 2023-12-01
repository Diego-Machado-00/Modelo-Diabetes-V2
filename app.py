from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("models/model_perceptron.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = request.form.get("Age")
    sex = request.form.get("Sex")
    physActivity = request.form.get("PhysActivity")
    fruits = request.form.get("Fruits")
    veggies = request.form.get("Veggies")
    hvyAlcoholConsump = request.form.get("HvyAlcoholConsump")
    smoker = request.form.get("Smoker")
    highBP = request.form.get("HighBP")
    highChol = request.form.get("HighChol")
    bMI = request.form.get("BMI")
    genHlth = request.form.get("GenHlth")
    physHlthad = request.form.get("PhysHlth")
    diffWalk = request.form.get("DiffWalk")
    heartDiseaseorAttack = request.form.get("HeartDiseaseorAttack")
    prediction = model.predict([[age, sex, physActivity, fruits, veggies, hvyAlcoholConsump, smoker, highBP, highChol, bMI, genHlth, physHlthad, diffWalk, heartDiseaseorAttack]])
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