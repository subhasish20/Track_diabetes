from flask import  Flask
from flask import  render_template
from flask import  request
import pickle
import  pandas as pd

model = pickle.load(open('pipeline_random_forest.pkl','rb'))
app = Flask(__name__,template_folder='templates',static_folder='static')

@app.route('/')
def home():
    return  render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_data = {
        'gender': request.form['gender'],
        'age': float(request.form['age']),
        'hypertension': int(request.form['hypertension']),
        'heart_disease': int(request.form['heart_disease']),
        'smoking_history': request.form['smoking_history'],
        'bmi': float(request.form['bmi']),
        'HbA1c_level': float(request.form['hba1c']),
        'blood_glucose_level': float(request.form['glucose'])
    }


    # Convert to DataFrame (assuming your model expects these columns in the pipeline)
    input_df = pd.DataFrame([user_data])

    # Predict using the loaded model
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        return  render_template('effected.html',data=user_data)
    else:
        return render_template('not_effected.html',data=user_data)


if __name__ == '__main__':
    app.run(debug=True)