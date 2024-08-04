from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('salary-calculater.pkl', 'rb'))


@app.route("/predict", methods=['POST'])
def predict():
    experience = int(request.form['experience'])
    exam_score = int(request.form['exam_score'])
    interview_score = int(request.form['interview_score'])
    prediction = model.predict([[experience, exam_score, interview_score]])
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction[0]))


@app.route("/")
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
