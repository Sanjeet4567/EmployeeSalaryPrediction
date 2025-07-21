from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
application.secret_key = "your_secret_key_here"  # Use a strong random secret key

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            age=int(request.form.get('age')),
            fnlwgt=int(request.form.get('fnlwgt')),
            educational_num=int(request.form.get('educational_num')),
            capital_gain=int(request.form.get('capital_gain')),
            capital_loss=int(request.form.get('capital_loss')),
            hours_per_week=int(request.form.get('hours_per_week')),
            workclass=request.form.get('workclass'),
            education=request.form.get('education'),
            marital_status=request.form.get('marital_status'),
            occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            gender=request.form.get('gender'),
            native_country=request.form.get('native_country')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        # Save result in session and redirect to avoid form resubmission
        result = "> 50K" if pred[0] == 1 else "<50K"
        session['final_result'] = result

        return redirect(url_for('predict_datapoint'))

    # GET request - show form with result if available
    final_result = session.pop('final_result', None)
    return render_template('form.html', final_result=final_result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
