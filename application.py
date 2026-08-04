from flask import Flask,request,render_template,jsonify
from src.piplines.prediction_pipelines import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('form.html')

    else:
        data=CustomData(
            carat=float(request.form.get('carat'))if request.form.get('carat') else 0.0,
            depth=float(request.form.get('depth'))if request.form.get('depth') else 0.0,
            table=float(request.form.get('table'))if request.form.get('table') else 0.0,
            x=float(request.form.get('x'))if request.form.get('x') else 0.0,
            y=float(request.form.get('y'))if request.form.get('y') else 0.0,
            z=float(request.form.get('z'))if request.form.get('z') else 0.0,
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')

        )

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)
        return render_template('form.html',final_result=results)

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
