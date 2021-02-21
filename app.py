from flask import Flask,render_template,request
import numpy as np 
import pickle
app=Flask(__name__)
model=pickle.load(open('LinearReg.pkl','rb'))

@app.route ('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    features=[int(i) for i in request.form.values()]
    features=[np.array(features)]
    #features=features.reshape(-1,1)
    predictions=model.predict(features)
    outputs=predictions[0]
    return render_template('index.html',prediction_text='Predicted is {}'.format(outputs))
    


if __name__== '__main__':
    app.run(debug=True)
