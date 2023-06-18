import pickle 
from flask import Flask,request,jsonify,render_template
## jsonify such that we return our result in the form of json
## render-template is probably responsible for finding out the url of html file 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application


# Our web application should be able to interact with ridge.pickle standard.pickle bcz it will be able to interact and it will be able to do transformation function, transformation means feature scaling 

## import ridge regressor and standard scaler pickle 
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')         ## HEre as soon as we write render template it will find index.html and always make sure it is inside templates folder  

@app.route('/predictdata',methods=['GET','POST'])      ## searching for google.com by default we get a static page this becomes a GET request and probably if we go and search for pwskills  and execute here we can see we are posting some information over here i.e., we aer sending some query based on that query we are retriving the results 
def predict_datapoint():
    if request.method=="POST":         ##  whenever it is post we need to interact with ridge model and we need to do prediction and get output  
        Temperature=float(request.form.get('Temperature'))       ## These all are the input forms 
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        ## Once we get input parameters we use standard scalar for doing the transformation for new datapoint 
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])     ## We will showing result in the home.html

            
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)

## When we given by default 0.0.0.0 as our host address this is basically mapped to the local ip address of any machine that we are working 
