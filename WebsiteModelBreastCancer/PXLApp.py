from flask import Flask, url_for, request, json, Response, jsonify, render_template, redirect, flash
from wtforms import  SubmitField, IntegerField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
from functools import wraps
import joblib
import numpy as np 
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.config['SECRET_KEY']='mysecret'

test_data_f =  { "points_mean" : 10, "dimension_mean" : 25000, "smoothness_se" : 110, "symmetry_se" : 1600, "radius_worst" : 1200, "perimeter_worst" : 570}  

class PredictForm(FlaskForm):
	points_mean = IntegerField('points_mean')
	dimension_mean = IntegerField('dimension_mean')
	smoothness_se = IntegerField('smoothness_se')
	symmetry_se = IntegerField('symmetry_se')
	radius_worst = IntegerField('radius_worst')
	perimeter_worst = IntegerField('perimeter_worst')
	submit = SubmitField('Calculate')

class Config(object):
	SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'

model_pkl = pickle.load(open('BreastCancerModel.pkl','rb'))

@app.route('/hi', methods = ['GET'])
def api_hi():
	data = {
		'hello': 'hiworld',
		'number': 456
	}
	js = json.dumps(data)

	resp = Response(js, status=200, mimetype='application/json')
	resp.headers['Link']= 'http://www.cteq.eu'
	return resp

@app.route('/start',methods=['GET','POST'])
def start():
	form = PredictForm()
	if form.validate_on_submit():
		flash('Breast Cancer')
		test_data_f['points_mean']=form.points_mean.data
		test_data_f['dimension_mean']=form.dimension_mean.data
		test_data_f['smoothness_se']=form.smoothness_se.data
		test_data_f['symmetry_se']=form.symmetry_se.data
		test_data_f['radius_worst']=form.radius_worst.data
		test_data_f['perimeter_worst']=form.perimeter_worst.data
		data = test_data_f
		result=model_pkl.predict(pd.DataFrame(pd.DataFrame(data, index=[0])))[0]      
		return render_template('result.html',title='Breast Cancer', form=form, M=result)
	return render_template('index.html', title='Breast Cancer', form=form)


# This can be used with curl to test the api/webserver
@app.route('/predict', methods=['POST'])
def price_predict():
	if request.method == 'POST':
	# Get the data from the POST method
		data = request.get_json(force=True)
	# Predict using Model loaded from pkl file 
	return jsonify(model_pkl.predict(pd.DataFrame(pd.DataFrame(data, index=[0])))[0])

@app.route('/apitest_json')
def apitest_json():
	test_data =  {"points_mean" : 10, "dimension_mean" : 25000, "smoothness_se" : 110, "symmetry_se" : 1600, "radius_worst" : 1200, "perimeter_worst" : 570}    
	return jsonify(model_pkl.predict(pd.DataFrame(pd.DataFrame(test_data, index=[0])))[0])

@app.route('/apitest')
def apitest():
	return jsonify(model_pkl.predict(pd.DataFrame([[10,25000,110,1600,1200]]))[0])

if __name__ == '__main__':
	app.run(debug=True)
