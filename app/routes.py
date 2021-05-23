from app import app
from flask import request, jsonify
import pandas as pd
from tensorflow import keras
import pickle
import numpy as np
import os

@app.route('/')
def index():    
    return "Pdet api is working properly"

@app.route('/get_result', methods=['GET'])
def get_result():
    knee_ax = int(request.args.get("knee_ax"))
    knee_ay = int(request.args.get("knee_ay"))
    knee_az = int(request.args.get("knee_az"))
    knee_gx = int(request.args.get("knee_gx"))
    knee_gy = int(request.args.get("knee_gy"))
    knee_gz = int(request.args.get("knee_gz"))
    back_ax = int(request.args.get("back_ax"))
    back_ay = int(request.args.get("back_ay"))
    back_az = int(request.args.get("back_az"))
    back_gx = int(request.args.get("back_gx"))
    back_gy = int(request.args.get("back_gy"))
    back_gz = int(request.args.get("back_gz"))
    chest_ax = int(request.args.get("chest_ax"))
    chest_ay = int(request.args.get("chest_ay"))
    chest_az = int(request.args.get("chest_az"))
    chest_gx = int(request.args.get("chest_gx"))
    chest_gy = int(request.args.get("chest_gy"))
    chest_gz = int(request.args.get("chest_gz"))
    position = request.args.get("position")
    sitting = 0
    standing = 0
    lying = 0
    print(type(knee_ax))
    print(knee_ax,knee_ay,knee_az,knee_gx,knee_gy,knee_gz)
    print(back_ax,back_ay,back_az,back_gx,back_gy,back_gz)
    print(chest_ax,chest_ay,chest_az,chest_gx,chest_gy,chest_gz)
    if position=='sitting': sitting=1
    elif position=='standing': standing=1
    else: lying=1
    res = check_health(knee_ax, knee_ay, knee_az, knee_gx, knee_gy, knee_gz, back_ax, back_ay, back_az, back_gx, back_gy, back_gz, chest_ax, chest_ay, chest_az, chest_gx, chest_gy, chest_gz, lying, sitting, standing)
    print(res)
    posture = "UnHealthy"
    if res==0:
        posture="healthy"
    return {"result": posture, "msg": "Success"}


def check_health(knee_ax, knee_ay, knee_az, knee_gx, knee_gy, knee_gz, back_ax, back_ay, back_az, back_gx, back_gy, back_gz, chest_ax, chest_ay, chest_az, chest_gx, chest_gy, chest_gz, lying, sitting, standing):
    cols = ['knee_ax','knee_ay','knee_az','knee_gx','knee_gy','knee_gz','back_ax', 'back_ay', 'back_az', 'back_gx', 'back_gy','back_gz', 'chest_ax','chest_ay','chest_az','chest_gx','chest_gy','chest_gz','lying','sitting','standing']
    df= pd.DataFrame([[knee_ax, knee_ay, knee_az, knee_gx, knee_gy, knee_gz, back_ax, back_ay, back_az, back_gx, back_gy, back_gz, chest_ax, chest_ay, chest_az, chest_gx, chest_gy, chest_gz, lying, sitting, standing]], columns = cols)
    print(df.head())
    sc = pickle.load(open(os.getcwd()+'\\app\\scaler.pkl','rb'))
    column_names_to_normalize = ['knee_ax','knee_ay','knee_az','knee_gx','knee_gy','knee_gz','back_ax', 'back_ay', 'back_az', 'back_gx', 'back_gy','back_gz', 'chest_ax','chest_ay','chest_az','chest_gx','chest_gy','chest_gz']
    x = df[column_names_to_normalize].values
    x_scaled = sc.transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize)
    df[column_names_to_normalize] = df_temp

    classifier = keras.models.load_model(os.getcwd()+'\\app\\classifier_h.h5')
    q = classifier.predict(np.array( [df.iloc[0],] ) )
    if(q <= 0.5):
        q = 0
    else:
        q = 1 
    return q    
