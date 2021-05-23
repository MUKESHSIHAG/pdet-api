def check_health(knee_ax, knee_ay, knee_az, knee_gx, knee_gy, knee_gz, back_ax, back_ay, back_az, back_gx, back_gy, back_gz, chest_ax, chest_ay, chest_az, chest_gx, chest_gy, chest_gz, lying, sitting, standing):

  import pandas as pd
  from tensorflow import keras
  import tensorflow as tf
  import pickle
  import numpy as np

  cols = ['knee_ax','knee_ay','knee_az','knee_gx','knee_gy','knee_gz','back_ax', 'back_ay', 'back_az', 'back_gx', 'back_gy','back_gz', 'chest_ax','chest_ay','chest_az','chest_gx','chest_gy','chest_gz','lying','sitting','standing']
  df= pd.DataFrame([[knee_ax, knee_ay, knee_az, knee_gx, knee_gy, knee_gz, back_ax, back_ay, back_az, back_gx, back_gy, back_gz, chest_ax, chest_ay, chest_az, chest_gx, chest_gy, chest_gz, lying, sitting, standing]], columns = cols)
  print(df.head())
  sc = pickle.load(open('scaler.pkl','rb'))
  column_names_to_normalize = ['knee_ax','knee_ay','knee_az','knee_gx','knee_gy','knee_gz','back_ax', 'back_ay', 'back_az', 'back_gx', 'back_gy','back_gz', 'chest_ax','chest_ay','chest_az','chest_gx','chest_gy','chest_gz']
  x = df[column_names_to_normalize].values
  x_scaled = sc.transform(x)
  df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize)
  df[column_names_to_normalize] = df_temp

  classifier = keras.models.load_model('classifier_h.h5')
  # classifier = tf.lite.TFLiteConverter.from_keras_model('classifier')
  q = classifier.predict(np.array( [df.iloc[0],] ) )
  if(q <= 0.5):
    q = 0
  else:
    q = 1 
  return q    
