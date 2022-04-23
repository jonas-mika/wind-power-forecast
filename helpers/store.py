import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score
from matplotlib import pyplot as plt
import seaborn as sns

FILENAME = 'model.sav'

np.random.seed(10)

def save_model(new_model, X, y, X_test, y_test):
  new_model_preds = new_model.predict(X_test)
  new_model_score = np.round(new_model.score(X_test, y_test), 4)
  print(f"\nNew R2-Test Score: {new_model_score}")

  if FILENAME in os.listdir():
    saved_model = pickle.load(open(FILENAME, 'rb'))
    saved_model_preds = saved_model.predict(X_test)
    saved_model_score = np.round(saved_model.score(X_test, y_test), 4)

    print(f"Saved Model R2-Test Score: {saved_model_score}\n")

    res = pd.DataFrame(data={'speed': X_test[:, 0],
                             'y_test': y_test.reshape(-1), 
                             'new_model': new_model_preds.reshape(-1),
                             'saved_model': saved_model_preds.reshape(-1)
                             })
    print(res) 
    
    if new_model_score > saved_model_score:
      print('> saving new model')
      pickle.dump(new_model.fit(X, y), open(FILENAME, 'wb'))
    elif new_model_score < saved_model_score:
      print('> new model performs worse. not saving')
    else: 
      print('> the two models are identical')

  else:
    print('> No model saved so far. Saving New Model')
    pickle.dump(new_model.fit(X, y), open(FILENAME, 'wb'))

def get_current_performance(X_test, y_test):
  return saved_model_score

def forecast(X):
  saved_model = pickle.load(open(FILENAME, 'rb'))

  return saved_model.predict(X)
