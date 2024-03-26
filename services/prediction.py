from schemas.features import Features
import joblib
import numpy as np
import pandas as pd
from tables.pred_table import PredTable
from data_science_process.contant import columns_to_predict


class ModelPrediction:
    def __init__(self, db) -> None:
        self.db = db

    def make_prediction(self, **kwargs):
        model = joblib.load("ml_model/model.pkl")


        #features = [value for key, value in kwargs.items()]
        #features = pd.DataFrame(features, columns=columns_to_predict, index=[0])
        features = pd.DataFrame(kwargs, index=[0])
        features.columns = columns_to_predict

        predict = model.predict(features)[0]
        predict_label = {0:'Poor',
                         1:'Standard',
                         2:'Good'}
        predict_sting = predict_label[predict]
        
        new_instance = PredTable(**kwargs)
        new_instance.prediction = predict_sting

        self.db.add(new_instance)
        self.db.commit()

        return predict