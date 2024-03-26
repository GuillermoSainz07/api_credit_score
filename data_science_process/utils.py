import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scikitplot import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from typing import List
import warnings

def encoder_y(data:str)-> pd.Series:
    encoder_dict = {'Poor':0,
                    'Standard':1,
                    'Good':2}
    
    return encoder_dict[data]


class AnalyzingModelPerformance:

    def __init__(self,
                 X_test:pd.DataFrame,
                 y_test: pd.Series,
                 models:list,
                 names:list):
        """
        models: List of intances of models
        names : list of strings with the name of the model
        """
        
        self.X_test = X_test
        self.y_test = y_test
        self.models = models
        self.names = names

        if len(names) %2 != 0:
            warnings.warn("n models is not even: Spaces without graphs", UserWarning)

        self.__n = int(np.ceil(len(names)/2))
        

    def models_reports(self) -> tuple:
        """
        return: Tuple of classification reports
        """

        cls_reports = (classification_report(self.y_test, model.predict(self.X_test))
                        for model,name in zip(self.models, self.names))
        
        return cls_reports 
    
    def plot_roc_auc_models(self, figsize = (12,10)) -> None:
        """
        return: Plot roc auc curves for each model
        """    
        
        fig, ax = plt.subplots(self.__n,2, figsize=figsize)
        for idx, model in enumerate(self.models):
            y_pred_proba = model.predict_proba(self.X_test)
            if len(self.names) == 2:
                metrics.plot_roc(self.y_test, y_pred_proba,
                                plot_micro=False,
                                plot_macro=False,
                                ax=ax[idx],
                                title=f'ROC {self.names[idx]} Model')
            else:
                metrics.plot_roc(self.y_test, y_pred_proba,
                                plot_micro=False,
                                plot_macro=False,
                                ax=ax[idx%2,idx//2],
                                title=f'ROC {self.names[idx]} Model')
            
        return

    def plot_cm_models(self, figsize = (12,10)) -> None:
        """
        return: Plot confusion matrix for each model
        """   
        
        fig, ax = plt.subplots(self.__n,2, figsize=figsize)
        for idx, model in enumerate(self.models):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            if len(self.names) == 2:
                ConfusionMatrixDisplay(cm).plot(ax=ax[idx])
                ax[idx].set_title(f'Confusion Matrix {self.names[idx]}', fontweight='bold')
            else:
                ConfusionMatrixDisplay(cm).plot(ax=ax[idx%2, idx//2])
                ax[idx%2, idx//2].set_title(f'Confusion Matrix {self.names[idx]}', fontweight='bold')

        return 

    def plot_cumulative_gains_models(self,figsize = (12,10)) -> None:
        """
        return: Plot cumulative gains curves for each model
        Warning: This plot should only be used to binary classification 
        """    
        
        fig, ax = plt.subplots(self.__n,2, figsize=figsize)
        for idx, model in enumerate(self.models):
            y_pred_proba = model.predict_proba(self.X_test)

            if len(self.names) == 2:
                metrics.plot_lift_curve(self.y_test,
                                        y_pred_proba,
                                        ax=ax[idx],
                                        title=f'Lift Curve {self.names[idx]} Model')
            else:
                metrics.plot_lift_curve(self.y_test,
                                        y_pred_proba,
                                        ax=ax[idx%2, idx//2],
                                        title=f'Lift Curve {self.names[idx]} Model')

        return 
        
    def plot_lift_curve_models(self,figsize = (12,10))-> None:
        """
        return: Plot lift curve aggregating for each model
        Warning: This plot should only be used to binary classification
        """    

        fig, ax = plt.subplots(self.__n,2, figsize=figsize)
        for idx, model in enumerate(self.models):
            y_pred_proba = model.predict_proba(self.X_test)
            if len(self.names) == 2:
                metrics.plot_cumulative_gain(self.y_test,
                                            y_pred_proba,
                                            ax=ax[idx],
                                            title=f'Cumulative Gain {self.names[idx]} Model')
            else:
                metrics.plot_cumulative_gain(self.y_test,
                                                y_pred_proba,
                                                ax=ax[idx%2, idx//2],
                                                title=f'Cumulative Gain {self.names[idx]} Model')
        return 
    