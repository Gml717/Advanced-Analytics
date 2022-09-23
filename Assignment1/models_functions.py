from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn import metrics


class RemoveMissinVvalueCol(BaseEstimator, TransformerMixin):
    """ Reduce the dataset extent by only allowing columns to exhibit a treshold of real values. """
    def __init__(self, treshold):
        self.treshold = treshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        X_.dropna(thresh=self.treshold * len(X_), axis=1, inplace=True)
        return X_


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that deals with dates, binary and categorical encoding """
    # Class constructor method 
    def __init__(self):
        pass

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Helper function that converts values to Binary depending on input
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'

    #Helper function that bin postal_code to respective belgian province
    def postal_code_cleaning(self, code):
        if code < 1000:
            return np.NaN
        elif 1000 <= code <= 1299:
            return 'Brussels'
        elif 1300 <= code <= 1499:
            return 'Waals-Brabant'
        elif 1500 <= code <= 1999:
            return 'Vlaams-Brabant'
        elif 3000 <= code <= 3499:
            return 'Vlaams-Brabant'
        elif 2000 <= code <= 2999:
            return 'Antwerpen'
        elif 3500 <= code <= 3999:
            return 'Limburg'
        elif 4000 <= code <= 4999:
            return 'Luik'
        elif 5000 <= code <= 5999:
            return 'Namen'
        elif 6000 <= code <= 6599:
            return 'Henegouwen'
        elif 7000 <= code <= 7999:
            return 'Henegouwen'
        elif 6600 <= code <= 6999:
            return 'Luxemburg'
        elif 8000 <= code <= 8999:
            return 'West-Vlaanderen'
        elif 9000 <= code <= 9999:
            return 'Oost-Vlaanderen'
        

    # Helper function that bin age into equal intervals
    # Thresholds are observed from the training data
    def handle_age(self, value):

        if pd.isna(value):
            return 'unknown'
        else:
            if value < 20:
                return '<20'
            elif 20 <= value < 30:
                return '[20,30['
            elif 30 <= value < 40:
                return '[30,40['
            elif 40 <= value < 50:
                return '[40,50['
            elif 50 <= value < 60:
                return '[50,60['
            elif 60 <= value < 70:
                return '[60,70['
            elif 70 <= value < 80:
                return '[70,80['
            elif 80 <= value < 90:
                return '[80,90['
            else:
                return '>=90'

    def transform(self, X, y=None):

        # Convert these columns to binary for one-hot-encoding later
        filter_col = [col for col in X if col.startswith('has_')] + ['customer_gender', 'customer_self_employed',
                                                                     "homebanking_active"]
        for column in filter_col:
            X.loc[:, column] = X[column].apply(self.create_binary)
        
        # data Featurization of date variables
        X['province'] = X.customer_postal_code.apply(self.postal_code_cleaning)

        X['customer_birth_date'] = X['customer_birth_date'].apply(pd.to_datetime)

        X['customer_age'] = ((pd.to_datetime('today') - X.customer_birth_date) / np.timedelta64(1, 'Y')).apply(np.floor)

        X["customer_age"] = X["customer_age"].apply(self.handle_age)

        X.drop(['customer_birth_date', 'customer_postal_code'], axis=1, inplace=True)

        return X



class NumericalTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that deals with dates """
    # Class Constructor
    def __init__(self):
        pass

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self
    
    # data Featurization of date variables
    def clean_dates(self, df):
        df[['customer_since_all', 'customer_since_bank']] = df[['customer_since_all', 'customer_since_bank']].apply(
            pd.to_datetime)

        df.loc[:, 'customer_all_dur'] = (pd.to_datetime('today') - df.customer_since_all).dt.days
        df.loc[:, 'customer_bank_dur'] = (pd.to_datetime('today') - df.customer_since_bank).dt.days

        return df.drop(['customer_since_all', 'customer_since_bank'], axis=1)

    # Custom transform method we wrote that creates mentioned features and drops redundant ones
    def transform(self, X, y=None):
        X_ = X.copy()
        data = self.clean_dates(X_)
        return data


def evaluation(model,X_train, y_train,X_test,y_test):
    """evaluation of model using ROC CURV """
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)


    fpr, tpr, thresholds = metrics.roc_curve(y_test, ypred, pos_label=None)
    print(metrics.auc(fpr, tpr))

    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                               cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10))

    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()




