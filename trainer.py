import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from data_script.data import Data
import pandas as pd
import numpy as np


class Trainer:
    def __init__(self):
        self.data = Data()

    def train(self):
        x_train, x_test, y_train, y_test = self.data.get_data()
        x_train_woutNan, x_test_woutNan, y_train_woutNan, y_test_woutNan = self.data.get_data_woutNanEdit()

        print("\nNumber of features: " + str(len(self.data.data_prepared.columns)))
        print("Number of rows " + str(len(self.data.data_prepared)))
        print("Shape: " + str(self.data.data_prepared.shape))

        print("\nNumber of features without Nan editing: " + str(len(self.data.data_prepared_woutNanEdit.columns)))
        print("Number of rows without Nan editing: " + str(len(self.data.data_prepared_woutNanEdit)))
        print("Shape: " + str(self.data.data_prepared_woutNanEdit.shape))

        self.linear_regression_try(x_test_woutNan, x_train_woutNan, y_test_woutNan, y_train_woutNan)

    def linear_regression_try(self, x_test_woutNan, x_train_woutNan, y_test_woutNan, y_train_woutNan):
        self.data.data_prepared_woutNanEdit.plot(x='noRooms', y='baseRent', style='o')
        plt.title('noRooms vs baseRent')
        plt.xlabel('noRooms')
        plt.ylabel('baseRent')
        plt.show()
        plt.figure(figsize=(15, 10))
        plt.tight_layout()
        seabornInstance.distplot(self.data.data_prepared_woutNanEdit['baseRent'])
        plt.show()
        # Ein Objekt für lineare Regression erzeugen
        ols = LinearRegression()
        # Je höher der Wert von MSE, desto schlechter ist das Modell.
        print("Kreuzvalidierung der linearen Regression mittels (negativem) MSE: " +
              str(cross_val_score(ols, x_train_woutNan, y_train_woutNan, scoring='neg_mean_squared_error')))
        # Je näher R2 an 1 liegt,desto besser ist das Modell.
        print("Kreuzvalidierung der linearen Regression mittels R-Quadrat: " +
              str(cross_val_score(ols, x_train_woutNan, y_train_woutNan, scoring='r2')))
        # training the algorithm
        ols.fit(x_train_woutNan, y_train_woutNan)
        print("linearRegression_intercept: " + str(ols.intercept_))
        # For retrieving the slope:
        print("linearRegression_coef: " + str(ols.coef_))
        target_predict = ols.predict(x_test_woutNan)
        df = pd.DataFrame({'Actual': y_test_woutNan.values.flatten(), 'Predicted': target_predict.flatten()})
        print("\nResults: " + str(df))
        df1 = df.head(25)
        df1.plot(kind='bar', figsize=(16, 10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.legend()
        plt.show()
        print(str(x_test_woutNan.shape))
        print(y_test_woutNan.shape)
        print(target_predict.shape)

        print("Debug")
