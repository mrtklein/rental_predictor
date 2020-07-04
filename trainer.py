import os
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from data_script.data import Data
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from data_script.data_preparation import Data_Preparation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib


class Trainer:
    def __init__(self):
        self.data = Data()
        self.preparation = Data_Preparation()
        self.x_train, self.x_test, self.y_train, self.y_test = self.data.get_data()
        self.inputData_prepared, self.outputData = self.data.get_data_full()
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
        self.path_model = path + '\\models\\'
        self.models = glob.glob(self.path_model + r"*.pkl")

    def train(self):
        print("\nNumber of features: " + str(len(self.data.data_prepared.columns)))
        print("Number of rows " + str(len(self.data.data_prepared)))
        print("Shape: " + str(self.data.data_prepared.shape))

        print("\nNumber of features without Nan editing: " + str(len(self.data.data_prepared_woutNanEdit.columns)))
        print("Number of rows without Nan editing: " + str(len(self.data.data_prepared_woutNanEdit)))
        print("Shape: " + str(self.data.data_prepared_woutNanEdit.shape))

        self.linear_regression()
        self.decisionTreeRegression()
        self.randomForestRegression()

        #Load a Model
        # model=joblib.load("mymodel.pkl")

    def linear_regression(self):
        print(
            "\n***********************************************************************Linear Regression:")
        # Ein Objekt für lineare Regression erzeugen
        lin_reg = LinearRegression()

        # # training the algorithm
        lin_reg.fit(self.inputData_prepared, self.outputData)
        # print("linearRegression_intercept: " + str(ols.intercept_))
        # For retrieving the slope:
        # print("linearRegression_coef: " + str(ols.coef_))
        y_predict = lin_reg.predict(self.inputData_prepared)

        df = pd.DataFrame({'Actual': self.outputData.values.flatten(), 'Predicted': y_predict.flatten()})
        print("\nResults of prediciting inputData_prepared: \n" + str(df))
        df1 = df.head(25)
        df1.plot(kind='bar', figsize=(16, 10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.legend()
        plt.show()

        lin_mse = mean_squared_error(self.outputData, y_predict)
        lin_rmse = np.sqrt(lin_mse)
        print("RMSE of linear regression: " + str(lin_rmse))
        lin_scores = cross_val_score(lin_reg, self.inputData_prepared, self.outputData,
                                     scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        print("CrossValidation cv=10: ")
        self.display_scores(lin_rmse_scores)

        fileCounter = self.getFileCounter("lin_reg_%s.pkl")
        joblib.dump(lin_reg, self.path_model + "lin_reg_%s.pkl" % fileCounter)

    def decisionTreeRegression(self):
        print(
            "\n***********************************************************************DecisionTree Regression: Predict inputData_prepared")
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(self.inputData_prepared, self.outputData)

        y_prediction = tree_reg.predict(self.inputData_prepared)
        tree_mse = mean_squared_error(self.outputData, y_prediction)
        tree_rmse = np.sqrt(tree_mse)
        print("RMSE of decisionTreeRegression: " + str(tree_rmse))
        print("CrossValidation cv=10: ")
        scores = cross_val_score(tree_reg, self.inputData_prepared, self.outputData, scoring="neg_mean_squared_error",
                                 cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        self.display_scores(tree_rmse_scores)
        fileCounter = self.getFileCounter("tree_reg_%s.pkl")
        joblib.dump(tree_reg, self.path_model + "tree_reg_%s.pkl" % fileCounter)

    def randomForestRegression(self):
        print(
            "\n***********************************************************************RandomForestRegression: Predict inputData_prepared")
        forest_reg = RandomForestRegressor()
        forest_reg.fit(self.inputData_prepared, self.outputData)
        y_prediction = forest_reg.predict(self.inputData_prepared)
        forest_mse = mean_squared_error(self.outputData, y_prediction)
        forest_rmse = np.sqrt(forest_mse)
        print("RMSE of randomForestRegression: " + str(forest_rmse))

        print("CrossValidation cv=10: ")
        scores = cross_val_score(forest_reg, self.inputData_prepared, self.outputData, scoring="neg_mean_squared_error",
                                 cv=10)
        forest_rmse_scores = np.sqrt(-scores)
        self.display_scores(forest_rmse_scores)
        fileCounter = self.getFileCounter("forest_reg_%s.pkl")
        joblib.dump(forest_reg, self.path_model + "forest_reg_%s.pkl" % fileCounter)

    def crossValidation(self, ols, x_train, y_train):
        # Je höher der Wert von MSE, desto schlechter ist das Modell.
        print("Kreuzvalidierung der linearen Regression mittels (negativem) MSE: " +
              str(cross_val_score(ols, x_train, y_train, scoring='neg_mean_squared_error')))
        # Je näher R2 an 1 liegt,desto besser ist das Modell.
        print("Kreuzvalidierung der linearen Regression mittels R-Quadrat: " +
              str(cross_val_score(ols, x_train, y_train, scoring='r2')))

    def display_scores(self, scores):
        print("Scores: ", scores)
        print("Mean: ", scores.mean())
        print("Standart deviation: ", scores.std())

    def getFileCounter(self, fileBaseName):
        fileCounter = 1
        while os.path.exists(self.path_model + fileBaseName % fileCounter):
            fileCounter += 1
        return fileCounter
