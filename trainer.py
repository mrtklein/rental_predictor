import os
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from data_script.data import Data
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from data_script.data_preparation import Data_Preparation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.svm import SVR


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

        results = {}
        results['LinearRegression'] = self.linear_regressor()
        results['PolynomialRegression'] = self.polynomial_regressor()
        results['LassoRegression'] = self.lasso_Regressor()
        results['DecisionTreeRegression'] = self.decisionTreeRegressor()
        results['RandomForestRegression'] = self.randomForestRegressor()
        results['ElasticNetRegression'] = self.ElasticNetRegressor()
        results['LinearSVRRegression'] = self.LinearSVRRegressor()
        results['RBF_SVRRegressor'] = self.RBF_SVRRegressor()
        print("\nCompare RMSE mean score of Training models: ")
        df_results = pd.DataFrame.from_dict(results, orient='index', columns=["RMSE"])
        print(df_results)

        # Load a Model
        # model=joblib.load("mymodel.pkl")

    def linear_regressor(self):
        print(
            "\n***********************************************************************Linear Regression")
        # Ein Objekt für lineare Regression erzeugen
        lin_reg = LinearRegression()

        # # training the algorithm
        lin_reg.fit(self.x_train, self.y_train)
        # print("linearRegression_intercept: " + str(ols.intercept_))
        # For retrieving the slope:
        # print("linearRegression_coef: " + str(ols.coef_))

        # Kreuzvalidierung
        lin_scores = cross_val_score(lin_reg, self.inputData_prepared, self.outputData,
                                     scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        print("CrossValidation cv=10: ")
        self.display_scores(lin_rmse_scores)

        y_predict = lin_reg.predict(self.x_test)
        lin_mse = mean_squared_error(self.y_test, y_predict)
        lin_rmse = np.sqrt(lin_mse)
        print("RMSE of linear regression on validation data x_test: " + str(lin_rmse))

        # Manuelles löschen zweier unerklärlichen Minus predictions
        y_test_corrected = np.delete(self.y_test.values, 201)
        y_test_corrected = np.delete(y_test_corrected, 163)
        y_predict_corrected = np.delete(y_predict, 201)
        y_predict_corrected = np.delete(y_predict_corrected, 163)
        lin_mse = mean_squared_error(y_test_corrected, y_predict_corrected)
        lin_rmse = np.sqrt(lin_mse)
        print("\nRMSE of corrected linear regression: " + str(lin_rmse))

        # Speichern des Modells mit aufsteigender Numerierung (vermeiden des Überschreiben des Modells)
        fileCounter = self.__getFileCounter__("lin_reg_%s.pkl")
        joblib.dump(lin_reg, self.path_model + "lin_reg_%s.pkl" % fileCounter)

        # format result für eine bessere Lesbarkeit
        result = '{0:.10f}'.format(lin_rmse)
        return result

    def polynomial_regressor(self):
        print(
            "\n***********************************************************************Polynomial Regression")
        poly_features = PolynomialFeatures(degree=2, include_bias=True)
        x_train_poly = poly_features.fit_transform(self.x_train)
        x_test_poly = poly_features.fit_transform(self.x_test)

        # Ein Objekt für lineare Regression erzeugen
        poly_reg = LinearRegression()

        # # training the algorithm
        poly_reg.fit(x_train_poly, self.y_train)

        # Kreuzvalidierung
        poly_scores = cross_val_score(poly_reg, self.inputData_prepared, self.outputData,
                                      scoring="neg_mean_squared_error", cv=10)
        poly_rmse_scores = np.sqrt(-poly_scores)
        print("CrossValidation cv=10: ")
        self.display_scores(poly_rmse_scores)

        y_predict = poly_reg.predict(x_test_poly)
        poly_mse = mean_squared_error(self.y_test, y_predict)
        poly_rmse = np.sqrt(poly_mse)
        print("RMSE of polynomial regression on validation data x_test: " + str(poly_rmse))

        # Manuelles löschen zweier unerklärlichen Minus predictions
        y_test_corrected = np.delete(self.y_test.values, 201)
        y_test_corrected = np.delete(y_test_corrected, 163)
        y_predict_corrected = np.delete(y_predict, 201)
        y_predict_corrected = np.delete(y_predict_corrected, 163)
        poly_mse = mean_squared_error(y_test_corrected, y_predict_corrected)
        poly_rmse = np.sqrt(poly_mse)
        print("\nRMSE of corrected polynomial regression on validation data x_test: " + str(poly_rmse))

        # Speichern des Modells mit aufsteigender Numerierung (vermeiden des Überschreiben des Modells)
        fileCounter = self.__getFileCounter__("poly_reg_%s.pkl")
        joblib.dump(poly_reg, self.path_model + "poly_reg_%s.pkl" % fileCounter)
        # format result für eine bessere Lesbarkeit
        result = '{0:.10f}'.format(poly_rmse)
        return result

    def lasso_Regressor(self):
        print(
            "\n***********************************************************************Lasso Regression")
        bestScore = -99999999
        # we look for a good hyperparameter, so we go through possible values with a for loop
        alphas = np.arange(0.01, 1.5, 0.01)

        # Ein Objekt für Lasso Regression erzeugen
        lasso = Lasso()
        # Dictionary mit Hyperparameterkandidaten erzeugen
        hyperparameters = dict(alpha=alphas)

        # Gittersuche erstellen
        gridsearch = GridSearchCV(lasso, hyperparameters, cv=10, verbose=0, scoring="neg_mean_squared_error")
        # Gittersuche anpassen
        best_model = gridsearch.fit(self.inputData_prepared, self.outputData)

        print('Bester Strafterm:', best_model.best_estimator_.get_params()['alpha'])

        print("CrossValidation cv=10: ")
        # Evaluate Model
        # Verschachtelte Kreuzvalidierung ausführen und durchschnittlichen Score ausgeben
        # ******************************************************************************************Erklärung Kochbuch S.215-216
        scores = cross_val_score(gridsearch, self.inputData_prepared, self.outputData, scoring="neg_mean_squared_error",
                                 cv=10)
        lasso_rmse_scores = np.sqrt(-scores)
        self.display_scores(lasso_rmse_scores)
        fileCounter = self.__getFileCounter__("lasso_reg_%s.pkl")
        joblib.dump(best_model, self.path_model + "lasso_reg_%s.pkl" % fileCounter)
        # format result für eine bessere Lesbarkeit
        result = '{0:.10f}'.format(lasso_rmse_scores.mean())
        return result

    def decisionTreeRegressor(self):
        print(
            "\n***********************************************************************DecisionTree Regression")
        # we look for a good hyperparameter, so we go through possible values with a for loop
        min_samples_leafs = np.arange(1, 20,
                                      1)  # ideal min_samples_leaf values tend to be between 1 to 20 for the CART algorithm (https://arxiv.org/abs/1812.02207)
        # Ein Objekt für DecisionTree Regression erzeugen
        tree_reg = DecisionTreeRegressor()

        # Dictionary mit Hyperparameterkandidaten erzeugen
        hyperparameters = dict(min_samples_leaf=min_samples_leafs)
        # Gittersuche erstellen
        gridsearch = GridSearchCV(tree_reg, hyperparameters, cv=10, verbose=0, scoring="neg_mean_squared_error")
        # Gittersuche anpassen
        best_model = gridsearch.fit(self.inputData_prepared, self.outputData)
        print('Die beste Mindestanzahl der Proben, die an einem leafe node erforderlich sind:',
              best_model.best_estimator_.get_params()['min_samples_leaf'])

        print('Tiefe des Entscheidungsbaums: ', best_model.best_estimator_.get_depth())
        print('Anzahl an leafe nodes: ', best_model.best_estimator_.get_n_leaves())

        # Evaluate Model
        # Verschachtelte Kreuzvalidierung ausführen und durchschnittlichen Score ausgeben
        print("CrossValidation cv=10: ")
        scores = cross_val_score(gridsearch, self.inputData_prepared, self.outputData, scoring="neg_mean_squared_error",
                                 cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        self.display_scores(tree_rmse_scores)
        fileCounter = self.__getFileCounter__("tree_reg_%s.pkl")
        joblib.dump(tree_reg, self.path_model + "tree_reg_%s.pkl" % fileCounter)
        # format result für eine bessere Lesbarkeit
        result = '{0:.10f}'.format(tree_rmse_scores.mean())
        return result

    def randomForestRegressor(self):
        print(
            "\n***********************************************************************RandomForest Regression")
        # Create the parameter grid based on the results of random search
        param_grid = {
            'min_samples_leaf': np.arange(1, 20, 2),
            'n_estimators': [100, 200, 300]
        }

        forest_reg = RandomForestRegressor()
        # Gittersuche erstellen
        gridsearch = GridSearchCV(forest_reg, param_grid, cv=10, verbose=0, scoring="neg_mean_squared_error")
        # Gittersuche anpassen
        best_model = gridsearch.fit(self.inputData_prepared, self.outputData)

        print('Die beste Mindestanzahl der Proben, die an einem leafe node erforderlich sind:',
              best_model.best_estimator_.get_params()['min_samples_leaf'])
        print("Die beste Anzahl an Entscheidungsbäumen: ", best_model.best_estimator_.get_params()['n_estimators'])

        # Evaluieren
        print("CrossValidation cv=10: ")
        scores_absolute = cross_val_score(forest_reg, self.inputData_prepared, self.outputData,
                                 scoring="neg_mean_absolute_error",
                                 cv=10)


        scores = cross_val_score(forest_reg, self.inputData_prepared, self.outputData, scoring="neg_mean_squared_error",
                                 cv=10)
        forest_rmse_scores = np.sqrt(-scores)
        self.display_scores(forest_rmse_scores)
        fileCounter = self.__getFileCounter__("forest_reg_%s.pkl")
        joblib.dump(forest_reg, self.path_model + "forest_reg_%s.pkl" % fileCounter)

        # Plot Merkmale
        # Merkmalswichtigkeiten berechnen
        importances = best_model.best_estimator_.feature_importances_
        # Merkmalswichtigkeiten in fallender Reihenfolge sortieren
        indices = np.argsort(importances)[::-1]
        # Merkmalsnamen umordnen, sodass sie der sortierten Merkmalswichtigkeit entsprechen
        names = [self.inputData_prepared.columns[i] for i in indices]

        fig = plt.figure()
        plt.title("Wichtigkeit der Merkmale")
        # Säulen hinzufügen
        plt.bar(range(self.inputData_prepared.shape[1]), importances[indices])
        # Merkmalsnamen als Bezeichnungen der x-Achse hinzufügen
        plt.xticks(range(self.inputData_prepared.shape[1]), names, rotation=90)
        plt.show()
        fig.savefig('randomForestMerkmale.jpg')

        # format result für eine bessere Lesbarkeit
        result = '{0:.10f}'.format(forest_rmse_scores.mean())
        return result

    def ElasticNetRegressor(self):
        print(
            "\n***********************************************************************Elastic Net ")
        # we look for a good hyperparameter, so we go throgh possible values with a for loop
        alphas = np.arange(0.1, 1.5, 0.1)
        l1_ratio = np.arange(0.0, 1.1, 0.1)
        # Dictionary mit Hyperparameterkandidaten erzeugen
        hyperparameters = dict(alpha=alphas, l1_ratio=l1_ratio)

        elastic = ElasticNet()
        # Gittersuche erstellen
        gridsearch = GridSearchCV(elastic, hyperparameters, cv=10, verbose=0, scoring="neg_mean_squared_error")
        # Gittersuche anpassen
        best_model = gridsearch.fit(self.inputData_prepared, self.outputData)

        print('BestAplpha:', best_model.best_estimator_.get_params()['alpha'])
        print('Best L1 Ratio:', best_model.best_estimator_.get_params()['l1_ratio'])
        print('Best score :', np.sqrt(-best_model.best_score_))
        return np.sqrt(-best_model.best_score_)

    def LinearSVRRegressor(self):
        print(
            "\n***********************************************************************Linear SVR ")
        # we look for a good hyperparameter, so we go throgh possible values with a for loop
        C = np.arange(100, 700, 10)
        # Dictionary mit Hyperparameterkandidaten erzeugen
        hyperparameters = dict(C=C)

        svr = LinearSVR()
        # Gittersuche erstellen
        gridsearch = GridSearchCV(svr, hyperparameters, cv=10, verbose=0, scoring="neg_mean_squared_error")
        # Gittersuche anpassen
        best_model = gridsearch.fit(self.inputData_prepared, self.outputData)

        print('Best C:', best_model.best_estimator_.get_params()['C'])
        print('Best score :', np.sqrt(-best_model.best_score_))
        return np.sqrt(-best_model.best_score_)

    def RBF_SVRRegressor(self):
        print(
            "\n***********************************************************************RBF SVR ")
        # we look for a good hyperparameter, so we go throgh possible values with a for loop
        C = np.arange(200, 400, 10)
        gamma = np.arange(0.01, 1.5, 0.2)
        # Dictionary mit Hyperparameterkandidaten erzeugen
        hyperparameters = dict(C=C, gamma=gamma)

        svr = SVR()
        # Gittersuche erstellen
        gridsearch = GridSearchCV(svr, hyperparameters, cv=10, verbose=1, scoring="neg_mean_squared_error")
        # Gittersuche anpassen
        best_model = gridsearch.fit(self.inputData_prepared, self.outputData)

        print('Best C:', best_model.best_estimator_.get_params()['C'])
        print('Best gamma:', best_model.best_estimator_.get_params()['gamma'])
        print('Best score :', np.sqrt(-best_model.best_score_))
        return np.sqrt(-best_model.best_score_)

    def display_scores(self, scores):
        print("Scores: ", scores)
        print("Mean: ", scores.mean())
        print("Standart deviation: ", scores.std())

    def __getFileCounter__(self, fileBaseName):
        fileCounter = 1
        while os.path.exists(self.path_model + fileBaseName % fileCounter):
            fileCounter += 1
        return fileCounter

    def __plot_results__(self, y_predict):
        df = pd.DataFrame({'Actual': self.y_test.values.flatten(), 'Predicted': y_predict.flatten()})
        print("\nResults of prediciting inputData_prepared: \n" + str(df))
        df1 = df.head(25)
        df1.plot(kind='bar', figsize=(16, 10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.legend()
        plt.show()

    def __plot_learning_curves__(self, model, title):
        train_errors, val_errors = [], []
        for m in range(1, len(self.x_train)):
            model.fit(self.x_train[:m], self.y_train[:m])
            y_train_predict = model.predict(self.x_train[:m])
            y_val_predict = model.predict(self.x_test)
            train_errors.append(mean_squared_error(self.y_train[:m], y_train_predict))
            val_errors.append(mean_squared_error(self.y_test, y_val_predict))
        plt.title(title)
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=1, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
        plt.ylabel('RMSE', fontsize=12)
        plt.xlabel('Training set size', fontsize=12)
        # fig.savefig('test.jpg')
        plt.legend()
        plt.show()
