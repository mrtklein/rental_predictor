from sklearn.model_selection import train_test_split
from reader import Reader
from data_script.data_preparation import Data_Preparation
from pandas.plotting import scatter_matrix


class Data:
    def __init__(self):
        reader = Reader()
        df_all = reader.dataframe_loader()
        preparer = Data_Preparation()
        self.data_prepared = preparer.get_preparedData(df_all[0])
        self.data_prepared_woutNanEdit = preparer.get_preparedData_withoutNanTreatment(df_all[0])

    def get_data_full(self):
        output_data = self.data_prepared['baseRent']
        input_data = self.data_prepared.drop(['baseRent'], axis=1)

        return input_data, output_data

    def get_data(self):
        output_data = self.data_prepared['baseRent']
        input_data = self.data_prepared.drop(['baseRent'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test

    def get_data_woutNanEdit(self):
        output_data = self.data_prepared_woutNanEdit['baseRent']
        input_data = self.data_prepared_woutNanEdit.drop(['baseRent'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test
