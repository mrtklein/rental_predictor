import numpy as np
import pandas as pd
from sklearn import preprocessing
# for KKNimputer scikitt-learn must be at least 0.22 (pip install --upgrade scikit-learn)
from sklearn.impute import KNNImputer
from visualisation.visualisation import Visualisation as vis
from pandas import ExcelWriter

class Data_Preparation:
    def __init__(self):
        # class memeber for minmax Scale.
        self.minmax_scale = None

    def get_preparedData(self, df_unprepared):
        # print(df_unprepared.info())
        # print("\nCorrelation matrix unprepared: ")
        # corr_matrix = df_unprepared.corr()
        # corr_matrix_sort=corr_matrix["baseRent"].sort_values(ascending=False)
        # writer = ExcelWriter('CorrelationMatrix_unpreparedData.xlsx')
        # corr_matrix_sort.to_excel(writer, 'Sheet5')
        # writer.save()
        data_selected = self.selectData(df_unprepared)
        data_prepared = self.dropTables(data_selected)
        data_normalized = self.normalizeColumns(data_prepared)
        data_imputed = self.imputeData(data_normalized)
        # self.printdatadetails(data_imputed)
        # print("\nCorrelation matrix prepared: ")
        # corr_matrix = data_imputed.corr()
        # corr_matrix_sort=corr_matrix["baseRent"].sort_values(ascending=False)
        return data_imputed

    def get_preparedData_withoutNanTreatment(self, df_unprepared):
        # self.printdatadetails(df_unprepared)
        data_selected = self.selectData(df_unprepared)
        data_prepared = self.dropTables(data_selected)
        data_normalized = self.normalizeColumns(data_prepared)
        data_woutNan = data_normalized.dropna(axis=1, how='any')
        return data_woutNan

    def selectData(self, df):
        df_cologne = df[df.regio2 == "Köln"]
        # filter out trash
        df_baseRent = df_cologne[(df_cologne.baseRent > 100) & (df_cologne.baseRent < 50000)]

        # remove a weired datapoint manually
        df_baseRent = df_baseRent[df_baseRent.geo_plz != 76530]

        return df_baseRent

    def dropTables(self, df):
        # drop useless columns:
        # dropping text columns (e.g. description)
        # dropping other price columns because we will focus on the basePrice
        # dropping als xxxRange columns because we will normalize it for our subset by our self
        # dropping advertisment (telekom) columns
        # dropping useles columns (e.g. date, scoutId, regio, exact address)
        # dropping sparse columns (e.g. heating type, heatingCosts)
        return df.drop(
            ["regio1", "regio2", "telekomTvOffer", "telekomHybridUploadSpeed", "houseNumber", "street", "streetPlain",
             "description", "electricityBasePrice", "electricityKwhPrice", "date", "scoutId", "geo_bln",
             "geo_krs", "pricetrend", "telekomUploadSpeed", "baseRentRange", "livingSpaceRange", "facilities",
             'serviceCharge',
             'heatingType', 'totalRent', 'firingTypes', 'yearConstructedRange', 'thermalChar', 'noRoomsRange',
             'facilities', "regio3",
             'heatingCosts', 'energyEfficiencyClass', 'lastRefurbish'], axis=1)

    def normalizeColumns(self, df):
        # convert (yes/no) columns in (0/1) columns
        df.newlyConst = df.newlyConst.astype(float)
        df.balcony = df.balcony.astype(float)
        df.hasKitchen = df.hasKitchen.astype(float)
        df.cellar = df.cellar.astype(float)
        df.lift = df.lift.astype(float)
        df.garden = df.lift.astype(float)

        # convert categorical into ordinal
        petsAllowedDict = {"no": 0.0, "negotiable": 0.5, "yes": 1.0}
        df.petsAllowed = df.petsAllowed.replace(petsAllowedDict)

        conditionDict = {"well_kept": 0.3,
                         "mint_condition": 1,
                         "fully_renovated": 0.6,
                         "first_time_use_after_refurbishment": 1,
                         "modernized": 0.6,
                         "refurbished": 0.6,
                         "first_time_use": 1,
                         "negotiable": 0,
                         "need_of_renovation": 0}
        df.condition = df.condition.replace(conditionDict)

        interiorQualDict = {"sophisticated": 0.6, "normal": 0.3, "luxury": 1, "simple": 0}
        df.interiorQual = df.interiorQual.replace(interiorQualDict)

        # dealing with Nan in ParkSpaces
        # we can assume that most landlords just set Parking spaces to NaN when they have none,
        # as opposed to setting it manually to 0. Therefor we manually set all NaNs in this column
        df.noParkSpaces = df.noParkSpaces.replace({np.nan: 0})

        #filling Nans in TypeOfFlat
        df.typeOfFlat = df.typeOfFlat.replace({np.nan: "apartment"})

        # one hot encoding
        df = pd.get_dummies(df, prefix=["geo_plz", "typeOfFlat"], columns=["geo_plz", "typeOfFlat"], dummy_na=False)
        #df = df.drop(["geo_plz_nan"], axis=1)


        # normalize float values
        self.minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
        df[['picturecount', 'yearConstructed', 'livingSpace', 'noRooms', 'floor', 'numberOfFloors']] \
            = self.minmax_scale.fit_transform(
            df[['picturecount', 'yearConstructed', 'livingSpace', 'noRooms', 'floor', 'numberOfFloors']])
        # , "baseRent"
        return df

    def imputeData(self, df):
        columns = df.columns
        imputer = KNNImputer(missing_values=np.nan)
        df = imputer.fit_transform(df)
        df = pd.DataFrame(df, columns=columns)

        return df

    def printdatadetails(self, df):
        print("\n\n")
        print("Details:*******************************************")
        columns = df.columns.values
        for col in columns:
            column_dtype = df[col].dtype
            print("\n")
            print("Column_name: " + col)
            print("---DataType: " + str(column_dtype))
            print("---Number of rows:" + str(len(df[col].index)))
            print("---Number of nan: " + str(df[col].isna().sum()))
            if np.issubdtype(column_dtype, np.number):
                print('---Maximum:', df[col].max())
                print('---Minimum:', df[col].min())
                print('---Mean:', df[col].mean())
                print('---Sum:', df[col].sum())
                print('---Count:', df[col].count())
            elif column_dtype == 'bool':
                pass
            else:
                col_categories = df[col].unique()
                print("Categories: ")
                for item in col_categories:
                    if not pd.isna(item):
                        print(item + ",", sep=' ', end=" ")
