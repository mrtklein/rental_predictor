import numpy as np
import pandas as pd


class Data_Preparation:
    def get_preparedData(self, df_unprepared):
        print(df_unprepared.info())
        data_selected = self.selectData(df_unprepared)
        data_prepared = self.dropTables(data_selected)
        self.printdatadetails(data_prepared)
        return data_prepared

    def selectData(self, df):
        df_cologne = df[df.regio2 == "Köln"]
        # filter out trash
        # totalRent nicht sinnvoll da oft NaS
        df_baseRent = df_cologne[(df_cologne.baseRent > 100) & (df_cologne.baseRent < 50000)]

        # thermalChar removen? Sehr sparse, und gibt nicht so richtig sinn
        # geokoordinaten: sind oft falsch, bzw sparse. Ausserdem schwer zu verarbeiten
        # pricetrend removen? -> da steckt schon preis drinn...
        # total preis sollte sinnvoll berechnet sein. z.b. total preis wenn nAn, ansonsten base preis + flex, wenn nur basepreis vllt +10% oder so

        # print uniqe values and their count
        # self.data_prepared.interiorQual.value_counts(dropna=False)
        return df_baseRent

    def dropTables(self, df):
        # drop useless columns
        return df.drop(
            ["regio1", "regio2", "telekomTvOffer", "telekomHybridUploadSpeed", "houseNumber", "street", "streetPlain",
             "description", "electricityBasePrice", "electricityKwhPrice", "date", "scoutId", "geo_bln",
             "geo_krs", "pricetrend", "telekomUploadSpeed", "baseRentRange", "livingSpaceRange", "facilities"], axis=1)

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
