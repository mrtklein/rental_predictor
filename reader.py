import os
import pandas as pd
import glob
import pickle


class Reader:
    def __init__(self):
        # print(self.printdir())
        full_path = os.path.realpath(__file__)
        path, filename = os.path.split(full_path)
        self.data_path = path + '\\datasets\\'
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        filepathes = glob.glob(self.data_path + r"*.data")
        if len(filepathes) > 0:
            pass
        else:
            self.csv_reader(path + '\\sources\\*.csv')

    def dataframe_loader(self, filename=None):
        df_all = []
        for file in glob.glob(self.data_path + "*.data"):
            name = self.path_splitter(file, name=True)
            df = pickle.load(open(file, "rb"))
            df.name = name
            df_all.append(df)
            if filename is not None and filename == name:
                return df
        return df_all

    def csv_reader(self, path):
        test = glob.glob(path)
        if len(glob.glob(path)) > 0:
            for filepath in glob.glob(path):
                df = pd.read_csv(filepath)
                name = self.path_splitter(filepath, name=True)
                self.safe_dataframe(df, name)
        else:
            raise Exception("Please add at least one csv file within the folder sources!")

    def safe_dataframe(self, df, filename):
        pickle.dump(df, open(self.data_path + filename + ".data", "wb"))

    def path_splitter(self, path, name=True, filename=False, diretory=False, filetype=False):
        if filename:
            return os.path.basename(path)
        elif diretory:
            return os.path.dirname(path)
        elif name:
            n, ftype = os.path.splitext(os.path.basename(path))
            return n
        elif filetype:
            n, ftype = os.path.splitext(os.path.basename(path))
            return ftype

    def printdir(self):
        print("Path at terminal when executing this file")
        print(os.getcwd() + "\n")
        print("This file path, relative to os.getcwd()")
        print(__file__ + "\n")
        print("This file full path (following symlinks)")
        full_path = os.path.realpath(__file__)
        print(full_path + "\n")
        print("This file directory and name")
        path, filename = os.path.split(full_path)
        print(path + ' --> ' + filename + "\n")
        print("This file directory only")
        print(os.path.dirname(full_path))
