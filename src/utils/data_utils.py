import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Dataset:
    """
    Class housing functions used for working with datasets.
    """

    def __init__(self, dataset_name: str):

        dataset = "iris.data" if dataset_name == "Iris dataset" else "breast_cancer.data"
        self._data = pd.read_csv(f"data/{dataset}", header = None, na_values='?')
        self._dataset_name = dataset_name
        self._label_encoder = LabelEncoder()

        self._preprocess_data()
        self._split_data()

    def _preprocess_data(self):

        # rename the columns
        column_names = [f"feature_{i}" for i in range(self._data.shape[1] - 1)]
        column_names.append("label")
        self._data.columns = column_names

        # remove unecessary columns
        if self._dataset_name == "Breast Cancer dataset":
            self._data.drop("feature_0", axis = 1, inplace = True)

        # remove unwanted classes
        if self._dataset_name == "Iris dataset":
            self._data = self._data[self._data["label"] != "Iris-setosa"]

        # clean N/A rows
        self._data = self._data.dropna(axis = 0)

        # encoding the labels
        self._data["label"] = self._label_encoder.fit_transform(self._data["label"])

        # insert constant 1 column to substitute bias
        self._data.insert(loc=0, column='feature_bias', value=1)

        # splitting into data and labels
        self._y = self._data["label"]
        self._x = self._data.drop("label", axis = 1)

        # convert to numpy array
        self._y = self._y.to_numpy()
        self._x = self._x.to_numpy()

    def _split_data(self):
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self._x, self._y, test_size=0.2, random_state=42
        )

    @property
    def shape(self):
        return self._x.shape