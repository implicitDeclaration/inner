import numpy as np
import sys
from torch.utils.data import Dataset
import pandas as pd

sys.path.append("../")


def census_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("../datasets/census", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 13)
    nb_classes = 2

    return X, Y, input_shape, nb_classes


def census_pd(file_path="./datasets/census", shuffle=False):
    whole_data = pd.read_csv(file_path)
    if shuffle:
        whole_data.sample(frac=1)
    x_data = whole_data.copy()[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']]
    y_data = whole_data.copy()[['n']]
    q = []
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    for c in columns:
        q.append(len(x_data[c].unique()))
    #print(f'unique data len is : {q}')
    return whole_data, x_data, y_data


def get_batch_data(start, end, label, attr='a', file_dir='./datasets/bank'):
    return None
    # data = pd.read_csv(file_dir)
    # data_length = len(data)
    # data1 = data[data['n'] == 1]
    # data0 = data[data['q'] == 0]
    #
    # x1_data = data1[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']]
    # x0_data = data0[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']]
    #
    # if label == 0:
    #     return np.array(x0_data.iloc[start: end])
    # else:
    #     return np.array(x1_data.iloc[start: end])


def get_condi_distr(col):
    '''
    :param col:
    :return: the attribute rank and corresponding value
    '''
    all_cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    if col not in all_cols:
        raise NotImplementedError

    file_dir = './datasets/census'
    data = pd.read_csv(file_dir)
    over_income = data.loc[data['n'] == 1]
    value_all = data[col].unique()
    value_cnt = over_income[col].value_counts(ascending=False)
    attrs = value_cnt.index
    for a in value_all:  # add missing elements
        if a not in attrs:
            value_cnt[a] = 0
    attrs = value_cnt.index
    return attrs, value_cnt


class CensusLabelSelect(Dataset):  # 'h' is race, 'i' is gender
    def __init__(self, file_dir, attr='i', limit=0):
        self.data = pd.read_csv(file_dir)
        self.raw_x = self.data.copy()[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']]

        self.male_data = self.data.loc[self.data['i'] == 0]
        self.female_data = self.data.loc[self.data['i'] == 1]

        if limit == 0:
            self.x_data = self.male_data.copy()[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']]
            self.y_data = self.male_data.copy()['n']
        elif limit == 1:
            self.x_data = self.female_data.copy()[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']]
            self.y_data = self.female_data.copy()['n']
        else:
            self.y_data = self.data[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']]
            self.x_data = self.data['n']
        self.data_length = len(self.x_data)
        # self.label_mean_var = self.get_dist(self.y_data, self.y_data.columns, attr)
        # self.x_mean_var = self.get_dist(self.raw_x, self.raw_x.columns, attr=None)
        # for c in range(len(self.x_data.columns)):  # normalization
        #     self.x_data[self.x_data.columns[c]] = (self.x_data[self.x_data.columns[c]] - self.x_mean_var[c][0]) / self.x_mean_var[c][
        #         1]

    def __getitem__(self, idx):
        x = self.x_data.iloc[idx]
        y = self.y_data.iloc[idx]
        return np.array(x), np.array(y)

    def __len__(self):
        return self.data_length

    def get_dist(self, x, colunms, attr):
        mean_var = []
        for l in colunms:
            if l == attr:
                dist = [0, 1]
                mean_var.append(dist)
            else:
                col = x[l]
                dist = [col.mean(), col.var()]
                mean_var.append(dist)
        return mean_var

    def get_output_clamp(self):
        label_max = []
        label_min = []
        for l in self.x_data.columns:
            col = self.x_data[l]
            label_max.append(col.max())
            label_min.append(col.min())
        return label_max, label_min

    def column_min_max(self, col='a'):
        label_max = self.data[col].max()
        label_min = self.data[col].min()
        return label_max, label_min


class CensusDatasetWhole(Dataset):  # 'h' is race, 'i' is gender
    def __init__(self, file_dir, attr=None, dummy=False, resample=False):
        self.data = pd.read_csv(file_dir)
        if resample:
            self.data = self.data_resample()
        self.data_length = len(self.data)
        if attr is not None:
            interested_col = self.data[attr]
            dum_col = pd.get_dummies(interested_col)
            print('interested attribute is {}, dummy result is {}'.format(attr, dum_col.columns))
            self.y_data = self.data.copy()
            del self.y_data[attr]
            self.y_data = pd.concat([dum_col, self.y_data], axis=1)
        else:
            self.y_data = self.data.copy()
        print('label columns is {}'.format(self.y_data.columns))
        self.x_data = self.data.copy()
        if dummy:
            dum_col = self.data['n']
            dum_col = pd.get_dummies(dum_col)
            del self.x_data['n']
            self.x_data = pd.concat([dum_col, self.x_data], axis=1)
        self.label_mean_var = self.get_dist(self.y_data, self.y_data.columns)
        self.x_mean_var = self.get_dist(self.x_data, self.x_data.columns)
        for c in range(len(self.x_data.columns)):  # normalization
            self.x_data[self.x_data.columns[c]] = (self.x_data[self.x_data.columns[c]] - self.x_mean_var[c][0]) / self.x_mean_var[c][
                1]

    def __getitem__(self, idx):
        x = self.x_data.iloc[idx]
        y = self.y_data.iloc[idx]
        return np.array(x), np.array(y)

    def __len__(self):
        return self.data_length

    def get_dist(self, x, colunms):
        mean_var = []
        for l in colunms:
            col = x[l]
            dist = [col.mean(), col.var()]
            mean_var.append(dist)
        return mean_var

    def get_output_clamp(self):
        label_max = []
        label_min = []
        for l in self.x_data.columns:
            col = self.x_data[l]
            label_max.append(col.max())
            label_min.append(col.min())
        return label_max, label_min

    def column_min_max(self, col='a'):
        label_max = self.data[col].max()
        label_min = self.data[col].min()
        return label_max, label_min


class CensusDataset(Dataset):
    def __init__(self, file_dir='./datasets/census'):
        self.data = pd.read_csv(file_dir)
        self.data_length = len(self.data)
        self.x_data = self.data[['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']]
        self.y_data = self.data['n']
        self.x_mean_var = self.get_dist(self.x_data, self.x_data.columns)
        # for c in range(len(self.x_data.columns)):  # normalization
        #     self.x_data[self.x_data.columns[c]] = (self.x_data[self.x_data.columns[c]] - self.x_mean_var[c][0]) / self.x_mean_var[c][
        #         1]

    def get_dist(self, x, colunms):
        mean_var = []
        for l in colunms:
            col = x[l]
            dist = [col.mean(), col.var()]
            mean_var.append(dist)
        return mean_var

    def data_resample(self):
        cla1 = self.data[self.data['q'] == 1]
        cla0 = self.data[self.data['q'] == 0]
        rep = len(cla0)//len(cla1) - 1
        for i in range(rep):
            self.data = pd.concat([self.data, cla1] ,ignore_index=True)
        return self.data

    def __getitem__(self, idx):
        x = self.x_data.iloc[idx]
        y = self.y_data[idx]
        return np.array(x), np.array(y)

    def __len__(self):
        return self.data_length
#
# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
#
#
