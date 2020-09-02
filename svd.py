import numpy as np
import pandas as pd
from utils import special_sort, dissimilarity
from math import sqrt


def create_utility_matrix(data, formatizer={"user": 0, "item": 1, "value": 2}):
    """

    :param data: Array-like, 2, nx3
    :param formatizer: indices, pass the formatizer
    :return: the utility matrix. 2D, n x m, n = number of users, m = number of items
    """

    formatizer = {"user": 0, "item": 1, "value": 2}
    itemField = formatizer["item"]
    userField = formatizer["user"]
    valueField = formatizer["value"]

    userList = data.iloc[:, userField].tolist()
    itemList = data.iloc[:, itemField].tolist()
    valueList = data.iloc[:, valueField].tolist()

    users = list(set(data.iloc[:, userField]))
    items = list(set(data.iloc[:, itemField]))

    users_index = {users[i]: i for i in range(len(users))}

    pd_dict = {item: [np.nan for i in range(len(users))] for item in items}

    for i in range(0, len(data)):
        item = itemList[i]
        user = userList[i]
        value = valueList[i]

        pd_dict[item][users_index[user]] = value

    X = pd.DataFrame(pd_dict)
    X.index = users
    users = list(X.index)
    items = list(X.columns)

    return np.array(X), users, items


class SVDRecommender:
    """
    Singular Value Decomposition is an important technique used in recommendation systems.
    Using SVD, the complete utility matrix is decomposed into user and item features.
    Thus the dimensionality of the matrix is reduced and we get the most important features neglecting the weaker ones.

    formatizer: a dictionary having the keys 'user', 'item' and 'value' each having an integer value, that denotes
                the column numbers of the corresponding things in the array provided in the fit and predict method.
                The 'value' will be used only in the fot method.
    """

    def __init__(self, number_of_features=15, method='default'):
        self.parameters = {"number_of_features", "method"}
        self.method = method
        self.number_of_features = number_of_features

    # Getter method.
    def get_params(self, deep=False):
        out = dict()
        for param in self.parameters:
            out[param] = getattr(self, param)

        return out

    # Setter method.
    def set_params(self, **params):
        for a in params:
            if a in self.parameters:
                setattr(self, a, params[a])
            else:
                raise AttributeError("No such attribute exist to be set")

    def fit(self, user_item_matrix, userList, itemList):

        """
        :return: Does not return anything. Just fits the data and forms U, s, V by SVDRecommender.
        """

        self.users = list(userList)
        self.items = list(itemList)

        self.user_index = {self.users[i]: i for i in range(len(self.users))}
        self.item_index = {self.items[i]: i for i in range(len(self.items))}

        mask = np.isnan(user_item_matrix)
        masked_arr = np.ma.masked_array(user_item_matrix, mask)

        self.predMask = ~mask

        self.item_means = np.mean(masked_arr, axis=0)
        self.user_means = np.mean(masked_arr, axis=1)
        self.item_means_tiled = np.tile(self.item_means, (user_item_matrix.shape[0], 1))

        # Utility matrix or ratings matrix that can be fed to svd
        self.utilMat = masked_arr.filled(self.item_means)

        # For the default method.
        if self.method == 'default':
            self.utilMat = self.utilMat - self.item_means_tiled

        # Singular Value Decomposition starts.
        # k denotes the number of features of each user and item.
        # The top matrices are cropped to take the greatest k rows or columns. U, V, s are already sorted descending.
        k = self.number_of_features
        U, s, V = np.linalg.svd(self.utilMat, full_matrices=False)

        U = U[:, 0:k]
        V = V[0:k, :]
        s_root = np.diag([sqrt(s[i]) for i in range(0, k)])

        self.Usk = np.dot(U, s_root)
        self.skV = np.dot(s_root, V)
        self.UsV = np.dot(self.Usk, self.skV)

        self.UsV = self.UsV + self.item_means_tiled

    def predict(self, X, formatizer={"user": 0, "item": 1}):
        """

        :param X: The test set. 2D, array-like consisting of two elements in each row corresponding to the userId and itemId.
        :return: 1
        :return: 1, a list giving the value/rating corresponding to each use-item pair in each row of X.
        """

        users = X.iloc[:, 0].tolist()
        items = X.iloc[:, 1].tolist()

        if self.method == 'default':

            values = []
            for i in range(len(users)):
                user = users[i]
                item = items[i]

                # User and item in the test set may not always occur in the train set.
                # In these cases we can not find those values from the utility matrix.
                # That is why a check is necessary.
                # 1. both user and item in train
                # 2. only user in train
                # 3. only item in train
                # 4. none in train

                if user in self.user_index:
                    if item in self.item_index:
                        values.append(self.UsV[self.user_index[user], self.item_index[item]])
                    else:
                        values.append(self.user_means[self.user_index[user]])
                elif item in self.item_index and user not in self.user_index:
                    values.append(self.item_means[self.item_index[item]])
                else:
                    values.append(np.mean(self.item_means) * 0.6 + np.mean(self.user_means) * 0.4)

        return values

    def recommend(self, user_list, N=10, values=False):

        # utilMat element not zero means that element has already been discovered b y the user and can npt be
        # recommended.
        predMat = np.ma.masked_where(self.predMask, self.UsV).filled(fill_value=-999)
        output = []

        if values:
            for user in user_list:
                try:
                    j = self.user_index[user]
                except:
                    raise Exception("Invalid user:", user)
                max_indices = predMat[j, :].argsort()[-N:][::-1]
                output.append([(self.items[index], predMat[j, index]) for index in max_indices])

        else:
            for user in user_list:
                try:
                    j = self.user_index[user]
                except:
                    raise Exception("Invalid user:", user)
                max_indices = predMat[j, :].argsort()[-N:][::-1]
                output.append([self.items[index] for index in max_indices])

        return output

    def topN_similar(self, x, column="item", N=10, weight=True):
        """
        Gives out the most similar contents compared to the input content given. For a user input gives out similar users.
        For an item input, gives out the most similar items.

        :param x: the identifier string for the user or item.
        :param column: either 'user' or 'item'.
        :param N: the number of best matching similar content to output.
        :param weight: True or False. True means the feature differences are weighted. Puts more penalty on the differences
        between bigger features.
        :return: A list of tuples.
        """

        output = list()

        if column == 'user':
            if x not in self.user_index:
                raise Exception("Invalid user")
            else:
                for user in self.user_index:
                    if user != x:
                        temp = dissimilarity(self.Usk[self.user_index[user], :], self.Usk[self.user_index[x], :], weighted=weight)
                        output.append((user, temp))

        if column == 'item':
            if x not in self.item_index:
                raise Exception("Invalid item")
            else:
                for item in self.item_index:
                    if item != x:
                        temp = dissimilarity(self.skV[:, self.item_index[item]], self.skV[:, self.item_index[x]], weighted=weight)
                        output.append((item, temp))

        output = special_sort(output, order='ascending')
        output = output[:N]

        return output
