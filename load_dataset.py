import pandas as pd
from os.path import join


def load_dataset():
    basepath = '/home/sandornoemi/Letöltések'
    dataset_directory_name = 'ml-100k'
    full_filename = join(basepath, dataset_directory_name, "u.data")

    ratings_data = pd.read_csv(full_filename)
    return ratings_data


def loadMovieLens100k(train_test_split=True, all_columns=False):
    path = '/home/sandornoemi/Letöltések'
    directory_name = 'ml-100k'
    train_filename = join(path, directory_name, 'ua.base')
    test_filename = join(path, directory_name, 'ua.test')
    users_filename = join(path, directory_name, 'u.user')
    items_filename = join(path, directory_name, 'u.item')

    train = pd.read_csv(train_filename, delimiter="\t", header=None)
    test = pd.read_csv(test_filename, delimiter="\t", header=None)
    users = pd.read_csv(users_filename, delimiter="|", header=None, encoding="ISO-8859-1")
    items = pd.read_csv(items_filename, delimiter="|", header=None, encoding="ISO-8859-1")

    # Delete the 'timestamp' column.
    del train[3], test[3]
    train.columns = ['userId', 'movieId', 'rating']
    test.columns = ['userId', 'movieId', 'rating']

    if all_columns:
        del users[4], items[1], items[2], items[3], items[4]
        items = items.rename(columns={0: 'itemId'})
        users.columns = ['userId', 'age', 'gender', 'occupation']

        train = pd.merge(train, users, on="userId")
        train = pd.merge(train, items, on="itemId")

        train['userId'] = train['userId'].astype('str')
        train['itemId'] = train['itemId'].astype('str')
        train['rating'] = train['rating'].astype('float')

        test = pd.merge(test, users, on="userId")
        test = pd.merge(test, items, on="itemId")

        test['userId'] = test['userId'].astype('str')
        test['itemId'] = test['itemId'].astype('str')
        test['rating'] = test['rating'].astype('float')

    if train_test_split:
        return train, test, users, items
    else:
        train = pd.concat([train, test], ignore_index=True)
        train = train.sample(frac=1).reset_index(drop=True)
        return train, users, items
