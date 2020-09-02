import numpy as np
import pandas as pd

from svd import SVDRecommender
from svd import *
from load_dataset import *
from algorithms.basic_perturbation import *
from metrics.metrics import rmse
from sklearn.preprocessing import MinMaxScaler

train, test, _, _ = loadMovieLens100k(train_test_split=True)
svd = SVDRecommender(number_of_features=8)

# Create the user-item matrix (utility matrix), the userIds on the rows and the itemIds on the columns.
user_item_matrix, users, items = create_utility_matrix(train, formatizer={'user': 'userId', 'item': 'movieId',
                                                                          'value': 'rating'})

#print(user_item_matrix)

# Normalize the rating values between 0 and 1.
rating_scaler = MinMaxScaler()
normalized_user_item_matrix = rating_scaler.fit_transform(user_item_matrix)

#print(normalized_user_item_matrix)

# Using perturbation on rating values.
perturbed_user_item_matrix = rating_perturbation(normalized_rating_dataset=normalized_user_item_matrix,
                                                 global_sensitivity=1, privacy_parameter=0.1)
#print(perturbed_user_item_matrix)

# Fits the SVD model to the matrix data.
svd.fit(perturbed_user_item_matrix, users, items)

# Predict the ratings from test set.
predicts = svd.predict(test, formatizer={"user": "userId", "item": "itemId"})
print(rmse(predicts, list(test['rating'])))

test_users = [1, 65, 444, 321]
# Recommends 4 undiscovered items per each user.
result = svd.recommend(test_users, N=4)
print(result)

# Output 5 most similar users to user with userId
similars = svd.topN_similar(x=65, N=5, column='user')
print(similars)