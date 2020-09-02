import numpy as np

def get_global_sensitivity(input_dataset):

    minimum_rating = np.amin(input_dataset)
    maximum_rating = np.amax(input_dataset)

    return (maximum_rating - minimum_rating)



def rating_perturbation(normalized_rating_dataset, global_sensitivity, privacy_parameter):
    perturbed_rating_matrix = np.empty(normalized_rating_dataset.shape)
    #sensitivity = get_global_sensitivity(normalized_rating_dataset)

    for u in range(0, np.shape(normalized_rating_dataset)[0]):
        for i in range(0, np.shape(normalized_rating_dataset)[1]):
            perturbed_rating_matrix[u, i] = normalized_rating_dataset[u, i] + np.random.laplace(global_sensitivity / privacy_parameter)

    return perturbed_rating_matrix