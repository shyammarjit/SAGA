import math 
from sklearn.feature_selection import mutual_info_classif
import pandas as pd 
import numpy as np

def dropping(x_vector, y_vector, input_percentage):
    # remove the lowest k percentage features from mi_pcc
    cutoff = int(math.floor(input_percentage*(x_vector.shape[1])))

    fecture_vector = list(x_vector.columns)
    removed_feature_list, selected_features = [], []
    for i in range(0, cutoff):
      for j in range(0, mi_pcc_sort.shape[0]):
            if(mi_pcc_sort[i]==mi_pcc[j]):
                removed_feature_list.append(fecture_vector[j])
    
    for k in fecture_vector:
        st = 1
        for i in removed_feature_list:
            if(i==k):
                st = 0
                break
        if(st==1):
            selected_features.append(k)
    
    data = x_vector[selected_features]

    mi_selected_features = mutual_info_classif(data, y_vector)
    return data, mi_selected_features


def MICC(x_vector, y_vector, percentage):
    """
    Input:  x_vector: 2D array shape of (no_of_samples, no_of_features)
            y_vector: 1D array shape of (no_of_sample, )
    Output: data: which is equal to x_vector shape of (no_of_samples, no_of_selected_features)
    Functionality:  Select the top k percentage of features based on mi and pcc.
                    Adopted from paper: https://sci-hub.se/10.1109/calcon49167.2020.9106516
    """
    feature_pcc, selected_features, removed_feature_list = [], [], []
    a = 0.9
    
    # Drop constant features
    x_vector = x_vector.loc[:,x_vector.apply(pd.Series.nunique) != 1]

    # calculate the mutual information between class and feature vector
    mi = mutual_info_classif(x_vector, y_vector)

    # calculate the pcc between feature to feature
    fecture_vector = list(x_vector.columns)
    corr_matrix = x_vector.corr()
    corr_matrix = abs(np.array(corr_matrix))
    for i in range(0, len(corr_matrix)):
        feature_pcc.append((np.sum(corr_matrix[i]) - 1)/(len(corr_matrix)-1))
    
    # compute the final rank value for feature vector
    mi_pcc = a*mi - (1-a)*np.array(feature_pcc)

    # remove the lowest k percentage features from mi_pcc
    cutoff = int(math.floor(percentage*(x_vector.shape[1])))

    # sort the array in ascending order
    mi_pcc_sort = mi_pcc.copy()
    mi_pcc_sort.sort()

    sort_features_list = []
    mi_sorted = []
    for i in range(0, mi_pcc_sort.shape[0]):
        for j in range(0, mi_pcc.shape[0]):
            if(mi_pcc_sort[i]==mi_pcc[j]):
                sort_features_list.append(fecture_vector[j])
                #mi_sorted.append()
                mi_pcc[j] = 100000000
                break
    selected_features = sort_features_list[cutoff:]

    # create the x data based on selected feature vector
    data = x_vector[selected_features]
    return data,  mutual_info_classif(data, y_vector)