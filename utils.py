import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_features(ind, dataset):
    """
    Input: takes an individual of shape (no_of_features, )
    Output: returns the name of the selected features.
    Functionality: converts an individual form to a selected feature vector form.
    """
    # store all the feature vectors of the given dataset
    all_features = list(dataset['x_train'].columns)
    features_names = []
    for i in range(0, len(ind)):
        if(ind[i]==1):
            features_names.append(all_features[i])
    return features_names


def isEqual(ind1, ind2):
        """
        Input: two lists (Indivisual_1, Indivisual_2)
        Output: boolen value i.e. True or False
        Functionality:  If two individual ind1 and ind2 are same then output is True, otherwise False.
        """
        status = False
        for k in range(0, len(ind1)):
            if(ind1[k] == ind2[k]):
                status = True
                continue
            else:
                status = False
                break
        return status

def get_individual(individual_length):
    """
    Input: 
    Output: individual
    Functionality:  generate an individual.
                genes are randomly generated.
    """
    individual = []
    for i in range(0, individual_length):
        gene = random.randint(0, 1)
        individual.append(gene)
    return individual
    

def validate_args(cfg):
    # dropping_ratio must be in [0, 1)
    if(cfg.dropping_ratio>=0 and cfg.dropping_ratio<1):
        pass
    else:
        raise AttributeError("dropping_ratio must be in [0, 1)")
    
    # max_iter must be in range [100, inf]
    if(cfg.max_iter<100):
        raise AttributeError("max_iter must be in range [100, inf]")

    # alpha must be in range (0, 1]
    if(cfg.alpha>0 and cfg.alpha<=1):
        pass
    else:
        raise AttributeError("alpha: cooling factor must be in range (0, 1]")

    # esp must be in range (0, 1)
    if(cfg.esp>0 and cfg.esp<1):
        pass
    else:
        raise AttributeError("esp: 'Weight for inverse hamming distance' must be in range (0, 1)")
    
def create_dataframe(ind, acc, fit):
    """
    Input: individual, accuarcy and fitness
    output: return a dataframe
    """
    data = {'Individual':['123'], 'Accuracy':['123'], 'Fitness': ['123']}
    df = pd.DataFrame(data) # Create DataFrame
    for i in range(0, len(ind)):
        df.loc[len(df.index)] = [ind[i], acc[i], fit[i]]
    df = df.drop(0) # Now delete the fist row
    return df

def concatenate_dataframe(df1, df2):
    """
    Input: two dataframes
    Output: concatenates two input dataframes
    """
    frames = [df1, df2]
    concatenated_dataframe = pd.concat(frames)
    return concatenated_dataframe


def getFitness(cfg, individual, mi_features, dataset):
    """
    Input: takes an individual of shape (no_of_features, )
    Output: retuns a tuple of (fitnesss_value, accuracy)
    Functionality: computes the fitness value of the given individual.
    """
    assert mi_features.shape[0]==dataset['x_train'].shape[1]==dataset['x_test'].shape[1]

    sum_mi = 0

    # calculate the total no of features in the dataset.
    total_no_features = int(dataset['x_train'].shape[1])

    # get the name of selected feature vectors from individual 
    selected_features = get_features(individual, dataset)

    # calculate the no of selected feature vector
    selected_no_fectures = len(selected_features)

    # If the no of selected feature vector is 0 then return (0, 0)
    if(selected_no_fectures==0):
        return (0, 0)

    # objective: classification accuracy
    _classifier = KNeighborsClassifier(n_neighbors = cfg.n_neighbors)
    x_train_temp = dataset['x_train'][selected_features]
    x_test_temp = dataset['x_test'][selected_features]
    _classifier.fit(x_train_temp, dataset['y_train'])
    predictions = _classifier.predict(x_test_temp)
    accuracy = accuracy_score(y_true = dataset['y_test'], y_pred = predictions)

    # objective: mutual information between the selected feature vector and class vector
    for i in range(0, len(individual)):
        if(individual[i]==1):
            sum_mi += mi_features[i]
    sum_mi = sum_mi/selected_no_fectures # normalize

    # objective: fraction of features left out
    fraction_features = (total_no_features - selected_no_fectures)/total_no_features

    # calculate the final multi-objective fitness function
    fitness = cfg.beta * accuracy + (1 - cfg.beta) * (cfg.gamma * fraction_features  + (1 - cfg.gamma) * sum_mi)
    
    return (fitness, accuracy)

class Solution():
    """structure of the solution."""
    def __init__(self):
        self.best_individual = None
        self.best_fitness = None
        self.best_accuracy = None
        self.final_population = None
        self.accuracy_all_features = None