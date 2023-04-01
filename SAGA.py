import argparse, os
import pandas as pd
from utils import *
from population import *
from MICC import MICC
from GA import Genetic_Algorithm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif


def load_data(dataset_path):
    # Load the data
    df = pd.read_csv(dataset_path)

    # dataset must contain class column
    try:
        features = list(df.columns) # list of all the features
        y = df['class']
        features.remove('class')
    except:
        raise AttributeError("Missing class column. Dataset must contain class column")
    
    x = df[features] # convert dataframe to numpy

    return x, y


def get_data(cfg):
    dataset_path = cfg.dataset_path + cfg.dataset + "/"
        
   
    if(cfg.load_preprocessed_data):
        x_train, y_train = load_data(dataset_path + cfg.dataset + "_train.csv")
        x_test, y_test = load_data(dataset_path + cfg.dataset + "_test.csv")
        return x_train, y_train, x_test, y_test
    else:
        x, y = load_data(dataset_path + cfg.dataset + ".csv")
        return x, y
    

def get_train_test_data(x, y):
    # Devide the data into train:test in 80:20 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)
    data = dict()
    data['x_train'], data['x_test'] = x_train, x_test
    data['y_train'], data['y_test'] = y_train, y_test
    return data

def compute_accuracy(cfg, data):
    classifier = KNeighborsClassifier(n_neighbors = cfg.n_neighbors)
    classifier.fit(data['x_train'], data['y_train'])
    predictions = classifier.predict(data['x_test'])
    total_acc = accuracy_score(y_true = data['y_test'], y_pred = predictions)
    return total_acc*100

def SAGA(cfg):
    if(cfg.load_preprocessed_data): 
        # load the preprocessed data (where feature space are already reduced).
        data = dict()
        x_train, y_train, x_test, y_test = get_data(cfg)
        data['x_train'], data['x_test'] = x_train, x_test
        data['y_train'], data['y_test'] = y_train, y_test
        mi_features = mutual_info_classif(data['x_train'], data['y_train'])
        accuracy_all_features = compute_accuracy(cfg, data)
    else:
        x, y = get_data(cfg) # get the data and class label
        data = get_train_test_data(x, y) # get the train-test data

        # compute the accuracy using all the features
        accuracy_all_features = compute_accuracy(cfg, data)

        if(cfg.dropping_ratio>0):
            # drop the intial features: Mutually Informed Correlation Coefficient (MICC)
            data['x_train'], mi_features = MICC(data['x_train'], data['y_train'], cfg.dropping_ratio)
            fecture_vectors = list(data['x_train'].columns)
            data['x_test'] = data['x_test'][fecture_vectors]
        
        else: # No feature dropping
            mi_features = mutual_info_classif(data['x_train'], data['y_train'])
    
    # get initial population (CBPI followed by SA)
    population, fitness, accuracy = get_population(cfg, data, mi_features)
    assert population.shape[0] == fitness.shape[0] == accuracy.shape[0]

    # Genetic Algorithm
    genetic_algorithm = Genetic_Algorithm(cfg, mi_features)
    solution = genetic_algorithm.run(population, fitness, accuracy, data)
    solution.accuracy_all_features = accuracy_all_features
        
    return solution

def main(cfg):
    # check whether dataset exist or not
    dataset_path = cfg.dataset_path + cfg.dataset + "/"
    if(os.path.isdir(dataset_path)):
        print("{} dataset folder found.".format(cfg.dataset.upper()))
        # get the data and class label
        x, y = get_data(cfg)
        total_features = x.shape[1] # count total no of features
    else:
        raise AttributeError("Dataset not found")
    
    if(cfg.load_preprocessed_data):
        print("Loading preprocessed data.")
    else:
        print("Loading Original data.")
        
    
    
    runs_accuracy, runs_fitness, run_nof = [], [], []
    for run in range(cfg.num_runs):
        solution = SAGA(cfg)
        
        # print the info
        info = []
        info += [f"run [{run}/{cfg.num_runs}]"]
        info += [f"accuracy [{solution.best_accuracy:.2f}/{solution.accuracy_all_features:.2f}]"]
        info += [f"num_features [{sum(solution.best_individual)}/{total_features}]"]
        info += [f"fitness {solution.best_fitness:.2f}"]
        print(" ".join(info))
        
        # store the info
        runs_accuracy.append(solution.best_accuracy)
        runs_fitness.append(solution.best_fitness)
        run_nof.append(sum(solution.best_individual))
    
    # sort them based on Accuracy then Fitness
    best_accuracy, best_fitness, best_nof = zip(*sorted(zip(runs_accuracy, runs_fitness, run_nof), reverse = True))
    print("\nBest accuracy: {}".format(best_accuracy[0]))
    print("Best fitness: {:.2f}".format(best_fitness[0]))
    print("No of selected features: {}".format(best_nof[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # list all the dataset names
    dataset = ['amlgse2191', 'braintumor', 'leukaemiamattest', 'lung', 'prostatemattest', 'srbctgene', \
                'GSE62191', 'GSE92538', 'PXD000672', 'PXD002099', 'PXD002882', 'PXD003028', 'PXD006129']
    # dataset path
    dataset_path = "./DATASET/"

    # output path (where log and convegence graphs will be saved)
    output_path = "./output/"

    parser.add_argument("--dataset", default = 'srbctgene', type = str, choices = dataset, help = "name of the dataset")
    parser.add_argument("--dataset_path", default = dataset_path, type = str, help = "path of the dataset")
    parser.add_argument("--output_path", default = output_path, type = str, help = "path output")
    parser.add_argument("--dropping_ratio", default = 0.995, type = float, help = "must be in range of [0, 1)")
    parser.add_argument("--max_iter", default = 100, type = int, help = "maximum number of iterations in SA")
    parser.add_argument("--alpha", default = 0.93, type = float, help = "cooling factor in SA, must be in range of (0, 1]")
    parser.add_argument("--n_neighbors", default = 5, type = int, help = "no of neighbors in KNN")
    parser.add_argument("--esp", default = 0.7, type = float, help = "Weight for inverse hamming distance")
    parser.add_argument("--mul_factor_pop", default = 4, type = int, help = "Multiplication factor of the population")
    parser.add_argument("--population_size", default = 30, type = int, help = "Population size of GA")
    parser.add_argument("--beta", default=0.80, type = float, help = "Weight for accuracy in fitness function")
    parser.add_argument("--gamma", default=0.70, type = float, help =  "Weight for the fraction of features left out in fitness function")
    parser.add_argument("--SA_mutation_methods", default = "30-40", choices=["middle", "30-40"], help = "mutation method in SA")
    parser.add_argument("--save_conv_graph", default = False, choices=[True, False],\
        help = "whether you want to save the GA convengence graph or not.")
    parser.add_argument("--num_generations", default=50, help = "number of generations in GA")
    
    parser.add_argument("--prob_cross", default = 0.7, help = "")
    parser.add_argument("--prob_mut", default  = 0.3, help = "")
    parser.add_argument("--cross_limit", default  = 10, help = "")
    parser.add_argument("--prob_cross_max", default  = 0.7, help = "")
    parser.add_argument("--prob_cross_min", default  = 0.6, help = "")
    parser.add_argument("--prob_mut_max", default  = 0.4, help = "")
    parser.add_argument("--prob_mut_min", default  = 0.3, help = "")
    parser.add_argument("--num_runs", default=5, type = int, help = "number of independent runs,\
                        final result will be the best result of all these runs")
    parser.add_argument("--cbpi_ver", default="ver0", choices=['ver0', 'ver1'], help = "")
    parser.add_argument("--load_preprocessed_data", default = False)
    args = parser.parse_args()

    # validate all the arguments
    validate_args(args)

    # SAGA
    main(args)

