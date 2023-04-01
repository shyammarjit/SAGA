import numpy as np
import pandas as pd
import random, math, copy, warnings
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore")
from utils import get_individual, get_features, getFitness

class Simulated_Annealing(object):
    
    def __init__(self, cfg, mi_features, dataset):
        """Intialization of the Hyperparameters of SA."""
        self.cfg = cfg
        self.max_iter = cfg.max_iter
        self.mi_features = mi_features
        self.dataset = dataset

    def isValidInd(self, individual):
        """
        Input: Individual of shape (no_of_selected_feature, )
        Output: binary i.e. True or False
                False: If we have the Invalid Individual
                True: If we have the alid Individual
        Functionality:  checks whether a individual is valid or not.
                        validity means if we have the 0 in all gene positions 
                        of an individual then that individual is not valid.
        """
        if(len(set(individual)) == 1):
            return False
        else:
            return True

    def find_new_individual(self, individual):
        """
        Input:  Current Individual of shape (no_of_selected_feature, )
        Output: New state i.e. Next Individual of shape (no_of_selected_feature, )
        Functionality: generates a new solution/individual from the current solution/individual
        """
        """
        changes: instead of middle point now it's a random point in between 
        int(0.30*total_features) and celling(0.70total_features)
        """
        position = []
        
        if(self.cfg.SA_mutation_methods == "30-40"):
            # methods = "30-40"
            temp_1 = int(self.total_features*0.30)
            temp_2 = int(self.total_features*0.70)
            end_point = random.randint(temp_1, temp_2)

            position.append(random.randint(0, end_point))
            position.append(random.randint(end_point+1, self.total_features-1))
        else:
            # methods = "middle"
            position.append(random.randint(0, math.floor(self.total_features/2)))
            position.append(random.randint(math.floor(self.total_features/2)+1, self.total_features-1))
        
        for i in range(0, 2):
            if individual[position[i]] == 1:
                individual[position[i]] = 0
            else:
                individual[position[i]] = 1
        
        # check wheather the new state/individual is valid or not
        if(self.isValidInd(individual)):
            return individual
        else:
            # if the individual is not valid then generate the new state or new indivisual
            return get_individual(self.total_features)

    def result(self, dataset):
        """
        show the result after the convergence.
        """
        # calculate the no of selected features in optimal individual
        no_of_selected_features = 0
        for i in range(0, len(self.best_individual)):
            if(self.best_individual[i] == 1):
                no_of_selected_features += 1
        
        # get the selected feature vectors name
        selected_features = get_features(self.best_individual, self.dataset)
        _classifier = KNeighborsClassifier(n_neighbors = 5)
        x_train_temp = self.dataset['x_train'][selected_features]
        x_test_temp = self.dataset['x_test'][selected_features]
        _classifier.fit(x_train_temp, dataset['y_train'])
        predictions = _classifier.predict(x_test_temp)
        accuracy = accuracy_score(y_true = self.dataset['y_test'], y_pred = predictions)

        return self.best_individual, self.best_fitness, accuracy

    def simulated_annealing(self, initial_indivsual):
        """
        Input: initial_indivsual, intitial individual of shape(no_of_features, )
        """
        self.total_features = int(self.dataset['x_train'].shape[1])

        fitness, all_ind = [], []
        my_iter = 0
        
        # get the initial fitness
        initial_fitness, intial_acc = getFitness(self.cfg, initial_indivsual, self.mi_features, self.dataset)

        current_individual, current_fitness, current_acc = initial_indivsual, initial_fitness, intial_acc
        self.initial_indivsual = current_individual
        
        run_wise_fitness, run_wise_accuracy, run_wise_no_of_features = [], [], []

        """
        initial temp is equal to 2*N
        where N is the total no of features
        """
        temp = 2*self.total_features

        while my_iter < self.max_iter:
            next_individual = self.find_new_individual(current_individual)
            next_fitness, next_accuracy = getFitness(self.cfg, next_individual, self.mi_features, self.dataset)
            if current_fitness > next_fitness:
                # i.e. bad moves then accept new individual with some probability value
                e_x = (my_iter/temp)*((current_fitness-next_fitness)/current_fitness)
                probabilty = math.exp(-e_x)
                random_value = random.random()
                if random_value <= probabilty: # then accept that bad move
                    current_individual = next_individual
                    current_fitness = next_fitness
                    current_acc = next_accuracy
            else:
                # directly accepts the new state
                current_individual = next_individual
                current_fitness = next_fitness
                current_acc = next_accuracy
            # store the fitness, accuracy and no_of_features in a dataframe and then at the end store as a csv file
            run_wise_fitness.append(current_fitness)
            run_wise_accuracy.append(current_acc)
            run_wise_no_of_features.append(len(get_features(current_individual, self.dataset)))
            

            all_ind.append(np.array(current_individual))
            fitness.append(current_fitness)
            my_iter += 1
            temp = self.cfg.alpha*temp # alpha: 0.93 is the another hyperparameter i.e. it is a cooling factor
        self.best_individual = list(all_ind[fitness.index(max(fitness))])
        self.best_fitness = max(fitness)