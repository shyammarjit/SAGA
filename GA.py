import random, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

class Genetic_Algorithm(object):

    def __init__(self, cfg, mi_features):
        self.cfg = cfg
        self.population_size = cfg.population_size
        self.save_conv_graph = cfg.save_conv_graph
        self.Leader_fitness = float("-inf")
        self.Leader_accuracy = float("-inf")
        self.history = []
        self.cur_iter = 0
        self.num_generations = cfg.num_generations

        self.solution = None
        self.verbose = False

        self.prob_cross = cfg.prob_cross
        self.prob_mut = cfg.prob_mut
        self.cross_limit = cfg.cross_limit
        self.prob_cross_max = cfg.prob_cross_max
        self.prob_cross_min = cfg.prob_cross_min
        self.prob_mut_max = cfg.prob_mut_max
        self.prob_mut_min = cfg.prob_mut_min
        self.mi_features = mi_features

    def sort_agents(self, agents, fitness, accuracy):
        # sort the agents according to fitness
        idx = np.argsort(-fitness)
        sorted_agents = agents[idx].copy()
        sorted_fitness = fitness[idx].copy()
        sorted_accuracy = accuracy[idx].copy()
        return sorted_agents, sorted_fitness, sorted_accuracy
    
    def get_fitness(self, pop):
        """
        Input: population, 2D array
        Output: an 1D array
        Functionality: calculate the fitness of the population.
        """
        fitness, accuracy = [], []
        for i in range(0, len(pop)):
            ind_fitness = getFitness(self.cfg, pop[i], self.mi_features, self.dataset)
            fitness.append(ind_fitness[0])
            accuracy.append(ind_fitness[1])
        accuracy = np.array(accuracy)
        return np.array(fitness), accuracy

    def initialize(self):
        # set the initial population based on fitness
        self.population, self.fitness, self.accuracy = self.sort_agents(agents = self.population, fitness = self.fitness, accuracy = self.accuracy)
        # save the best fittest individual and it's fitness value
        self.Leader_agent, self.Leader_fitness = self.population[0], self.fitness[0]

    def save_details(self):
        # save some details of every generation
        cur_obj = {
            'population': self.population,
            'fitness': self.fitness,
            'accuracy': self.accuracy,
        }
        self.history.append(copy.deepcopy(cur_obj))

    def display(self):
        # display the current generation details
        if self.verbose:
            for i in range(0, len(self.population)):
                print("agents = ", self.population[i], " fitness = ", self.fitness[i], " accuracy = ", self.accuracy[i])

    def post_processing(self):
        # post processing steps
        fit_acc = self.get_fitness(self.population)
        self.fitness = fit_acc[0]
        self.accuracy = fit_acc[1]
        self.population, self.fitness, self.accuracy = self.sort_agents(agents = self.population, fitness = self.fitness, accuracy = self.accuracy)
        
        if(self.fitness[0] > self.Leader_fitness):
            self.Leader_fitness = self.fitness[0]
            self.Leader_agent = self.population[0, :]
            self.Leader_accuracy = self.accuracy[0]

    def save_solution(self):
        # create a solution object
        self.solution = Solution()

        # update attributes of solution
        self.solution.best_individual = self.Leader_agent
        self.solution.best_fitness = self.Leader_fitness*100
        self.solution.best_accuracy = self.Leader_accuracy*100
        self.solution.final_population = self.population

    def crossover(self, parent_1, parent_2):
        # perform crossover with crossover probability prob_cross
        num_features = parent_1.shape[0]
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        for i in range(num_features):
            if(np.random.rand() < self.prob_cross):
                child_1[i] = parent_2[i]
                child_2[i] = parent_1[i]

        return child_1, child_2

    def mutation(self, chromosome):
        # perform mutation with mutation probability prob_mut
        num_features = chromosome.shape[0]
        mut_chromosome = chromosome.copy()

        for i in range(num_features):
            if(np.random.rand() < self.prob_mut):
                mut_chromosome[i] = 1-mut_chromosome[i]
        
        return mut_chromosome

    def roulette_wheel(self, fitness):
        # perform roulette wheel selection
        maximum = sum([f for f in fitness])
        selection_probs = [f/maximum for f in fitness]
        return np.random.choice(len(fitness), p=selection_probs)

    def cross_mut(self):
        # perform crossover, mutation and replacement
        count = 0
        while(count < self.cross_limit):
            id_1 = self.roulette_wheel(self.fitness)
            id_2 = self.roulette_wheel(self.fitness)

            if(id_1 != id_2):
                child_1, child_2 = self.crossover(self.population[id_1, :], self.population[id_2, :])
                child_1 = self.mutation(child_1)
                child_2 = self.mutation(child_2)

                child = np.array([child_1, child_2])
                child_fit_acc = self.get_fitness(child)
                child_fitness = child_fit_acc[0]
                chile_acc = child_fit_acc[1]
                child, child_fitness, chile_acc = self.sort_agents(child, child_fitness, chile_acc)

                for i in range(2):
                    for j in reversed(range(self.population_size)):
                        if(child_fitness[i] > self.fitness[j]):
                            self.population[j, :] = child[i]
                            self.fitness[j] = child_fitness[i]
                            self.accuracy[j] = chile_acc[i]
                            break
                count+= 1
    
    def get_results(self):
        # sort the dataframe based on first Accuarcy then based on Fitness in decending order.
        self.ga_df.sort_values(["Accuracy", "Fitness"], axis=0, ascending=False, inplace=True)
        
        # store all of the columns i.e. Individual, Accuracy, Fitness
        best_ind = list(self.ga_df['Individual'])[0]
        best_ind_acc = list(self.ga_df['Accuracy'])[0]
        best_ind_fit = list(self.ga_df['Fitness'])[0]
        
        return best_ind, best_ind_acc, best_ind_fit, len(best_ind)

    def run(self, population, population_fitness, population_accuracy, dataset):
        self.population = population
        self.fitness = population_fitness
        self.num_features = dataset['x_train'].shape[1]
        self.accuracy = population_accuracy
        self.dataset = dataset
        
        global ga_run_wise_accuracy, ga_run_wise_fitness, ga_run_wise_no_of_features
        self.max_fit_per_gen, self.max_acc_per_gen = [], []
        self.initialize()   # initialize the algorithm
        self.save_details() # save the initial details
        self.max_fit_per_gen.append(np.max(self.fitness))
        self.max_acc_per_gen.append(np.max(self.accuracy))

        # create the dataframe to store the population and it's fitness value and the accuracy
        self.ga_df = create_dataframe(self.population, self.accuracy, self.fitness)
        ga_run_wise_accuracy, ga_run_wise_fitness, ga_run_wise_no_of_features = [], [], []
        for generation in range(self.num_generations):    # while the end criterion is not met
            ga_run_wise_accuracy.append(self.accuracy)
            ga_run_wise_fitness.append(self.fitness)
            for ind in self.population:
                ga_run_wise_no_of_features.append(len(get_features(ind, dataset)))

            self.cross_mut()

            self.cur_iter+= 1

            # store all the population and fitness and the accuracy in the dataframe format
            # add current individual, accuracy and fitness in the dataframe
            self.ga_df_next = create_dataframe(self.population, self.accuracy, self.fitness)
            self.ga_df = concatenate_dataframe(self.ga_df_next, self.ga_df)

            """
            # Adopt the crossover and mutation probability
            f_dash is the fitness of one of the parent
            f_avg is the average fitness of the population.
            f is the fitness of the one for mutation. i.e. kind of a thershold value.
            """
            f_avg = np.mean(self.fitness)
            # for crossover
            f_dash = self.fitness[random.randint(0, int(self.fitness.shape[0]/2))]
            if(f_dash > f_avg):
                self.prob_cross = self.prob_cross_max - generation*((self.prob_cross_max - self.prob_cross_min)/self.num_generations)
            else:
                self.prob_cross = self.prob_cross_max
        
            # for mutation
            f = 0.80
            if(f>f_avg):
                self.prob_mut = self.prob_mut_min + generation*((self.prob_mut_max - self.prob_mut_min)/self.num_generations)
            else:
                self.prob_mut = self.prob_mut_min

            #----------------------------------------------

            self.post_processing()          # do the post processing steps
            self.display()                  # display the details of 1 iteration
            self.save_details()             # save the details
            self.max_fit_per_gen.append(np.max(self.fitness))
            self.max_acc_per_gen.append(np.max(self.accuracy))

        fit_acc = getFitness(self.cfg, self.Leader_agent, self.mi_features ,self.dataset)
        self.Leader_fitness = fit_acc[0]
        self.Leader_accuracy = fit_acc[1]

        self.save_solution()
        
        return self.solution
        