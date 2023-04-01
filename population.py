from CBPI import *
from SA import Simulated_Annealing
import pandas as pd
from utils import *

def check_overlapping(pop, pop_fitness, pop_acc):
    """
    Input: population, population_fitness, population_accuracy
    Output: population, population_fitness, population_accuracy
    Functionality:  Takes an population and checks if there are
                    multiple copies of a individual then remove
                    all these copies.
    """
    enable_output = False # to show the results
    pop_copy = pop.copy()
    pop_fitness_copy = pop_fitness.copy()
    pop_acc_copy = pop_acc.copy()
    remove_index = []
    for i in range(0, len(pop)):
        counter = i
        for j in pop[i+1:]:
            counter += 1
            """
            If two individuals are same then store these same
            indiviual's index number for delete.
            """
            if(isEqual(pop[i], j)):
                remove_index.append(counter)
    remove_index = list(set(remove_index))
    remove_index.sort(reverse = True)
    
    # delete the copy individuals from populations
    for i in remove_index:
        del pop_copy[i]
        del pop_fitness_copy[i]
        del pop_acc_copy[i]
    
    if(enable_output):
        if(len(pop_copy) == len(pop)):
            print("\nNo overlapping has been occured!!")
            print("Current population size: ", len(pop_copy))
        else:
            print("Overlapping occured. Current population size: ", len(pop_copy))
    
    return pop_copy, pop_fitness_copy, pop_acc_copy


def get_population(cfg, dataset, mi_features):
    """
    Input: an integer, population_size
    Output: SA_generated_population, SA_generated_population_fitness,
            SA_generated_population_accuarcy
    Functionality:  generate population with twice of the actual population size.
                    then remove the repetating indivisuals.
                    then choose the best individuals out of the current population.
    """
    # clustering based population initalization (CBPI)
    CBPI = population_initalization(cfg, mi_features)
    SA = Simulated_Annealing(cfg, mi_features, dataset) # get the SA object
    
    population, population_fitness, population_accuracy  = [], [], []
    while(len(population)<=cfg.population_size):
        cbpi_pop, _, _ = CBPI.CBPI(dataset)

        for i in range(0, cfg.population_size):
            # get the cbpi generated individual
            ind = cbpi_pop[i]
            
            # pass the randomly genrated individual to the SA
            SA.simulated_annealing(ind)
            
            best_individual, fitness, acc = SA.result(dataset) # get the SA generated results
            
            population.append(best_individual) # store the SA generated output indivisual 
            population_fitness.append(fitness) # store the SA generated fitness value of the final individual 
            population_accuracy.append(acc) # store the SA generated accuarcy value of the final individual 
      
        # check the repetation is there or not
        population, population_fitness, population_accuracy = check_overlapping(population, population_fitness, population_accuracy)
        #break
    
    """
    If the current population size is equal to the desired population size then return, else
    choose the best indivisuals out the of the current population.
    """

    if(len(population)==cfg.population_size):
        return population, population_fitness, population_accuracy
    else:
        """
        Now choose the best indivisuals out the of the current population.
        Sort by accuracy first then sort by fitness.
        To implement this use dataframe.
        """
        sa_population, sa_fitness, sa_acc = [], [], []
        
        #store in a dataframe format
        sa_df = create_dataframe(population, population_accuracy, population_fitness)

        # sort the dataframe based on first Accuarcy then based on Fitness in decending order.
        sa_df.sort_values(["Accuracy", "Fitness"], axis=0, ascending=False, inplace=True)
        
        # store all of the columns i.e. Individual, Accuracy, Fitness
        sa_population_all = list(sa_df['Individual'])
        sa_acc_all = list(sa_df['Accuracy'])
        sa_fitness_all = list(sa_df['Fitness'])

        # create the population containg best individuals
        for i in range(0, cfg.population_size):
            sa_population.append(list(sa_population_all[i]))
            sa_fitness.append(sa_fitness_all[i])
            sa_acc.append(sa_acc_all[i])
        
        return np.array(sa_population), np.array(sa_fitness), np.array(sa_acc)