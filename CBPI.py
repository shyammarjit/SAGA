import numpy as np
import random
from utils import getFitness, isEqual, get_individual

class population_initalization(object):
    """clustering_based_population_initalization: ver0"""
    def __init__(self, cfg, mi_features):
        self.cfg = cfg
        self.mi_features = mi_features
        self.cbpi_ver = cfg.cbpi_ver
    
    def check_overlapping_for_cluster(self, pop):
        """
        Input:  population
                overlapping_status: to show the results
        Output: population, population_fitness, population_accuracy
        Functionality:  Takes an population and checks if there are
                        multiple copies of a individual then remove
                        all these copies.
        """
        pop_copy = pop.copy()
        remove_index = []
        for i in range(0, len(pop)):
            counter = i
            for j in pop[i+1:]:
                counter += 1
                # If two individuals are same then store these same indiviual's index number for delete.
                if(isEqual(pop[i], j)):
                    remove_index.append(counter)
        remove_index = list(set(remove_index))
        remove_index.sort(reverse = True)
    
        # delete the copy individuals from populations
        for i in remove_index:
            del pop_copy[i]
        return pop_copy
    
    def hammingDist(self, ind1, ind2):
        """
        Input: two individuals - ind1 and ind2
        Output: an integer value
        Functionality: compute the hamming distance between two individuals
        """
        i, count = 0, 0
        while(i < len(ind1)):
            if(ind1[i] != ind2[i]):
                count += 1
            i += 1
        return count

    def compute_similarity(self, ind_datapoint, cluster_datapoint, ind_acc, cluster_acc):
        # compute the hamming distance between the cluster_datapoint and ind_datapoint
        hamming_dist = self.hammingDist(cluster_datapoint, ind_datapoint)
        # difference between classification accuracy
        diff_acc = abs(ind_acc-cluster_acc)
        if(diff_acc==0):
            diff_acc = 0.001
        if(hamming_dist==0):
            hamming_dist = 0.001
        similarity = self.cfg.esp*(1/diff_acc) + (1-self.cfg.esp)*(1/hamming_dist)
        return similarity

    def get_random_pop(self, size):
        random_population = []
        for i in range(0, size):
            random_population.append(get_individual(self.total_no_features))
        # check there is any overlapping in between or not
        random_population = self.check_overlapping_for_cluster(random_population)
        return random_population
            
    def CBPI_ver0(self, dataset):
        """
        Input: an 2D array of shape (population_size, no_of_features)
        Output: an 2D array of shape (population_size, no_of_features)
        Functionality: Cluster based population generation
        """
        self.total_no_features = dataset['x_train'].shape[1]
        population, population_fit, population_acc = [], [], []
        random_pop = self.get_random_pop(4*self.cfg.population_size)
        no_cluster_center = self.cfg.population_size # m = no_cluster_center i.e. should be 
        
        """
        Randomly Initialize the cluster center
        """
        cluster_centers = random.sample(random_pop, no_cluster_center)

        """
        now discard the cluster centers individuals from populations
        """
        remove_index = []
        for i in range(0, len(random_pop)):
            flag = 0
            for j in range(0, len(cluster_centers)):
                if(isEqual(random_pop[i], cluster_centers[j])):
                    flag = 1
                    break
            if(flag==1):
                remove_index.append(i)

        remove_index.sort(reverse = True)
        for i in remove_index:
            del random_pop[i]
        
        #------------------------------------------------------------

        # find the fitness of the non-cluter center individual
        non_cluster_pop_fit, non_cluster_pop_acc = [], []

        for i in range(0, len(random_pop)):
            ind_fit_acc = getFitness(self.cfg, random_pop[i], self.mi_features, dataset)
            non_cluster_pop_fit.append(ind_fit_acc[0])
            non_cluster_pop_acc.append(ind_fit_acc[1])

        # find the fitness of the cluster individual
        cluster_pop_fit, cluster_pop_acc = [], []
        for i in range(0, len(cluster_centers)):
            ind_fit_acc = getFitness(self.cfg, cluster_centers[i], self.mi_features, dataset)
            cluster_pop_fit.append(ind_fit_acc[0])
            cluster_pop_acc.append(ind_fit_acc[1])

        #-----------------------------------------------------------
        # Assign the individual to a particular cluster
        ind_belongs_to_cluter = []
        for i in range(0, len(random_pop)):
            ind_fit = non_cluster_pop_fit[i]  # fitness of the current individual
            ind_acc = non_cluster_pop_acc[i]  # accuracy of the current individual

            similar_list  = []
            for j in range(0, no_cluster_center):
                cluster_fit = cluster_pop_fit[j]  # fitness of the cluster center
                cluster_acc = cluster_pop_acc[j] # accuracy of the cluster center 
                
                # compute similarity between the cluster center and the current individual
                similarity = self.compute_similarity(random_pop[i], cluster_centers[j], ind_acc, cluster_acc)
                similar_list.append(similarity)
            # get the cluster no which has the highest similarity
            ind_belongs_to_cluter.append(similar_list.index(max(similar_list)))
        #-----------------------------------------------------------    
        #print(ind_belongs_to_cluter)
        """
        Now we will be taking only one individual out the cluster 
        to do so there may be some cluster which has only one point
        then at that case don't compute the part-2 rather than that
        directly add that cluster center into the final ouput population.
        """
        li1 = set(ind_belongs_to_cluter)
        li2 = [i for i in range(self.cfg.population_size)]
        direct_add = list(set(li1) - set(li2)) + list(set(li2) - set(li1))
        direct_add.sort(reverse=True)
        # directly add then into final population from cluster center
        for d in direct_add:
            population.append(cluster_centers[d])
            population_fit.append(cluster_pop_fit[d])
            population_acc.append(cluster_pop_acc[d])
        #----------------------------------------------------------



        #-----------------------------------------------------------
        for i in range(0, len(cluster_centers)):
            status = False
            # if the cluster has only the cluster center then don't comput the nested loop
            for temp in direct_add:
                if(temp==i):
                    status = True
                    break
            if(status):
                continue
            cluster_fit = cluster_pop_fit[i]

            # go for the individual belongs to this cluster
            ind_index = []
            same_cluster_ind_fit = []
            for j in range(0, len(ind_belongs_to_cluter)):
                if(ind_belongs_to_cluter[j]==i):
                    ind_index.append(j)
                    #store this individuals fitness value
                    same_cluster_ind_fit.append(non_cluster_pop_fit[j])
            # compute the maximum fitness
            if(max(same_cluster_ind_fit)>cluster_fit):
                # put the heighest fit individual to the population cluster
                put_index = ind_index[same_cluster_ind_fit.index(max(same_cluster_ind_fit))]
                population.append(random_pop[put_index])
                population_fit.append(non_cluster_pop_fit[put_index])
                population_acc.append(non_cluster_pop_acc[put_index])
            else:
                # put the cluster center into the population
                population.append(cluster_centers[i])
                population_fit.append(cluster_pop_fit[i])
                population_acc.append(cluster_pop_acc[i])
        #-------------------------------------------------------------------
        return np.array(population), np.array(population_fit), np.array(population_acc)

    def CBPI_ver1(self, population_size):
        """
        Input: an 2D array of shape (population_size, no_of_features)
        Output: an 2D array of shape (population_size, no_of_features)
        Functionality: Cluster based population generation
        """
        def goodness(input_cluster, cluster_acc):
            final_ind = []
            if(len(input_cluster)==1):
                return input_cluster[0]
            input_cluster = np.array(input_cluster)
            gij = []
            for ith_feature in range(0, input_cluster.shape[1]):
                numerator, denominator = 0, 0
                for no_of_inds in range(0, input_cluster.shape[0]):
                    numerator += input_cluster[no_of_inds][ith_feature]*cluster_acc[no_of_inds]
                    denominator += input_cluster[no_of_inds][ith_feature]
                temp = numerator/denominator
                gij.append(temp)
            x = np.array(gij)
            t_mean = np.mean(x[~np.isnan(x)])
            for ith_feature in range(0, len(gij)):
                if(np.isnan(gij[ith_feature])):
                    # if value is nan then just simply put 0
                    final_ind.append(0)
                    continue
                if(gij[ith_feature]>=t_mean):
                    final_ind.append(1)
                else:
                    final_ind.append(0)
            return final_ind
        
        population, population_fit, population_acc = [], [], []
        random_pop = self.get_random_pop(4*self.population_size)
        no_cluster_center = population_size # m = no_cluster_center i.e. should be 
        
        # Randomly Initialize the cluster center
        cluster_centers = random.sample(random_pop, no_cluster_center)

        # now discard the cluster centers individuals from populations
        remove_index = []
        for i in range(0, len(random_pop)):
            flag = 0
            for j in range(0, len(cluster_centers)):
                if(isEqual(random_pop[i], cluster_centers[j])):
                    flag = 1
                    break
            if(flag==1):
                remove_index.append(i)

        remove_index.sort(reverse = True)
        for i in remove_index:
            del random_pop[i]
        
        #------------------------------------------------------------
        # find the fitness of the non-cluter center individual
        non_cluster_pop_fit, non_cluster_pop_acc = [], []

        for i in range(0, len(random_pop)):
            ind_fit_acc = getFitness(random_pop[i])
            non_cluster_pop_fit.append(ind_fit_acc[0])
            non_cluster_pop_acc.append(ind_fit_acc[1])
        #------------------------------------------------------------
        # find the fitness of the cluster centers --> individual
        cluster_pop_fit, cluster_pop_acc = [], []
        for i in range(0, len(cluster_centers)):
            ind_fit_acc = getFitness(cluster_centers[i])
            cluster_pop_fit.append(ind_fit_acc[0])
            cluster_pop_acc.append(ind_fit_acc[1])

        #-----------------------------------------------------------
        # Assign the individual to a particular cluster
        ind_belongs_to_cluter = []
        for i in range(0, len(random_pop)):
            ind_fit = non_cluster_pop_fit[i]  # fitness of the current individual
            ind_acc = non_cluster_pop_acc[i]  # accuracy of the current individual

            similar_list  = []
            for j in range(0, no_cluster_center):
                cluster_fit = cluster_pop_fit[j]  # fitness of the cluster center
                cluster_acc = cluster_pop_acc[j]  # accuracy of the cluster center 
                
                # compute similarity between the cluster center and the current individual
                similarity = cal_similarity(random_pop[i], cluster_centers[j], ind_acc, cluster_acc)
                similar_list.append(similarity)
            # get the cluster no which has the highest similarity
            ind_belongs_to_cluter.append(similar_list.index(max(similar_list)))
        #-----------------------------------------------------------    
        #print(random_pop, len(random_pop), ind_belongs_to_cluter, len(ind_belongs_to_cluter))
        # now lets form one 3D list in which (cluster_no * no_of_indviduals)
        cluster, cluster_fit, cluster_acc = [], [], []

        for ith_cluster in range(0, len(cluster_centers)):
            temp_ind, temp_fit, temp_acc = [], [], []
            temp_ind.append(cluster_centers[ith_cluster])
            temp_fit.append(cluster_pop_fit[ith_cluster])
            temp_acc.append(cluster_pop_acc[ith_cluster])

            for non_cluster_center in ind_belongs_to_cluter:
                if(non_cluster_center==ith_cluster):
                    temp_ind.append(random_pop[non_cluster_center])
                    temp_fit.append(non_cluster_pop_fit[non_cluster_center])
                    temp_acc.append(non_cluster_pop_acc[non_cluster_center])
            cluster.append(temp_ind)
            cluster_fit.append(temp_fit)
            cluster_acc.append(temp_acc)

        for ith_cluster in range(0, len(cluster)):
            ind = goodness(cluster[ith_cluster].copy(), cluster_acc[ith_cluster].copy())
            population.append(ind)
            temp = getFitness(ind)
            population_fit.append(temp[0])
            population_acc.append(temp[0])

        return population, population_fit, population_acc
    
    def CBPI(self, dataset):
        if(self.cbpi_ver.lower()=="ver0"):
            population, population_fit, population_acc = self.CBPI_ver0(dataset)
        elif():
            population, population_fit, population_acc = self.CBPI_ver1(dataset)
        
        return population, population_fit, population_acc