import torch
import numpy as np
from torch.autograd import Variable
import itertools
from copy import deepcopy
import torch.optim as optim


class EvolutionTrainer(object):
    def __init__(self, model, optimizer, loss_func, 
                 train_loader, test_loader, args, 
                 convergence_threshold, batch_epochs=50):
        
        self.model = model
        self.args = args
        self.loss_func = loss_func
        self.batch_epochs = batch_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.convergence_threshold = convergence_threshold
        self.optimizer = optimizer
      
    def initialize_pathways(self):
        layer_configs = list(itertools.combinations_with_replacement(
                                    list(range(self.args.M)), self.args.N))
        layer_configs = np.array(layer_configs)
        indices = np.random.choice(len(layer_configs), (self.args.P, self.args.L))
        pathways = layer_configs[indices]

        return pathways # Shape: P x L x N
    
    def mutate(self, pathway):
        prob_mutate = 1./ (self.args.L * self.args.N) # Increase probability of mutation

        # Probability of mutation for every element
        prob = np.random.rand(self.args.L, self.args.N)

        # Mutations for chosen elements
        permutations = np.random.randint(-2, 2, size=(self.args.L, self.args.N))
        permutations[prob > prob_mutate] = 0

        # Mutate
        pathway = (pathway + permutations) % self.args.M
        
        return pathway
    
    def evaluate(self, pathway):
        correct = 0
        
        for x, y in self.test_loader:
            x, y = Variable(x, volatile=True), Variable(y, volatile=True)
            
            output = self.model(x, pathway)
            _, pred = torch.max(output.data, 1)
            
            correct += (pred == y.data).sum()
            
        accuracy = correct * 1.0 / len(self.test_loader) / self.args.batch_size

        return accuracy
    
    def train_model(self, pathway):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Stop training after 50 batches, evaluate fitness
            if batch_idx >= self.batch_epochs:
                fitness = self.evaluate(pathway)
                return fitness

            self.optimizer.zero_grad()

            data, target = Variable(data), Variable(target)
            output = self.model(data, pathway)

            loss = self.loss_func(output, target)

            loss.backward()
            self.optimizer.step()
    
    def train(self):
        self.model.train()
        
        fitnesses = []
        best_pathway = None
        best_fitness = -float('inf')
        pathways = self.initialize_pathways()
        gen = 0
        
        while best_fitness < self.convergence_threshold:
            chosen_pathways = pathways[np.random.choice(self.args.P, 2)]
            
            current_fitnesses = []
            
            for pathway in chosen_pathways:
                fitness = self.train_model(pathway)
                
                current_fitnesses.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_pathway = pathway
                
            # All pathways finished evaluating, copy the one with highest fitness
            # to all other ones and mutate
            pathways = np.array([best_pathway] + [self.mutate(deepcopy(best_pathway)) 
                                              for _ in range(self.args.P - 1)])
            
            fitnesses.append(max(current_fitnesses))
            
            if gen % 20 == 0:
                print('Generation {} best fitness is {}'.format(gen, best_fitness))
            gen += 1
        
        # Task training is done
        self.model.done_task(best_pathway)
        
        return best_pathway, gen, fitnesses