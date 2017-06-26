import torch
import numpy as np
import torch.nn as nn
import itertools
import torch.optim as optim


class PathNet(nn.Module):
    '''
        The architecture follows the paper specifications
            https://arxiv.org/pdf/1701.08734.pdf
        for SUPERVISED LEARNING tasks
    '''
    def __init__(self, args):
        super(PathNet, self).__init__()
        self.args = args
        
        self.relu = nn.ReLU()
        self.layer1 = [nn.Linear(28 * 28, 20) for i in range(self.args.M)]
        self.layer2 = [nn.Linear(20, 20) for i in range(self.args.M)]
        self.layer3 = [nn.Linear(20, 20) for i in range(self.args.M)]
        
        self.optimizer_params = []
        for m in range(self.args.M):
            self.optimizer_params.append({'params' : self.layer1[m].parameters()})
            self.optimizer_params.append({'params' : self.layer2[m].parameters()})
            self.optimizer_params.append({'params' : self.layer3[m].parameters()})
        
    def sum_layer(self, layer_outputs):
        if len(layer_outputs) == 1:
            return layer_outputs[0]
        
        return [layer_outputs[i] + layer_outputs[i+1] 
                for i in range(len(layer_outputs) - 1)][0]
    
    def forward(self, x, pathway):
        layer1_active_modules_index = list(set(pathway[0]))
        layer2_active_modules_index = list(set(pathway[1]))
        layer3_active_modules_index = list(set(pathway[2]))
        
        layer1_output = [self.relu(self.layer1[m](x)) for m in layer1_active_modules_index]
        layer1_output_sum = self.sum_layer(layer1_output)

        layer2_output = [self.relu(self.layer2[m](layer1_output_sum)) for m in layer2_active_modules_index]
        layer2_output_sum = self.sum_layer(layer2_output)

        layer3_output = [self.relu(self.layer3[m](layer2_output_sum)) for m in layer3_active_modules_index]
        layer3_output_sum = self.sum_layer(layer3_output)

        output = self.last_layer(layer3_output_sum)
        
        return output
        
    def initialize_new_task(self, last_layer):
        self.last_layer = last_layer
        self.optimizer_params.append({'params' : last_layer.parameters()})
        
    def output_shape_calculator(self):
        pass
    
    def get_optimizer_params(self):
        return self.optimizer_params
    
    def done_task(self, best_pathway):
        # Freeze best pathway
        # Re-initialize all others
        layer1_active_modules_index = list(set(best_pathway[0]))
        layer2_active_modules_index = list(set(best_pathway[1]))
        layer3_active_modules_index = list(set(best_pathway[2]))
        
        self.optimizer_params = []
    
        # Freeze and add parameters to train
        for i in range(self.args.M):
            if i in layer1_active_modules_index:
                self.layer1[i].requires_grad = False
            else:
                self.layer1[i].reset_parameters()
                self.layer1[i].requires_grad = True
                self.optimizer_params.append({'params' : self.layer1[i].parameters()})
        
            if i in layer2_active_modules_index:
                self.layer2[i].requires_grad = False
            else:
                self.layer2[i].reset_parameters()
                self.layer2[i].requires_grad = True
                self.optimizer_params.append({'params' : self.layer2[i].parameters()})
        
            if i in layer3_active_modules_index:
                self.layer3[i].requires_grad = False
            else:
                self.layer3[i].reset_parameters()
                self.layer3[i].requires_grad = True
                self.optimizer_params.append({'params' : self.layer3[i].parameters()})