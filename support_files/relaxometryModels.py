from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import support_files.noiseModel as noise_model
import support_files.weightedSequence as weightedSequence
import torch

        
class InversionRecovery_T1(weightedSequence.BaseWeightedSequence):
    number_of_parameters = 3

    def __init__(self, *args):
        self.echo_times = args[0]
        
    def forward_model(self, args):
        ro_map = torch.abs(args[0])
        b_map = args[1]
        t1_map = torch.abs(args[2])
        inverstion_time = args[3]
        return torch.abs(ro_map + (b_map * torch.exp(torch.div(inverstion_time*(-1), t1_map))))
        
    def generate_weighted_sequence(self, args):
        
        #TODO Have to insert checks for each variable.
        ro_map = args[0]
        b_map = args[1]
        t1_map = args[2]

        if isinstance(t1_map, torch.Tensor):
            generated_weighted_sequence = []
            for inverstion_time in self.echo_times:
                generated_weighted_sequence.append(self.forward_model([ro_map, b_map, t1_map, inverstion_time]))
            
            return torch.stack(generated_weighted_sequence)
        else:
            print("You should provide Tensors")
            return 1

    def gradients(self, args, measurements=[]):

        ro_map = args[0]
        b_map = args[1]
        t1_map = args[2]
        img_shape = ro_map.size()

        generated_weighted_sequence = self.generate_weighted_sequence([ro_map, b_map, t1_map])
        sum_weighted_sequence_criteria = self.loss_criterion(generated_weighted_sequence, measurements)
        sum_weighted_sequence_criteria.backward(retain_graph=True)

        param_map_gradient = ([param_map.grad.view(img_shape) for param_map in args])

        return torch.stack(param_map_gradient)
        

class LookLocker_T1(weightedSequence.BaseWeightedSequence):

    def __init__(self, *args):
        self.echo_times = args[0]
        self.noise_model = noise_model.Noise_operator("Gaussian")

    def forward_model(self, args):
        ro_map = torch.abs(args[0])
        b_map = args[1]
        t1_map = torch.abs(args[2])
        echo_time = args[3]

        e = torch.exp(torch.div(-echo_time, t1_map))
        s = torch.abs(ro_map * (1 - b_map * e))

        return torch.abs(ro_map + (b_map * torch.exp(torch.div(echo_time*(-1), t1_map))))

    def apply_ll_correction(self, args):
        ro_map = args[0]
        b_map = args[1]
        t1_map = args[2]

        if args[3]=="generate":
            return t1_map / np.abs((b_map/ro_map) -1)
        elif args[3]=="estimate":
            return t1_map * torch.abs((b_map/ro_map) -1)
        
    def generate_weighted_sequence(self, args):
        ro_map = args[0]
        b_map = args[1]
        t1_map = args[2]

        if isinstance(t1_map, torch.Tensor):
            generated_weighted_sequence = []
            for echo_time in self.echo_times:
                generated_weighted_sequence.append(self.forward_model([ro_map, b_map, t1_map, echo_time]))
            return torch.stack(generated_weighted_sequence)
        else:
            print("You should provide Tensors")
            return 1

    def gradients(self, args, measurements=[]):
        img_shape = args[0].size()

        generated_weighted_sequence = self.generate_weighted_sequence(args)
        sum_weighted_sequence_criteria = self.noise_model(generated_weighted_sequence, measurements, 1)
        sum_weighted_sequence_criteria.backward()
        param_map_gradient = ([param_map.grad.view(img_shape) for param_map in args])

        return torch.stack(param_map_gradient)


class GRE_T2_star(weightedSequence.BaseWeightedSequence):

    def __init__(self, *args):
        self.echo_times = args[0]

    def forward_model(self, args):
        ro_map = torch.abs(args[0])
        t2_star_map = args[1]
        echo_time = args[2]
        return torch.abs(ro_map * torch.exp(torch.div(echo_time*(-1), t2_star_map)))

    def generate_weighted_sequence(self, args):
        #TODO Have to insert checks for each variable.
        ro_map = args[0]
        t2_star_map = args[1]

        if isinstance(ro_map, torch.Tensor):
            generated_weighted_sequence = []
            for echo_time in self.echo_times:
                generated_weighted_sequence.append(self.forward_model([ro_map, t2_star_map, echo_time]))
            
            return torch.stack(generated_weighted_sequence)
        else:
            print("You should provide Tensors")
            return 1

    def gradients(self, args, measurements=[]):
        ro_map = args[0]
        t2_star_map = args[1]
        img_shape = ro_map.size()

        generated_weighted_sequence = self.generate_weighted_sequence([ro_map, t2_star_map])
        sum_weighted_sequence_criteria = self.loss_criterion(generated_weighted_sequence, measurements)
        sum_weighted_sequence_criteria.backward(retain_graph=True)

        param_map_gradient = ([param_map.grad.view(img_shape) for param_map in args])

        return torch.stack(param_map_gradient)


class MSE_T2(weightedSequence.BaseWeightedSequence):

    def __init__(self, *args):
        self.echo_times = args[0]
        self.noise_model = noise_model.Noise_operator("Gaussian")

    def forward_model(self, args):
        ro_map = torch.abs(args[0])
        t2_map = torch.abs(args[1])
        echo_time = args[2]
        return torch.abs(ro_map * torch.exp(torch.div(echo_time*(-1), t2_map)))

    def generate_weighted_sequence(self, args):
        #TODO Have to insert checks for each variable.
        ro_map = args[0]
        t2_map = args[1]

        if isinstance(t2_map, torch.Tensor):
            generated_weighted_sequence = []
            for echo_time in self.echo_times:
                generated_weighted_sequence.append(self.forward_model([ro_map, t2_map, echo_time]))
            
            return torch.stack(generated_weighted_sequence)
        else:
            print("You should provide Tensors")
            return 1
    
    def gradients(self, args, measurements=[]):
        ro_map = args[0]
        t2_map = args[1]
        img_shape = ro_map.size()

        generated_weighted_sequence = self.generate_weighted_sequence([ro_map, t2_map])
        sum_weighted_sequence_criteria = self.noise_model(generated_weighted_sequence, measurements, 1)
        sum_weighted_sequence_criteria.backward(retain_graph=True)

        param_map_gradient = ([param_map.grad.view(img_shape) for param_map in args])

        return torch.stack(param_map_gradient)
        

def get_relaxometry_model(configuration_file, echo_times):
    if configuration_file["contrast"] == "T1":
        relaxometry_model = LookLocker_T1(echo_times)
    elif configuration_file["contrast"] == "T2":
        relaxometry_model = MSE_T2(echo_times)
        
    return relaxometry_model
    
