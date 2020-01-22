from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")

import torch

import support_files.configuration as configuration
import support_files.image_utilities as image_utilities
import support_files.noise_model as noise_model
import support_files.relaxometry_models as relaxometry_models


def printLoss(args):
    print("Iteration: {}, Loss: {}".format(args['it'], args['loss']))


def forwardModel(kappa, observedSeries, noiseStd, modelOp):
    weightedSeries = modelOp[0].generate_weighted_sequence(kappa)
    lossNoise = modelOp[1].forward(weightedSeries, observedSeries, noiseStd)
    
    return lossNoise


def predictMapping(observedSeries, configurationFilePath, verbose):
    observedWeightedSeries = torch.from_numpy(observedSeries["weighted_series"])
    echoTimes = torch.from_numpy(observedSeries["echo_times"])

    opts = configuration.get_configuration_parameters(configurationFilePath)
    varKappa, brainMmask = image_utilities.get_initial_parameters(observedWeightedSeries, opts)
    
    noiseOp = noise_model.Noise_operator(opts['noise_type'])
    relaxometryOp = relaxometry_models.LookLocker_T1(echoTimes)
    
    varOptim = list(varKappa["kappa"]) + list([varKappa["std_noise"]])
    optimizer = torch.optim.Adam(varOptim, lr=opts["learning_rate_relaxometry"], betas=(0.8, 0.699))
    
    args = {}
    for epoch in range(opts["training_epochs"]):
        optimizer.zero_grad()
        estimated_loss = forwardModel(varKappa["kappa"], observedWeightedSeries, varKappa["std_noise"], [relaxometryOp, noiseOp])
        estimated_loss.backward()
        optimizer.step()
        
        args['loss'] = estimated_loss.item()
        args['it'] = epoch
        
        if verbose:
            printLoss(args)

    simulatedWeightedSeries = relaxometryOp.generate_weighted_sequence(varKappa["kappa"])
        
    return varKappa["kappa"], simulatedWeightedSeries.detach().numpy()
    

