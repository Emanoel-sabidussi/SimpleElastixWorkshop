from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import torch

def get_configuration_parameters(configuration_file_path):
    """
        Function: get_configuration_parameters(configuration_file_path)
        Description: This function parses a JSON file containing the configuration parameters a method

        Args:
            INPUT: 
                - configuration_file_path (STRING): Path to the JSON configuration file. 
            OUTPUT: 
                - settings (DICT): A dictionary containing the configuration parameters for the specific method.
    """

    with open(configuration_file_path) as json_file:  
        config_params = json.load(json_file)

    if config_params["Configuration-file"]["global-config"]["model"] == "MLE":
        settings = get_mle_config_params(config_params)
        settings["model"] = "mle"

    if settings["use_cuda"] and torch.cuda.is_available():
        print("Using CUDA")
        settings["device"] = torch.device("cuda:0")
    elif settings["use_cuda"]:
        print("CUDA is not available")
        settings["device"] = torch.device("cpu")
    else:
        print("Using CPU")
        settings["device"] = torch.device("cpu")

    return settings


def get_mle_config_params(configuration_parameters):
    parameters= {}
    parameters["method"] = configuration_parameters["Configuration-file"]["global-config"]["model"]
    parameters["contrast"] = configuration_parameters["Configuration-file"]["global-config"]["contrast"]
    parameters["use_cuda"] = configuration_parameters["Configuration-file"]["global-config"]["use_cuda"]
    
    # Inference parameters
    parameters['learning_rate_relaxometry'] = configuration_parameters["Configuration-file"]["optimizer-parameters"]["learning_rate_relaxometry"]
    parameters['learning_rate_motion'] = configuration_parameters["Configuration-file"]["optimizer-parameters"]["learning_rate_relaxometry"]
    parameters['training_epochs'] = configuration_parameters["Configuration-file"]["optimizer-parameters"]["training_epochs"]
    parameters['relaxometry_epochs'] = configuration_parameters["Configuration-file"]["optimizer-parameters"]["relaxometry_epochs"]
    parameters['motion_epochs'] = configuration_parameters["Configuration-file"]["optimizer-parameters"]["motion_epochs"]

    # Acquisition parameters
    parameters["voxel_size"] = configuration_parameters["Configuration-file"]["acquisition-parameters"]["voxel_size"]

    # Image information parameters
    parameters['noise_type'] = configuration_parameters["Configuration-file"]["image-parameters"]["noise_type"]
    parameters['padding_size'] = configuration_parameters["Configuration-file"]["image-parameters"]["padding_size"]

    # Data paths
    parameters['dataset_path'] = configuration_parameters["Configuration-file"]["data-paths"]["dataset_path"]
    parameters['save_inference_path'] = configuration_parameters["Configuration-file"]["data-paths"]["save_inference_path"]

    return parameters

