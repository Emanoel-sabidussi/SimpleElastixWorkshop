from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import elasticdeform
import numpy as np

class ApplyDeformation(object):
    def __init__(self, ll_p, hl_p, ll_s, hl_s, order):
        self.low_limit_points = ll_p
        self.high_limit_points = hl_p
        self.low_limit_sigma = ll_s
        self.high_limit_sigma = hl_s
        self.order = order

    def __call__(self, weighted_data, get_sequence):
        
        points_random = self.high_limit_points
        sigma_random = self.high_limit_sigma
        size_w_data = np.shape(weighted_data)[0]
        
        if get_sequence: 
            data = [w_data for w_data in weighted_data]
        else:
            data = weighted_data

        deformed_data = elasticdeform.deform_random_grid(data,
                                                        sigma=sigma_random, 
                                                        points=points_random, 
                                                        order=self.order,
                                                        axis=(0,1)
                                                        )
        
        deformed_weighted_data = np.asarray(deformed_data[:size_w_data])

        return deformed_weighted_data


def nonRigidDeform(data, opts, order, apply_seq):
    deformOp = ApplyDeformation(opts[0], opts[1], opts[2], opts[3], order)
    deformData = deformOp(data, apply_seq)
    
    return deformData


def applyDeform(data):
    print("Applying deformations")
    deformData = []
    deformData.append(data[0])
    for ind, data in enumerate(data[1:]):
        deformData.append(nonRigidDeform(data, [1, 1, 1, 2], 2, False))
        
    print("Deformation finished")
    return np.array(deformData)