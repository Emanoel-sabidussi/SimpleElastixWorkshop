from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import rice
from support_files.imageUtilities import get_data_information
import torch

import matplotlib.pyplot as plt


class ModifiedBesselFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, nu):
        ctx._nu = nu
        ctx.save_for_backward(inp)
        return torch.from_numpy(np.i0(inp.detach().numpy()))
    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        nu = ctx._nu
        # formula is from Wikipedia
        return 0.5* grad_out *(ModifiedBesselFn.apply(inp, nu - 1.0)+ModifiedBesselFn.apply(inp, nu + 1.0)), None


class Noise_operator(torch.nn.Module):
    def __init__(self, noise_op):
        super(Noise_operator,self).__init__()
        self.bessel = ModifiedBesselFn.apply
        self.type = noise_op

    def apply_noise(self, image_sequence, estimated_std):
        noisy_output = []

        for image in image_sequence:
            if self.type == 'Gaussian':
                noisy_output.append(self.__apply_gaussian_noise__(image, estimated_std))
            elif self.type == 'Rician':
                noisy_output.append(self.__apply_rician_noise__(image, estimated_std))
        noisy_output = torch.stack(noisy_output)
        return noisy_output

    def __apply_gaussian_noise__(self, image, est_std):
        [img_shape, img_elem, _] = get_data_information(image)
        image_vec = image.view(img_elem)
        noise_vector = np.random.normal(loc=0.0, scale=est_std, size=img_elem)
        if image.is_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        noisy_output = image_vec.double() + torch.tensor(noise_vector).double().to(device=device)
        return noisy_output.view(img_shape)

    def __apply_rician_noise__(self, image, est_std):
        [img_shape, img_elem, _] = get_data_information(image)
        if image.is_cuda:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        image_vec = torch.Tensor(rice.rvs(1,image.double().view(img_elem), est_std)).double().to(device=device)
        return image_vec.view(img_shape)

    def forward(self, estimated_series, observed_series, std_noise):
        if self.type == "Gaussian":
            term_1 = torch.sqrt(torch.Tensor([1/2]))/std_noise
            term_2 = observed_series - estimated_series
            error = torch.sum((term_2)**2)
            return error.double()
            
        if self.type == "Rician":
            bess_arg = (estimated_series * observed_series)/(std_noise**2)
            log_bessel = torch.log(self.bessel(bess_arg, 1))
            term1 = (estimated_series**2)/(2*(std_noise**2))
            rician_term = log_bessel - term1
            return -torch.sum(rician_term).double()
