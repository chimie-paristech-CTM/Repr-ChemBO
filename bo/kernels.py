from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory
# BayBE kernels (NOT GPyTorch!)
# BE CAREFUL !!
# Kernels share names in gpytorch and baybe BUT are not the same!!
from baybe.kernels import ScaleKernel, MaternKernel, RBFKernel
from baybe.priors.basic import GammaPrior, LogNormalPrior, SmoothedBoxPrior
# from gpytorch.priors import UniformPrior
import math
import numpy as np
import warnings

class MaternKernelFactory(KernelFactory):
    """
    Normalize each parameter group to have similar influence.
    Simpler than full block ARD, but effective.
    """

    def __init__(self, prior_set="max_custom_0", n_dim=None, kernel_name_user='Matern'):
        self.prior_set = prior_set
        self.n_dim = n_dim
        self.kernel_name_user = kernel_name_user
    
    def __call__(self, searchspace, train_x, train_y):  

        if self.prior_set == "EDBO+" or self.prior_set == "EDBO_QM":
            # EDBO+ ini == QM features prior from EDBO
            lengthscale_prior = GammaPrior(concentration=2.0, rate=0.2)
            lengthscale_initial_value = 5.0
            outputscale_prior = GammaPrior(concentration=5.0, rate=0.5)
            outputscale_initial_value = 8.0                

        # EDBO
        #   References:
        # * https://github.com/b-shields/edbo/blob/master/edbo/bro.py#L664
        # * https://doi.org/10.1038/s41586-021-03213-y
        # * https://emdgroup.github.io/baybe/stable/_modules/baybe/surrogates/gaussian_process/presets/edbo.html#EDBOKernelFactory
        
        elif self.prior_set == "BayBE_adaptive":
            # BayBE default factory
            if self.n_dim is None:
                raise ValueError("n_dim must be given for adaptive prior!")
            
            _DIM_LIMITS = (8, 75)
            lengthscale_prior = GammaPrior(
                np.interp(self.n_dim, _DIM_LIMITS, [1.2, 2.5]),
                np.interp(self.n_dim, _DIM_LIMITS, [1.1, 0.55]),
            )
            lengthscale_initial_value = np.interp(self.n_dim, _DIM_LIMITS, [0.2, 6.0])
            outputscale_prior = GammaPrior(
                np.interp(self.n_dim, _DIM_LIMITS, [5.0, 3.5]),
                np.interp(self.n_dim, _DIM_LIMITS, [0.5, 0.15]),
            )
            outputscale_initial_value = np.interp(self.n_dim, _DIM_LIMITS, [8.0, 15.0])
            
        elif self.prior_set in ["Chen"]:
            if self.n_dim is None:
                raise ValueError("n_dim must be given for adaptive prior!")
            
            x = math.sqrt(self.n_dim)
            l_mean = 0.4 * x + 4.0 # decided by fitting the result points.
            
            lengthscale_prior = GammaPrior(2.0*l_mean, 2.0)
            lengthscale_initial_value = l_mean
            outputscale_prior = GammaPrior(1.0*l_mean, 1.0) # can use a smaller rate for larger variance.
            outputscale_initial_value = l_mean
        
        # NOTE
        # In Gammaprior(conc, rate), mean = conc/rate, var = conc/rate^2
        # Outputscale should not vary with lengthscale/dimension... But when dim increases, there's more scarcity, then larger outputscale may be needed to explain the increasing uncertainty. But since we are not sure about this "law", we use a small rate in Gammaprior to enable large variance.
        
        # NOTE use a wide uniform prior to approximate no MAP / no prior
        elif self.prior_set in ['no']:
            # warnings.warn(f"The input prior name is {self.prior_set}. A SmoothedBoxPrior is used to approximate Uniform(0.01, 100), which actually means no prior. Note that initial lengthscale value is 1.0")
            lengthscale_prior = SmoothedBoxPrior(a=0.01, b=100, sigma=0.01)
            lengthscale_initial_value = 1.0
            outputscale_prior = SmoothedBoxPrior(a=0.01, b=100, sigma=0.01)
            outputscale_initial_value = 1.0
        else:
            raise ValueError(f"No-supported hyperprior {self.prior_set}.")


        if self.kernel_name_user in ['Matern']:
            return ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    lengthscale_prior=lengthscale_prior,
                    lengthscale_initial_value=lengthscale_initial_value,
                ),
                outputscale_prior=outputscale_prior,
                outputscale_initial_value=outputscale_initial_value,
            )
        elif self.kernel_name_user in ['RBF']:
            return ScaleKernel(
                RBFKernel(
                    lengthscale_prior=lengthscale_prior, lengthscale_initial_value=lengthscale_initial_value
                ),
                outputscale_prior=outputscale_prior,
                outputscale_initial_value=outputscale_initial_value,
            )
        else:
            raise ValueError(f"Un-supported kernel {self.kernel_name_user}.")
