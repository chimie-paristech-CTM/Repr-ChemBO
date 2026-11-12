from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate, _ModelContext
from baybe.surrogates.gaussian_process.presets.default import _default_noise_factory
from typing_extensions import override

class OriginGP(GaussianProcessSurrogate):
    """Nothing but the original GaussianProcessSurrogate, just convenient for re-writting: ARD, Fix lengthscales/outputscale, Normalization."""
    
    ARD = True
    FIX_LS = False
    FIX_OS = False
    FIX_MEAN = False
    NORM = 'local'
    
    @override
    def _fit(self, train_x, train_y) -> None:
        import botorch
        import gpytorch
        import torch

        assert self._searchspace is not None

        context = _ModelContext(self._searchspace)

        numerical_idxs = context.get_numerical_indices(train_x.shape[-1])
        
        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        if self.NORM == 'local':
            input_transform = botorch.models.transforms.Normalize(
                train_x.shape[-1], bounds=context.parameter_bounds, indices=numerical_idxs
            )
        elif self.NORM == 'global':
            bounds = context.parameter_bounds
            global_min = bounds[0].min()
            global_max = bounds[1].max()
            global_bounds = torch.stack([
                global_min.expand(bounds.shape[-1]),
                global_max.expand(bounds.shape[-1]),
            ])
            # bounds: for local norm, it's per-feature max/min; for global norm, we use global max/min explicitly.
            # indices: to exclude task parameter when transfer learning.
            input_transform = botorch.models.transforms.Normalize(train_x.shape[-1], bounds=global_bounds, indices=numerical_idxs)
        elif self.NORM == 'no':
            input_transform = None
        else:
            raise ValueError(f'un-supported normalization {self.NORM}.')
        
        outcome_transform = botorch.models.transforms.Standardize(train_y.shape[-1])
        
        # extract the batch shape of the training data
        batch_shape = train_x.shape[:-2]

        # create GP mean
        mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        
        if self.FIX_MEAN:
            mean_module.constant.requires_grad_(False)

        # define the covariance module for the numeric dimensions
        base_covar_module = self.kernel_factory(
            context.searchspace, train_x, train_y
        ).to_gpytorch(
            ard_num_dims=(train_x.shape[-1] - context.n_task_dimensions) if self.ARD else None,
            active_dims=numerical_idxs,
            batch_shape=batch_shape,
        )
        
        if self.FIX_LS:
            base_covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        if self.FIX_OS:
            base_covar_module.raw_outputscale.requires_grad_(False)

        # create GP covariance
        if not context.is_multitask:
            covar_module = base_covar_module
        else:
            task_covar_module = gpytorch.kernels.IndexKernel(
                num_tasks=context.n_tasks,
                active_dims=context.task_idx,
                rank=context.n_tasks, 
            )
            covar_module = base_covar_module * task_covar_module

        # create GP likelihood
        noise_prior = _default_noise_factory(context.searchspace, train_x, train_y)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior[0].to_gpytorch(), batch_shape=batch_shape
        )
        likelihood.noise = torch.tensor([noise_prior[1]])

        # construct and fit the Gaussian process
        self._model = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
        )

        if context.n_task_dimensions > 0:
            mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(
                self._model.likelihood, self._model
            )
        else:
            mll = gpytorch.ExactMarginalLogLikelihood(
                self._model.likelihood, self._model
            )

        botorch.fit.fit_gpytorch_mll(mll)
