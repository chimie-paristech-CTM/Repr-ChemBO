# Ablation Study

## base (no change)

Not for ablation, already tested in standard experiments.

15 benchmarks, of varying sizes, with molecular SMILES.

Chen or no prior.

Local or global normalization.

7 representations: no random representation. No one-hot for PCE and Kinase.

A total of 412 combinations, ignoring 20 random seeds.

## no_ard

Only turn off the ARD.

Lengthscales, outputscale and constant_mean are still learnable.

**To play a GOLLuM game, ARD is essential. ARD is not harmful but needs to be regularized.**

**We expect that global normalization needs ARD because of varying magnitude in each dimension.**

## fix_all

Lengthscales, outputscale and constant_mean are all fixed.

ARD makes no sense in this case.

**To demonstrate that GP is already very flexible, it just needs to be regularized.**

## fix_ls

Only fix lengthscales.

ARD makes no sense in this case.

**To demonstrate that Chen hyperprior is not degenerating to fixed lengthscales, but a calibrated fixed lengthscales is already very good.**

## relax_os

This was already tested on GENCI, where the outputscale prior is removed from Chen hyperprior.

**Only very slight performance drop was observed.**

**Or learnabe constant_mean is enough to explain the data.**
