# Repr-ChemBO

This is the repository for the paper **Which Chemical Priors Should Bayesian Optimization Trust, and How Strongly?**

## How to run

### Installation

For dependencies, we recommend installing via pip (after creating a virtual environment), with Python 3.11: `pip install -r requirements.txt`.

### Cache pre-trained models

Cache pre-trained models **in advance** through `python cache_pretrained_model.py`. A folder named "from_pretrained" will be created.

### Test configs in parallel

With the help of **Slurm**, one can launch many jobs in parallel. Through SBATCH scripts in the folder `scripts/`, configurations in the folder `configs/` are used respectively. In a specific `.json` file, settings annotated with ``"__grid__"`` will be a part of the cartesian product to form a list of jobs.

Note that one needs to adapt the script to his/her own machine.

### Test a certain example

If you just want to try a certain job with your laptop, use for example `python main.py --config config_test.json`to test the configuration described in `config_test.json`.

Of course, one can also test jobs listed in the folder `configs/` but in serial with their laptops: `python main.py --config configs/config_small_1.json`.

### Reproduce

To reproduce the results reported in the main text, see `run.sh`.

## Related repositories

- The basic workflow provided in our previous work: GitHub repo [HSF-ChemBO](https://github.com/chimie-paristech-CTM/HSF-ChemBO).
- A chemical BO tutorial for beginners: GitHub repo [HSF-ChemBO-tutorial](https://github.com/chimie-paristech-CTM/HSF-ChemBO-tutorial).
- An extension to Transfer Learning in chemical BO: GitHub repo [TL-ChemBO](https://github.com/chimie-paristech-CTM/TL-ChemBO).

## References

1. Guanming Chen, Maximilian Fleck, Thijs Stuyver; Leveraging Chemical Hidden-Space Representations Effectively in Bayesian Optimization for Experiment Design through Dimension-Aware Hyperpriors. J. Chem. Theory Comput. 9 June 2026; 22 (11): 5594–5608. https://doi.org/10.1021/acs.jctc.6c00251
2. Guanming Chen, Priyanka Parihar, Maximilian Fleck, Thijs Stuyver; Robust transfer learning for Bayesian optimization of chemical reactions. Digital Discovery 2026; 5 (10): 4121–4134. https://doi.org/10.1039/d6dd00527f
