import pandas as pd

from emb.dataset import DatasetLoader
from emb.featurizer import mol_featurizer
from emb.pretrained_repr import construct_oracle_map

from baybe.searchspace import SearchSpace
from baybe.objectives import SingleTargetObjective
from baybe.targets import NumericalTarget
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from bo.gp import OriginGP
from baybe import Campaign
from baybe.recommenders import BotorchRecommender, RandomRecommender, TwoPhaseMetaRecommender, SequentialMetaRecommender
from bo.recommender import LowerMedianRandomRecommender
from bo.kernels import MaternKernelFactory
from baybe.utils.random import set_random_seed

from utils.context_manager import ContextManager

from bo.scoring import Scoring

from baybe.simulation import simulate_scenarios

class BOrunner:
    def __init__(self, config, context_manager=None):
        self.config = config
        # Generate all the names and paths needed in a run
        self.context_manager = context_manager or ContextManager(config)
        self.context_manager.prepare_output_dir()
        # self.context_manager.prepare_embedding_dir()
        # Load dataset
        datasetloader = DatasetLoader(config)
        self.lookup = datasetloader.lookup
        self.mol_data = datasetloader.mol_data
        self.numeric_data = datasetloader.numeric_data
        self.objective_name = datasetloader.objective_name
        self.y_best = datasetloader.y_best
        # Context for this BOrunner
        self.n_iter = config['bo']['n_iter']
        self.switch_after = config['bo']['switch_after']
        self.batch_size = config['bo']['batch_size']
        self.mc_runs = config['bo']['mc_runs']
        self.impute_mode = config['bo']['impute_mode']
        # Controlled components in BO
        self.hyperprior = config['hyperprior']
        self.acq_func = config['acq_func']
        self.init_method = config['init_method']
        self.kernel = config['kernel']
        # Random seed
        self.seed = config['seed']
        # Embedding: (random seed may be needed in some cases)
        self.fp = config['fp']
        self.decorrelation = self.fp.get('decorrelation', False)
        self.normalize_feature = config['normalize_feature']
        self.normalize_level = config['normalize_level']
        
        self.random_search = bool(self.fp['name'] == 'random-search')
        
        self.ard = config.get('ard', True) # No ARD assigned, turn on by default
        self.fix_ls = config.get('fix_ls', False) # fix lengthscales, False as default
        self.fix_os = config.get('fix_os', False) # fix outputscale, False as default
        self.fix_mean = config.get('fix_mean', False) # fix constant_mean, False as default
                
        # Fix global randomness before RandomFingerprinter or RandomWrapper
        set_random_seed(self.seed)
        
        # Mol Featurizer: temp version, all in the pipeline, no pre_computation
        
        # For oracle test only: indicate "oracle_test" explicitly in the config.json
        if config.get('oracle_test', False):
            self.oracle_params = {"oracle_map": construct_oracle_map(self.mol_data, self.lookup, self.objective_name)}
        else:
            self.oracle_params = {}
        
        self.featurizer = mol_featurizer(config)
        # Prepare for run
        self.pre_run()
        self.campaign_name = self.context_manager.campaign_name
        
        self.scoring = Scoring(config)
        
    def pre_run(self):
        self.numeric_params, self.objective = self.get_num_params_and_obj(self.numeric_data, self.objective_name)
                
        self.mol_params = self.featurizer.generate_mol_fp(self.mol_data, **self.oracle_params)
        
        self.searchspace:SearchSpace = self._create_search_space(self.mol_params | self.numeric_params) # NOTE params order --> randomness
        
        self.f_dim = self._get_f_dim(self.searchspace)
        
        self.recommender = self._create_recommender(self.hyperprior, self.f_dim, self.kernel, self.switch_after, self.acq_func, self.init_method)
        
        self.campaign = Campaign(
            searchspace=self.searchspace,
            objective=self.objective,
            recommender=self.recommender,
        )
        
    def run(self):
        
        result = simulate_scenarios(
            {self.campaign_name: self.campaign},
            self.lookup,
            initial_data=None,
            batch_size=self.batch_size,
            n_doe_iterations=self.n_iter,
            n_mc_iterations=self.mc_runs,
            impute_mode=self.impute_mode
        ) # NOTE do not set a random_seed here into simulate_scenarios() because it will fall back to the _DEFAULT_SEED in BayBE 0.12.2 :) This is fixed in later versions. Set the global seed above instead.
        
        result.to_pickle(self.context_manager.result_path)
        
        scores = self.scoring.get_score(result, self.lookup, self.objective_name, self.y_best)
        
        infos = {
            "timestamp": self.context_manager.timestamp,
            "run_time_hms": self.context_manager.get_elapsed_time_hms(),
            "f_dim": self.f_dim,
        }

        self.context_manager.save_config(
            self.config | scores | infos,
            self.context_manager.run_config_path,
        )
        
        return result
    
    @staticmethod
    def get_num_params_and_obj(numeric_data=None, objective_name=None):
        
        objective = SingleTargetObjective(target=NumericalTarget(name=objective_name, mode="MAX")) # NOTE hardcoded maximization. When minimizing, objective is negated before all.
        
        numeric_params = {}
        for parameter_name, parameter_vals in numeric_data.items():
            numeric_params[parameter_name] = NumericalDiscreteParameter(name=parameter_name, values=parameter_vals)
            
        return numeric_params, objective
        
    @staticmethod
    def _create_search_space(params_dict, task_param=None):
        
        search_params_list = []
        
        for name, parameter in params_dict.items():
            search_params_list.append(parameter)
            
        if task_param is not None:
            search_params_list.append(task_param)
        
        searchspace = SearchSpace.from_product(parameters=search_params_list)
        
        return searchspace
    
    @staticmethod
    def _get_f_dim(searchspace:SearchSpace):
        return len(searchspace.comp_rep_columns) - len([p for p in searchspace.parameters if isinstance(p, TaskParameter)])
        
    def _create_recommender(self, hyperprior:str, f_dim:int, kernel:str, switch_after:int, acq_func:str, init_method:str):
        
        GP_class = OriginGP
        GP_class.ARD = self.ard
        GP_class.FIX_LS = self.fix_ls
        GP_class.FIX_OS = self.fix_os
        GP_class.FIX_MEAN = self.fix_mean
        
        if self.normalize_level.lower() == 'gp':
            GP_class.NORM = self.normalize_feature.lower()
        elif self.normalize_level.lower() == 'fp':
            GP_class.NORM = 'no'
        else:
            raise ValueError(f"un-supported normalize_level {self.normalize_level}, select from [gp, fp].")
        
        custom_surrogate = GP_class(
            kernel_or_factory= MaternKernelFactory(
                prior_set=hyperprior,
                n_dim=f_dim,
                kernel_name_user=kernel,
            )
        )
            
        if init_method == 'random':
            init_recom = RandomRecommender()
        elif init_method == 'random_low':
            init_recom = LowerMedianRandomRecommender(lookup=self.lookup.copy(deep=True), target_name=self.objective_name)
        else:
            raise ValueError(f"un-supported init_method {init_method}")
        
        if self.random_search:
            if switch_after > 0:
                recommender=TwoPhaseMetaRecommender(
                        initial_recommender=init_recom,
                        recommender=RandomRecommender(),
                        switch_after=switch_after # number of experiments required.
                    )
            else:
                recommender=RandomRecommender()
            
            return recommender
            
        if switch_after > 0:
            recommender=TwoPhaseMetaRecommender(
                    initial_recommender=init_recom,
                    recommender=BotorchRecommender(
                        surrogate_model=custom_surrogate,
                        acquisition_function=acq_func,
                    ),
                    switch_after=switch_after # number of experiments required.
                )
        else:
            recommender=BotorchRecommender(
                surrogate_model=custom_surrogate,
                acquisition_function=acq_func,
                )
        
        return recommender
