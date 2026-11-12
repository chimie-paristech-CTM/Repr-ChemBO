import pandas as pd
import json
from pathlib import Path

from rdkit import Chem

import warnings

# data and data config, relative path
FILEPATH_DICT = {
    'additives_1': ["datasets/additives/additive_rxn_screening_plate_1.csv", "datasets/additives/additives.json"],
    'additives_2': ["datasets/additives/additive_rxn_screening_plate_2.csv", "datasets/additives/additives.json"],
    'additives_3': ["datasets/additives/additive_rxn_screening_plate_3.csv", "datasets/additives/additives.json"],
    'additives_4': ["datasets/additives/additive_rxn_screening_plate_4.csv", "datasets/additives/additives.json"],
    
    'bh_1': ["datasets/buchwald-hartwig/bh_reaction_1.csv", "datasets/buchwald-hartwig/bh_sub.json"],
    'bh_2': ["datasets/buchwald-hartwig/bh_reaction_2.csv", "datasets/buchwald-hartwig/bh_sub.json"],
    'bh_3': ["datasets/buchwald-hartwig/bh_reaction_3.csv", "datasets/buchwald-hartwig/bh_sub.json"],
    'bh_4': ["datasets/buchwald-hartwig/bh_reaction_4.csv", "datasets/buchwald-hartwig/bh_sub.json"],
    'bh_5': ["datasets/buchwald-hartwig/bh_reaction_5.csv", "datasets/buchwald-hartwig/bh_sub.json"],
    'bh_merge': ["datasets/buchwald-hartwig/bh_reaction_all.xlsx", "datasets/buchwald-hartwig/bh_all.json"],
    
    'shields': ["datasets/shields/shields_dataset.xlsx", "datasets/shields/shields.json"],
    
    'suzuki-miyaura': ["datasets/suzuki-miyaura/suzuki.csv", "datasets/suzuki-miyaura/suzuki.json"],
    'suzuki-gollum': ["datasets/suzuki-miyaura/suzuki_miyaura_gollum.csv", "datasets/suzuki-miyaura/suzuki_gollum.json"],
    
    'kinase': ["datasets/molecules/enamine10k.csv.gz", "datasets/molecules/kinase.json"],
    'photoswitch': ["datasets/molecules/photoswitches.csv.gz", "datasets/molecules/photoswitch.json"],
    'pce': ["datasets/molecules/photovoltaics_pce10k.csv.gz", "datasets/molecules/pce.json"],
    'redox': ["datasets/molecules/redox_mer_with_iupac.csv.gz", "datasets/molecules/redox.json"],
    'solvation': ["datasets/molecules/redox_mer_with_iupac.csv.gz", "datasets/molecules/solvation.json"],
    
    'oer': ["datasets/numeric/oer_data.csv", "datasets/numeric/oer.json"],
    
    'c2-yield': ["datasets/c2-yield/c2_yield_data.csv", "datasets/c2-yield/c2_yield.json"],
    'hplc': ["datasets/hplc/hplc_data.csv", "datasets/hplc/hplc.json"],
    'vapdiff': ["datasets/vapdiff/vapdiff_data.csv", "datasets/vapdiff/vapdiff.json"],
}

BLANK_SUBSTANCE = "__BLANK__" # e.g., in suzuki-gollum, no ligand is also an option with valid yield.

class DatasetLoader:
    def __init__(self, config):
        self.dataset = config['dataset']['name']
        self.canonical_smiles = config['dataset']['canonical_smiles']
        self.normalize_objective = config['dataset']['normalize_objective'] # None, 0-100, NOTE BayBE uses objective standardization automatically
        self.objective_mode = config['dataset']['objective_mode'] # "MAX" or "MIN"
        
        self.normalize_feature = config['normalize_feature']
        self.normalize_level = config['normalize_level']
        
        self.de_duplicate = config['dataset']['de_duplicate'] # for some datasets, e.g., photoswitch and pce, the same mol has different objectives.
        
        self.fp_level = config['fp_level'] # mol = normal concatenation; rxn = only rxn column; prompt = only pre-filled prompt, e.g., GOLLuM procedure. 
        
        self._validate_config()
        
        self.raw_lookup = self.get_raw_lookup()
        
        self.lookup, self.mol_data, self.numeric_data, self.objective_name = self.process_dataset(self.raw_lookup)
    
    def _validate_config(self):
        
        if not self.dataset in FILEPATH_DICT.keys():
            raise KeyError(f"Not supported dataset: {self.dataset}. Supported: {FILEPATH_DICT.keys()}")
            
        if not self.objective_mode.lower() in ['min', 'max']:
            raise ValueError(f"un-supported objective_mode {self.objective_mode}, should be MAX or MIN.")
        
        if not self.normalize_objective:
            raise ValueError(f"Currently normalizing the objective is obligatory for implementation.")
        
        if not self.fp_level in ['mol', 'rxn', 'prompt']:
            raise ValueError(f"un-supported fp-level. Select from [mol, rxn, prompt]. mol = concatenate each component; rxn = use the single-column reaction representation; prompt = use the **pre-filled** prompting column, e.g., procedure column in GOLLuM. ")
        
        return
    
    @staticmethod
    def _canonical_smiles(smiles):
        if smiles == BLANK_SUBSTANCE:
            return BLANK_SUBSTANCE
        mol = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical
    
    @staticmethod
    def _normalize_1(pd_col):
        '''
        between 0 and 1.
        '''
        pd_col = pd_col - pd_col.min()
        pd_col = pd_col / ( pd_col.max() - pd_col.min()  )
        return pd_col

    @staticmethod
    def _normalize_100(pd_col):
        '''
        Normalize values between 0 and 100: used to transform "yield-like" targets.
        '''
        pd_col = pd_col - pd_col.min()
        pd_col = pd_col / ( pd_col.max() - pd_col.min()  )
        pd_col = pd_col * 100
        return pd_col
    
    @staticmethod
    def _de_duplicate_candidate(lookup:pd.DataFrame, cols_check:list[str]):
        """for some datasets, e.g., photoswitch and pce, the same mol has different objectives."""

        duplicate_groups_mask = lookup.duplicated(
            subset=cols_check,
            keep=False,
        )

        duplicate_groups = lookup.loc[duplicate_groups_mask]

        if not duplicate_groups.empty:
            print("The following candidates are duplicated, while only the first one is kept.")
            print(
                duplicate_groups.sort_values(cols_check).to_string(index=False)
            )

        return lookup.drop_duplicates(
            subset=cols_check,
            keep="first",
        ).reset_index(drop=True)
        
    @staticmethod
    def _is_blank(series: pd.Series) -> pd.Series:
        """Identify cells representing an intentional blank component."""
        text = series.astype("string")
        return series.isna() | text.str.strip().eq("")
    
    def _valid_rxn_prompt(self, molecular_params:dict):
        """The input argument molecular_params actually means reaction smiles or prompted natural language."""
        
        if not self.fp_level in ['rxn', 'prompt']:
            raise Exception('This function is designed for fp_level = rxn or prompt only.')
        
        if not molecular_params:
            raise ValueError(f"Void input dict or Invalid dataset configuration - can't find the required rxn or prompt (procedure) information.")
        
        for param_name, param_val in molecular_params.items():
            if param_val is None:
                raise ValueError(f"Invalid dataset configuration - can't find the required rxn or prompt (procedure) information.")
    
    def get_raw_lookup(self):
        dataset = self.dataset.lower()
            
        file_path = FILEPATH_DICT[dataset][0]
    
        if file_path.endswith('.xlsx'):
            lookup = pd.read_excel(file_path, index_col=None)
        elif file_path.endswith('.csv') or file_path.endswith('.csv.gz'):
            lookup = pd.read_csv(file_path, index_col=None)
        else:
            raise ValueError("Unsupported file format. Must be .xlsx or .csv")
        
        return lookup

    def process_dataset(self, raw_lookup: pd.DataFrame):
        config_path = FILEPATH_DICT[self.dataset][1]
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as file:
            data_config = json.load(file)

        lookup = raw_lookup.copy(deep=True)

        # parameter_name -> smiles_column
        if self.fp_level == "mol":
            molecular_params = data_config.get("molecular_params",{})
            numeric_columns = data_config.get("numeric_params", [])
            
        # NOTE 'molecular_params' actually means reaction smiles
        elif self.fp_level == "rxn":
            molecular_params = data_config.get("rxn_params", {})
            numeric_columns = []
            self._valid_rxn_prompt(molecular_params)
            
        # NOTE 'molecular_params' actually means natural language, prompting
        elif self.fp_level == "prompt":
            molecular_params = data_config.get("prompt_params", {})
            numeric_columns = []
            self._valid_rxn_prompt(molecular_params)
        
        objective_column = data_config["objective"]

        # Create molecule-name columns when absent
        for parameter_name, smiles_column in molecular_params.items():
            
            if parameter_name == smiles_column:
                raise ValueError(f'parameter_name {parameter_name} should be different from smiles_column {smiles_column}')
            
            if parameter_name not in lookup.columns:
                lookup[parameter_name] = lookup[smiles_column].copy()
                
            smiles_blank = self._is_blank(lookup[smiles_column])
            name_blank = self._is_blank(lookup[parameter_name])
            
            if smiles_blank.any():
                print(
                    f"Blank substance detected: "
                    f"parameter={parameter_name!r}, "
                    f"column={smiles_column!r}, "
                    f"rows={int(smiles_blank.sum())}"
                )

            # An absent SMILES means that this component is intentionally blank.
            lookup.loc[smiles_blank, parameter_name] = BLANK_SUBSTANCE
            lookup.loc[smiles_blank, smiles_column] = BLANK_SUBSTANCE

            # A real SMILES without a name is not a blank component.
            # Use the SMILES itself as its candidate name.
            unnamed_real_substance = name_blank & ~smiles_blank
            lookup.loc[unnamed_real_substance, parameter_name] = lookup.loc[
                unnamed_real_substance,
                smiles_column,
            ]

            # Protect against an accidental collision with the reserved name.
            sentinel_collision = (
                lookup[parameter_name].eq(BLANK_SUBSTANCE)
                & ~lookup[smiles_column].eq(BLANK_SUBSTANCE)
            )
            if sentinel_collision.any():
                raise ValueError(
                    f"{BLANK_SUBSTANCE!r} is reserved for blank substances, "
                    f"but it is used as a real value in parameter {parameter_name!r}."
                )

        # Normalize numerical features to [0, 1]. 
        if self.normalize_level.lower()=='fp' and self.normalize_feature.lower() in ['local', 'global']:
            for parameter_name in numeric_columns:
                lookup[parameter_name] = self._normalize_1(lookup[parameter_name])

        # Canonicalize SMILES (only for mol-level)
        if self.canonical_smiles and self.fp_level=='mol':
            for smiles_column in set(molecular_params.values()):
                lookup[smiles_column] = lookup[smiles_column].apply(self._canonical_smiles)
                
        # for some datasets, e.g., photoswitch and pce, the same mol has different objectives.
        # NOTE if canonical_smiles, then duplicated canonical candidates will also be removed.
        if self.de_duplicate:
            candidate_columns = (list(molecular_params.values()) + numeric_columns)
            lookup = self._de_duplicate_candidate(lookup, cols_check=candidate_columns)

        # NOTE Negate objective if minimizing
        if self.objective_mode.lower() == 'min':
            lookup[objective_column] = - lookup[objective_column]
        
        # Normalize objective to [0, 100]
        if self.normalize_objective:
            lookup[objective_column] = self._normalize_100(lookup[objective_column])
        
        self.y_best = lookup[objective_column].max()

        # parameter name -> {molecule name: molecule SMILES}
        mol_data = {
            parameter_name: dict(
                sorted(
                    set(
                        zip(
                            lookup[parameter_name],
                            lookup[smiles_column],
                        )
                    )
                )
            )
            for parameter_name, smiles_column
            in molecular_params.items()
        }

        # parameter name -> available numerical values
        numeric_data = {
            parameter_name: set(lookup[parameter_name])
            for parameter_name in numeric_columns
        }

        return (
            lookup,
            mol_data,
            numeric_data,
            objective_column,
        )

        