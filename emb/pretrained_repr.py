import re
import warnings
import torch
import numpy as np
from typing import Type, Union, Literal, Optional, Any
from functools import partial
from tqdm import tqdm
import torch.nn.functional as F
import hashlib
# CheMeleon
from rdkit.Chem import MolFromSmiles, Mol, AllChem
from rdkit import Chem

from utils.context_manager import CACHE_DIR, CACHE_ONLY

class RandomFingerprint:
    '''Baseline: a fixed random vector per molecule (same pipeline as the real
    embeddings, only the model output is replaced by a deterministic, hash-seeded
    random vector). Deterministic per SMILES so it caches/reproduces like the rest.'''
    def __init__(self, dim=768, mean=None, std=None, low=None, high=None):
        """We do not set random seed here cause the global seed is set in BORunner() before featurization."""
        self.dim = dim
        if (not mean is None) and (not std is None):
            self.mean = mean
            self.std = std
            self.mode = "normal"
        if (not low is None) and (not high is None):
            self.low = low
            self.high = high
            self.mode = "uniform"
        if (not mean is None) and (not std is None) and (not low is None) and (not high is None):
            raise ValueError("ambiguous setting for Random representation.")
        
        # based on the global seed set before:
        self._base_seed = int(np.random.randint(0, 2**32, dtype=np.uint32))

    def __call__(self, smiles_list) -> np.ndarray:
        out = np.empty((len(smiles_list), self.dim), dtype=np.float32)

        for i, smiles in enumerate(smiles_list):
            smiles_hash = int.from_bytes(
                hashlib.md5(str(smiles).encode()).digest()[:4],
                byteorder="little",
            )
            rng = np.random.default_rng(smiles_hash ^ self._base_seed)
            if self.mode == 'normal':
                out[i] = rng.normal(loc=self.mean, scale=self.std, size=self.dim)
            elif self.mode == 'uniform':
                out[i] = rng.uniform(low=self.low, high=self.high, size=self.dim)
            else:
                pass

        return out
    
def construct_oracle_map(mol_data, lookup, objective_name):
    param_name = next(iter(mol_data))
    name_to_smiles = mol_data[param_name]

    label_counts = (
        lookup
        .groupby(param_name)[objective_name]
        .nunique()
    )

    ambiguous = label_counts[label_counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            "Some candidates have multiple objective values; "
            "cannot construct a molecule-level oracle fingerprint."
        )

    name_to_oracle = (
        lookup[[param_name, objective_name]]
        .drop_duplicates(subset=[param_name])
        .set_index(param_name)[objective_name]
    )

    oracle_map = {
        smiles: float(name_to_oracle.loc[name])
        for name, smiles in name_to_smiles.items()
    }
    
    return oracle_map
    
class OracleFingerprint(RandomFingerprint):
    def __init__(self, dim=0, mean=None, std=None, low=None, high=None, oracle_map=None):
        super().__init__(dim, mean, std, low, high)

        if oracle_map is None:
            raise ValueError("oracle_map must be provided.")

        self.oracle_map = dict(oracle_map)

    def _get_oracle(self, smiles_list):
        missing = [s for s in smiles_list if s not in self.oracle_map]
        if missing:
            raise KeyError(f"Missing oracle values for inputs: {missing[:5]}")

        return np.asarray(
            [self.oracle_map[s] for s in smiles_list],
            dtype=np.float32,
        ).reshape(-1, 1)

    def __call__(self, smiles_list):
        oracle = self._get_oracle(smiles_list)

        if self.dim == 0:
            return oracle

        noise = super().__call__(smiles_list)
        return np.concatenate([oracle, noise], axis=1)

class CheMeleonFingerprint:
    def __init__(self, device: str | torch.device | None = None):
        from chemprop import featurizers, nn
        from chemprop.nn import RegressionFFN
        from chemprop.models import MPNN
        from chemprop.data import BatchMolGraph
        
        self._batch_mol_graph_cls = BatchMolGraph
        
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        mp_path = CACHE_DIR+"/chemeleon_mp.pt" # read cache only
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),  # not actually used
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        bmg = self._batch_mol_graph_cls([self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m) for m in molecules])
        bmg.to(device=self.model.device)
        with torch.no_grad():
            return self.model.fingerprint(bmg).numpy(force=True)
    
class LLM_Fingerprint:
    """Generic LLM fingerprinter supporting multiple encoder models loaded from local cache. Pooling strategies: 'average', 'cls', 'last_token_pool', 'weighted_average'.
    """
    def __init__(self, model_name, pooling_method='average', normalize_embeddings = False, device='cpu', prefix=None, layer_index=-1):
        
        from emb.llm_utils import get_model_and_tokenizer, average_pool, last_token_pool, weighted_average_pool

        self.model_name = model_name
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.layer_index = layer_index
        self.model, self.tokenizer = get_model_and_tokenizer(model_name, device)
        left_padding = self.tokenizer.padding_side == "left"
        self.model.eval()

        self.prefix = prefix
        
        self.pooling_functions = {
            "average": average_pool,
            "cls": lambda x, _: x[:, 0],
            "last_token_pool": partial(last_token_pool, left_padding=left_padding),
            "weighted_average": weighted_average_pool,
        }
    
    def __call__(self, texts) -> np.ndarray:
        """Return pooled embeddings of shape (n_texts, hidden_size)."""
        # optionally add prefix to each text
        if self.prefix:
            texts = [self.prefix + text for text in texts]
        
        batch_size = 8
        max_length = 512
        #max_length = 1000000
        
        embeddings_list = []
        for i in tqdm(
            range(0, len(texts), batch_size), desc=f"Processing with {self.model_name}", disable=True,
        ):
            batch_texts = texts[i : i + batch_size]

            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded_input, output_hidden_states=True)
                # NOTE HuggingFace: When passing output_hidden_states=True you may expect the outputs.hidden_states[-1] to match outputs.last_hidden_state exactly. However, this is not always the case. Some models apply normalization or subsequent process to the last hidden state when it’s returned.
                n_layers = len(outputs.hidden_states)
                if self.layer_index >= n_layers:
                    warnings.warn(
                        f"layer_index={self.layer_index} out of range (0-{n_layers-1}); "
                        "falling back to last layer"
                    )
                    hidden = outputs.hidden_states[-1]
                else:
                    hidden = outputs.hidden_states[self.layer_index]
                pooled = self.pooling_functions[self.pooling_method](
                    hidden, encoded_input["attention_mask"]
                )

                if self.normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)
                embeddings_list.append(pooled.cpu().numpy())

            torch.cuda.empty_cache()

        return np.concatenate(embeddings_list, axis=0)

class NaCh0_Fingerprint(LLM_Fingerprint):
    '''
    model: insilicomedicine/nach0_base (T5 encoder only)
    size: 768

    Supports two input modes transparently via _add_special_symbols():
    - SMILES-only: pass raw SMILES strings
    - Prompt mode: pass natural language text with SMILES tokens embedded inline
    '''
    _ATOMS = sorted(
        ['Ag','Al','As','Au','B','Ba','Bi','Br','C','Ca','Cd','Cl','Co','Cr',
         'Cs','Cu','F','Fe','Ga','Gd','Ge','H','Hg','I','In','K','Li','M',
         'Mg','Mn','Mo','N','Na','O','P','Pt','Ru','S','Sb','Sc','Se','Si',
         'Sn','V','W','Z','Zn','c','e','n','o','p','s',
         'Ir', # nach0 not trained on Ir, just for the tokenizer.
         ],
        key=len, reverse=True,
    )

    _PATTERN = re.compile(
        r"(\[|\]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9]|"
        + "|".join(_ATOMS) + ")"
    )

    def __init__(self, model_name, pooling_method='average', normalize_embeddings=None, device='cpu', prefix=None, layer_index=-1):
        super().__init__(model_name, pooling_method, normalize_embeddings, device, prefix, layer_index)
        
        self.full_model = self.model
        self.model = self.model.encoder

    def xxx_add_special_symbols(self, text: str) -> str:
        """Convert SMILES tokens in text to nach0's `<sm_X>` atom-level format.

        OUTDATED, COULD BE USED IN FUTURE... lol

        Handles two input forms transparently:
        - Raw SMILES string: tokenized directly.
        - Prompt with `<mol>[SMILES]</mol>` tags: SMILES inside tags are tokenized; prose is kept as-is.
        """
        # print("input to nach0:", text)
        def _tokenize_smiles(smiles: str) -> str:
            tokens = self._PATTERN.findall(smiles)
            return ''.join(f'<sm_{t}>' for t in tokens)

        def _replace_mol_tag(m):
            inner = m.group(1)
            # Notebook writes <mol>[smiles]</mol> — strip the outer [ ]
            candidate = inner[1:-1] if (inner.startswith('[') and inner.endswith(']')) else inner
            if MolFromSmiles(candidate):
                return _tokenize_smiles(candidate)
            if MolFromSmiles(inner):
                return _tokenize_smiles(inner)
            return m.group(0)

        text = re.sub(r'<mol>(.*?)</mol>', _replace_mol_tag, text)
        # print("input to nach0:", text)
        
        output = []
        for word in text.split():
            tokens = self._PATTERN.findall(word)
            if len(tokens) > 4 and word == ''.join(tokens) and MolFromSmiles(word):
                output.append(_tokenize_smiles(word))
            else:
                output.append(word)
        ret = ' '.join(output)
        # use this for testing. Compare to documentation
        # it should look weird :)
        # print("tokenized", ret)
        return ret

    # TODO, when used arbitrarily, SMILES parse error for input natural language or rxn smiles 
    def _add_special_symbols(self, text: str) -> str:
        """Convert SMILES tokens in text to nach0's `<sm_X>` atom-level format.

        SMILES are auto-detected.
        This is basically copied from the nach0 README
        """
        # print("input to nach0:", text)
        output = []
        for word in text.split():
            tokens = [token for token in self._PATTERN.findall(word)]
            if len(tokens) > 4 and (word == ''.join(tokens)) and MolFromSmiles(word):
                output.append(''.join(['<sm_'+t+'>' for t in tokens]))
            else:
                output.append(word)
        tokenized = ' '.join(output)
        # print("tokenized:", tokenized)
        return tokenized

    def _clean_output_sequence(self, output_sequence):
        return output_sequence.replace('</s>', '').replace('<sm_', '').replace(' sm_', '').replace('>', '').strip()

    # TODO consider re-factor this based on LLM_Fingerprint
    def __call__(self, texts, return_text=False, max_new_tokens=256):
        """Return encoder embeddings for a list of SMILES or prompts.

        If return_text=True, also runs the decoder and returns (embeddings, generated_texts).
        """
        preprocessed = [self._add_special_symbols(t) for t in texts]

        batch_size = 8
        embeddings_list = []
        generated_texts = []
        for i in tqdm(range(0, len(preprocessed), batch_size),
                    desc='Processing with nach0_base', disable=True):
            batch = preprocessed[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors='pt',
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=(self.layer_index != -1))
                if self.layer_index != -1:
                    n_layers = len(outputs.hidden_states)
                    if self.layer_index >= n_layers:
                        warnings.warn(
                            f"layer_index={self.layer_index} out of range (0-{n_layers-1}); "
                            "falling back to last_hidden_state"
                        )
                        hidden = outputs.last_hidden_state
                    else:
                        hidden = outputs.hidden_states[self.layer_index]
                else:
                    hidden = outputs.last_hidden_state
                pooled = self.pooling_functions[self.pooling_method](
                    hidden, encoded['attention_mask']
                )
                embeddings_list.append(pooled.cpu().numpy())
                if return_text:
                    token_ids = self.full_model.generate(
                        **encoded, max_new_tokens=max_new_tokens
                    )
                    generated_texts.extend(
                        self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
                    )
            torch.cuda.empty_cache()

        embeddings = np.concatenate(embeddings_list, axis=0)
        if return_text:
            return embeddings, self._clean_output_sequence(generated_texts)
        return embeddings
    
class PretrainedWrapper:
    """
    Ensures a fingerprinter is created and called in float32 mode (BayBE internally uses float64).

    Also handles invalid SMILES gracefully: None / empty / 'nan' entries are skipped during
    inference and replaced with zero vectors in the output array.
    
    NOTE: Could be problematic when multi-processing asynchronously.
    """
    def __init__(self, 
                pretrained_fingerprinter, 
                **kwargs: Any,
                ):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32) 
            
            self._fingerprinter = pretrained_fingerprinter(**kwargs)
            
        finally:
            torch.set_default_dtype(original_dtype)
            
    def _postprocess_embeddings(self, embeddings):
        """to be overriden"""
        return embeddings
    
    def __call__(self, smiles_list):
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            if isinstance(smiles_list, str):
                smiles_list = [smiles_list]

            _invalid = {None, "", "None", "nan"}
            valid_mask = [s not in _invalid for s in smiles_list]
            valid_smiles = [s for s, ok in zip(smiles_list, valid_mask) if ok]

            if valid_smiles:
                valid_embeddings = self._fingerprinter(valid_smiles)
                valid_embeddings = self._postprocess_embeddings(valid_embeddings)
                emb_dim = valid_embeddings.shape[1]
            else:
                emb_dim = self._fingerprinter(["C"]).shape[1]
                valid_embeddings = np.empty((0, emb_dim), dtype=np.float32)

            result = np.zeros((len(smiles_list), emb_dim), dtype=np.float32)
            vi = 0
            for i, ok in enumerate(valid_mask):
                if ok:
                    result[i] = valid_embeddings[vi]
                    vi += 1
            return result
        finally:
            torch.set_default_dtype(original_dtype)

class RandomWrapper(PretrainedWrapper):
    """Add some noise to the pretrained representations, based on PretrainedWrapper"""
    def __init__(self, pretrained_fingerprinter, noise_config:dict, **kwargs):
        super().__init__(pretrained_fingerprinter, **kwargs)
        
        self.noise_mean = noise_config['mean']
        self.noise_std = noise_config['std']
        
    def _postprocess_embeddings(self, embeddings:np.ndarray):
        """Add noise. Make sure to fix random seed before this calling"""
        noise = np.random.normal(self.noise_mean, self.noise_std, size=embeddings.shape)
        return embeddings + noise
    