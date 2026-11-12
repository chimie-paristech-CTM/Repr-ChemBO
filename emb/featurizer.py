from emb.pretrained_repr import PretrainedWrapper, CheMeleonFingerprint, NaCh0_Fingerprint, LLM_Fingerprint, RandomFingerprint, RandomWrapper, OracleFingerprint

from baybe.parameters import SubstanceParameter, CustomDiscreteParameter, CategoricalParameter

from emb.utils import custom_fingerprinter

FINGERPRINT_CLASSES = {
    cls.__name__: cls
    for cls in (
        CheMeleonFingerprint,
        NaCh0_Fingerprint,
        LLM_Fingerprint,
        RandomFingerprint,
        OracleFingerprint,
    )
}


class mol_featurizer:
    def __init__(self, config):
        self.fp_config = config['fp']
        self.fp_type = self.fp_config['name']
        self.decorr = self.fp_config.get('decorrelation', False)
        self.normalize_feature = config['normalize_feature']
        self.normalize_level = config['normalize_level']
        
        self.wrapper_noise = config['wrapper'].get("noise")

    def generate_mol_fp(self, data_dict, **runtime_params):
        fingerprinter = None
        fp_class_name = self.fp_config.get('class')
        if fp_class_name is not None:
            try:
                fp_class = FINGERPRINT_CLASSES[fp_class_name]
            except KeyError:
                supported = ', '.join(FINGERPRINT_CLASSES)
                raise ValueError(
                    f"Unknown fingerprint class {fp_class_name!r}. "
                    f"Supported classes: {supported}"
                ) from None
            
            if self.wrapper_noise is not None:
                fingerprinter = RandomWrapper(fp_class, self.wrapper_noise, **self.fp_config['params'], **runtime_params)
            else:
                fingerprinter = PretrainedWrapper(fp_class, **self.fp_config['params'], **runtime_params)
                
        # Automatically handle whatever data keys are available
        param_dict = {name: self._make_param(name, data, self.decorr, self.fp_type, fingerprinter) for name, data in data_dict.items() if data is not None}
        return param_dict
    
    def _make_param(self, name, data, decorr, fp_type=None, fingerprinter=None):
                
        if data is None:
            return None
        
        NORMALIZE = self.normalize_feature if self.normalize_level.lower()=='fp' else None
        # NORMALIZE = None # 'global' | 'local' | 'no'
        # NOTE In standard workflow, BayBE calls BoTorch to handle the input (local) normalization and outcome standardization automatically. Thus feature normalization is controlled by replacing the GPsurrogate or manipulating the concatenated search space, not here for individual component.
        
        if fp_type.lower() in ['mordred', 'ecfp', 'rdkit2ddescriptors']:
        
            return CustomDiscreteParameter(
                name=name,
                data=custom_fingerprinter(data, fingerprinter, PCA=(decorr=='PCA'), norm=NORMALIZE, substance_fp=fp_type.upper()),
                decorrelate=decorr if decorr!='PCA' else False,
                )
            
        elif fp_type in ['one-hot']: # decorrelation is meaningless
            return CategoricalParameter(name=name, values=data.keys(), encoding="OHE")
        elif fp_type in ['random-search']: # save memory for big dataset
            return CategoricalParameter(name=name, values=data.keys(), encoding="INT")
        else:
            return CustomDiscreteParameter(
                name=name,
                data=custom_fingerprinter(data, fingerprinter, PCA=(decorr=='PCA'), norm=NORMALIZE),
                decorrelate=decorr if decorr!='PCA' else False,
                )
