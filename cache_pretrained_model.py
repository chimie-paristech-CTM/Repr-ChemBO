def cache_pretrained_model():
    from utils.context_manager import CACHE_DIR
    import os
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # LLM
    from emb.llm_utils import get_model_and_tokenizer
    get_model_and_tokenizer("t5-base", local_files_only=False)
    get_model_and_tokenizer("GT4SD/multitask-text-and-chemistry-t5-base-augm", local_files_only=False)
    get_model_and_tokenizer("DeepChem/ChemBERTa-100M-MLM", local_files_only=False)
    get_model_and_tokenizer("insilicomedicine/nach0_base", local_files_only=False)
    
    #CheMeleon (Chemprop, GNN)
    from urllib.request import urlretrieve
    chemeleon_path = CACHE_DIR+"/chemeleon_mp.pt"
    if not os.path.exists(chemeleon_path):
        print("Downloading CheMeleon checkpoint...")
        urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", chemeleon_path)

if __name__ == "__main__":
    cache_pretrained_model()