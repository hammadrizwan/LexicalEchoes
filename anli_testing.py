from ANLI_data import ANLIDataset
from sentence_transformers import SentenceTransformer
import torch


# Make dataset
dataset = ANLIDataset([sample], tokenizer, label2id)

# DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through loader
def qwen(device,sub_file="replace_rel"):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B").to(device)
    # queries = [
    # "Angola is located in",
    # ]
    # documents = [
    #     "Mozambique is in",
    #     "Angola is a part of the continent of",
    # ]
    # sub_file="swap_att"
    file_path_scpp="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/scpp/data/"+sub_file+".json"
    file_save_path="./scpp_analysis/scpp_qwen_lexical_bias_direct_"+sub_file+".jsonl"
    qwen_test_scpp(file_path=file_path_scpp,model=model,file_save_path=file_save_path,device="cuda:0")
        
    del model           # Remove the Python object reference
    gc.collect()        # Run garbage collection to clean up CPU memory

    # Free GPU memory if model was on CUDA
    torch.cuda.empty_cache()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for file in ["swap_att","swap_obj","replace_att","replace_obj","replace_rel"]:
        print("Running KaLM for ", file)
        kalm(device,sub_file=file)
