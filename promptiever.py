import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
from quoraPD_analysis.distance_analysis import analyze_and_save_distances
from tqdm import tqdm
import json

class Promptriever:
    def __init__(self, model_name_or_path):
        self.model, self.tokenizer = self.get_model(model_name_or_path)
        self.model.eval().cuda()

    def get_model(self, peft_model_name):
        # Load the PEFT configuration to get the base model name
        peft_config = PeftConfig.from_pretrained(peft_model_name)
        base_model_name = peft_config.base_model_name_or_path

        # Load the base model and tokenizer
        base_model = AutoModel.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Load and merge the PEFT model
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()

        # can be much longer, but for the example 512 is enough
        model.config.max_length = 512
        tokenizer.model_max_length = 512

        return model, tokenizer

    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(self, sentences, max_length: int = 2048, batch_size: int = 4):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]

            batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
            batch_dict = {
                key: value.to(self.model.device) for key, value in batch_dict.items()
            }

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state
                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_size = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
class PAWSDataset(Dataset):
    def __init__(
        self,
        tsv_path: str,
        s1_key: str = "sentence1",
        s2_key: str = "sentence2",
        label_key: str = "label",
    ):
        self.sentence1_list=[]
        self.sentence2_list=[]
        self.labels_list=[]
        with open(tsv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                self.sentence1_list.append(_parse_field(r[s1_key]))
                self.sentence2_list.append(_parse_field(r[s2_key]))
                self.labels_list.append(int(r[label_key]))
                

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        # returns (sentence1:str, sentence2:str, label:int)
        return self.sentence1_list[idx], self.sentence2_list[idx], self.labels_list[idx]
def collate_text_pairs(batch):
    # batch is a list of samples; each sample is (s1:str, s2:str, y:int)
    s1, s2, y = zip(*batch)  # tuples of length B
    if len(batch) == 1:
        # flatten for convenience when B=1
        return s1[0], s2[0], torch.tensor([y[0]], dtype=torch.long)
    else:
        return list(s1), list(s2), torch.tensor(y, dtype=torch.long)
def _parse_field(s: str) -> str:
    """
    Turn values like b'Will a message ... ?' into a normal str.
    If it's already a plain string, just return it.
    """
    s = s.strip()
    # print(ast.literal_eval(s).decode("utf-8"))
    try:
        lit = ast.literal_eval(s)
        if isinstance(lit, (bytes, bytearray)):
            return lit.decode("utf-8", errors="replace")
        
        return str(lit)
    except Exception:
        # Fallback: remove a leading b and surrounding quotes if present
        if s.startswith(("b'", 'b"')):
            s = s[2:]
        if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
            s = s[1:-1]
        return s
data_path="/home/hrk21/projects/def-hsajjad/hrk21/LexicalBias/data/QuoraPD/paws_out/train.tsv"
dataset=PAWSDataset(data_path)
data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_text_pairs,  # keep strings as lists; labels as tensor
    # num_workers=0,
)

# Initialize the model
model = Promptriever("samaya-ai/promptriever-llama3.1-8b-instruct-v1")
# add specific relevance conditions if desired (and/or/not) and any other prompts
# instruction = "A relevant document would be semantically equivilant. Think carefully about these conditions when determining relevance."

# Combine query and instruction with **two spaces** after "query: "

incorrect_pairs=[]
correct_pairs=[]
instruction = "A relevant document would express the same meaning and intent as the query without adding, removing, or altering information. Think carefully about these conditions when determining relevance."
with open("./quoraPD_analysis/promptriever/distances.jsonl", "w", encoding="utf-8") as f:
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="running"):
                query = batch[0]
                input_text = f"query:  {query.strip()} {instruction.strip()}".strip()
                doc1 = "passage: "+batch[1]
                query_embedding = model.encode([input_text])
                doc_embeddings = model.encode([doc1])
                query_embedding=torch.tensor(query_embedding)
                doc_embeddings=torch.tensor(doc_embeddings)

                # Calculate similarities
                similarities = np.dot(query_embedding, doc_embeddings.T)[0]

                distance  = torch.norm(doc_embeddings - query_embedding, p=2).item()
                if(batch[2].item()==0):
                    incorrect_pairs.append(distance)
                else:
                    correct_pairs.append(distance)  
                f.write(json.dumps({"sentence1":batch[0],"sentence2":batch[1],"label":batch[2].item(),"distance":distance}) + "\n")
files = analyze_and_save_distances(
        incorrect_pairs,
        correct_pairs,
        title_prefix="Promptriever pairwise distances",
        out_dir="./quoraPD_analysis/promptriever/"   # change if you want a different folder
    )
# Example query and instruction





# # Example documents
# # NOTE: double space after `passage:`

# # doc2 = "passage:  Johns Hopkins University (often abbreviated as Johns Hopkins, Hopkins, or JHU) is a private research university in Baltimore, Maryland. Founded in 1876, Johns Hopkins was the second American university based on the European research institution model."

# # Encode query and documents
# query_embedding = model.encode([input_text])
# doc_embeddings = model.encode([doc1])

# print(f"Similarities: {similarities}") # Similarities: [0.53341305 0.53451955]
# # assert similarities[1] > similarities[0]



