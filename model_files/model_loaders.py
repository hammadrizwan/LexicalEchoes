from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel, PeftConfig
import torch
class Promptriever:
    def __init__(self, model_name_or_path,device="cuda"):
        self.model, self.tokenizer = self.get_model(model_name_or_path)
        self.model.eval().to(device)

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


def load_promptriever(device="auto"):
    model = Promptriever("samaya-ai/promptriever-llama3.1-8b-instruct-v1",device)
    return model

def load_from_sentence_transformer(model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5", device="auto"):
    model = SentenceTransformer(model_name).to(device)
    return model

def load_from_automodel(model_name="intfloat/multilingual-e5-large-instruct",access_token=None, device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return model,tokenizer

def load_from_causal(model_name="llama",access_token=None,device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  
        device_map=device,
        token=access_token,  # Use access token if required
    )
    model.eval()
    return model, tokenizer


def get_model(model_name="llama",access_token=None,device="auto"):
    if("llama" in model_name or "gemma" in model_name):
        model, tokenizer  = load_from_causal(model_name,access_token,device)
        return model, tokenizer
    elif("e5" in model_name or "Qwen" in model_name):
        model, tokenizer = load_from_automodel(model_name,access_token,device)
        return model, tokenizer
    elif("KaLM" in model_name):
            return load_from_sentence_transformer(model_name,device)
    elif("promptriever" in model_name):
            return load_promptriever(device)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    