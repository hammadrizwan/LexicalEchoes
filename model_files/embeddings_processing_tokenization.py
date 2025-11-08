import torch
import nethook

def get_instruction_mask(inputs, tokenizer, texts, embeddings, *,
                          lowercase_texts=False, device="auto"):
    """
    If embeddings.ndim == 3: (B, L, H)  -> returns (B, L, 1)
    If embeddings.ndim == 4: (T, B, L, H) -> returns (T, B, L, 1)
    """
    if device == "auto":
        device = embeddings.device

    # Normalize texts the same way you did in build_batched_llama_chat_inputs
    if lowercase_texts:
        norm_texts = [t.lower() for t in texts]
    else:
        norm_texts = list(texts)

    # Tokenize raw texts WITHOUT special/chat tokens to get the pure content pattern
    # (important: add_special_tokens=False)
    tok = tokenizer(norm_texts, add_special_tokens=False)

    # Accept either (B, L, H) or (T, B, L, H)
    if embeddings.ndim == 3:
        B, L, _H = embeddings.shape
        T = None
    elif embeddings.ndim == 4:
        T, B, L, _H = embeddings.shape
    else:
        raise ValueError(f"embeddings must be 3D or 4D, got {tuple(embeddings.shape)}")

    input_ids_batch = inputs["input_ids"]        # (B, L)
    attention_mask  = inputs["attention_mask"]   # (B, L)

    base_mask = torch.zeros((B, L, 1), dtype=embeddings.dtype, device=device)

    # For each item in the batch, reverse-search the pure-text token pattern
    for i, pat in enumerate(tok["input_ids"]):
        seq = input_ids_batch[i].tolist()
        m = len(pat)
        best_j = -1

        if m > 0 and m <= len(seq):
            # reverse search: prefer the last occurrence (most recent message)
            for j in range(len(seq) - m, -1, -1):
                if seq[j:j+m] == pat:
                    best_j = j
                    break

        if best_j >= 0:
            base_mask[i, best_j:best_j+m, 0] = 1.0
        else:
            # Fallback: use attention_mask so the row isn't empty
            base_mask[i] = attention_mask[i].unsqueeze(-1).type_as(base_mask)

    # Safety: guarantee at least 1 token selected per row
    empty = (base_mask.sum(dim=1) == 0).squeeze(-1)
    if empty.any():
        base_mask[empty] = attention_mask[empty].unsqueeze(-1).type_as(base_mask)

    # Broadcast across layers if needed
    if T is not None:
        return base_mask.unsqueeze(0).expand(T, B, L, 1)
    return base_mask

def build_batched_pt_inputs(tokenizer, texts, device="auto"):
    """
    Build a batch of prompts for Gemma-PT (base model) using plain tokenization.
    `texts` is a list[str]. If you were lowercasing upstream for IT, keep it for parity.
    Returns: dict with input_ids, attention_mask on device.
    """
    enc = tokenizer(
        [t.lower() for t in texts],   # keep/lift if you *don’t* want lowercase
        padding=True,                 # important for batching
        truncation=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

def build_batched_chat_inputs(tokenizer, texts, add_generation_prompt=True,PROMPT="", device="auto"):
    """
    Build a batch of chat-formatted prompts for Gemma-IT using tokenizer.apply_chat_template.
    `texts` is a list[str] (already lowercased / formatted upstream if you want).
    Returns: dict with input_ids, attention_mask on device.
    """
    # Each item in the batch is a full conversation (list of messages)
    conversations = [
        [
            
            {"role": "user",
             "content": [{"type": "text", "text": "{}".format(t.lower())}]}
        ]
        for t in texts
    ]
    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        padding=True,                # important for batching
        return_dict=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in inputs.items()}

def get_average_embeddings(embeddings, mask=None, content_mask=None, normalize=True):
    if(mask is not None):
        padded_zero_embeddings=embeddings * mask
        sum_hidden = padded_zero_embeddings.sum(dim=2)                # (B, L,H)
        lengths = mask.sum(dim=1)
    elif(content_mask is not None):
        padded_zero_embeddings=embeddings * content_mask
        sum_hidden = padded_zero_embeddings.sum(dim=2)
        lengths = content_mask.sum(dim=2)                                  # (B, L, 1)
    else:
        raise ValueError("Either mask or content_mask must be provided.")
    embeddings= sum_hidden / lengths.clamp(min=1)
    if normalize:
        return torch.nn.functional.normalize(embeddings, p=2, dim=2, eps=1e-12)
    else:
        return embeddings

def get_last_token_embeddings(embeddings, mask=None, content_mask=None, normalize=True):
    L, B, T, E = embeddings.shape
    if(mask is not None):
        base_mask = mask.squeeze(-1).to(embeddings.dtype)
    elif(content_mask is not None):
        base_mask = content_mask[0].squeeze(-1).to(embeddings.dtype)
    else:
        raise ValueError("Either mask or content_mask must be provided.")

    token_range = torch.arange(T, device=embeddings.device, dtype=embeddings.dtype)
    last_idx = (base_mask * token_range).argmax(dim=1).long()
    idx = last_idx.view(1, B, 1, 1).expand(L, B, 1, E)
    embeddings=  embeddings.gather(2, idx).squeeze(2)
    if normalize:
        return torch.nn.functional.normalize(embeddings, p=2, dim=2, eps=1e-12)
    else:
        return embeddings


def get_embeddings_pt(model, tokenizer, args, data, layers, normalize=True, device="auto"):
    with nethook.TraceDict(model, layers) as ret:
        with torch.no_grad():
            inputs = build_batched_pt_inputs(tokenizer,data["anchors"], device=device)
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            ) 
            anchor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
            mask = inputs["attention_mask"].unsqueeze(-1)
            if(args.mode):
                anchor_sentence_embeddings = get_average_embeddings(anchor_sentence_embeddings,mask=mask,normalize=normalize)
                
            else:
                anchor_sentence_embeddings = get_last_token_embeddings(anchor_sentence_embeddings,mask=mask,normalize=normalize)
            #_________________________________________________________________________________________
            inputs = build_batched_pt_inputs(tokenizer, data["paraphrases"], device=device)
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            ) 
            paraphrase_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
            mask = inputs["attention_mask"].unsqueeze(-1)
            if(args.mode):
                paraphrase_sentence_embeddings= get_average_embeddings(paraphrase_sentence_embeddings,mask=mask,normalize=normalize)

            else:
                paraphrase_sentence_embeddings = get_last_token_embeddings(paraphrase_sentence_embeddings,mask=mask,normalize=normalize)
            #_________________________________________________________________________________________
            inputs = build_batched_pt_inputs(tokenizer, data["distractors"], device=device)
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            ) 
            distractor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
            mask = inputs["attention_mask"].unsqueeze(-1)
            if(args.mode):
                distractor_sentence_embeddings = get_average_embeddings(distractor_sentence_embeddings,mask=mask,normalize=normalize)
            else:
                distractor_sentence_embeddings = get_last_token_embeddings(distractor_sentence_embeddings,mask=mask,normalize=normalize)

    return anchor_sentence_embeddings,paraphrase_sentence_embeddings,distractor_sentence_embeddings

def get_embedding_it(model, tokenizer, args, data, layers, normalize=True ,device="auto"):
    with nethook.TraceDict(model, layers) as ret:
        with torch.no_grad():
            inputs = build_batched_chat_inputs(tokenizer, data["anchors"], "", device=device)
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            ) 
            anchor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
            content_mask = get_instruction_mask(inputs, tokenizer, data["anchors"], anchor_sentence_embeddings, device=device)
             
            if(args.mode):
                # intrasims= compute_intrasim_same_mask_across_layers(anchor_sentence_embeddings, content_mask, eps=1e-12)
                anchor_sentence_embeddings = get_average_embeddings(anchor_sentence_embeddings, content_mask=content_mask,normalize=normalize)                  
            else:
                anchor_sentence_embeddings = get_last_token_embeddings(anchor_sentence_embeddings,content_mask=content_mask,normalize=normalize)
            
            #_________________________________________________________________________________________
            inputs = build_batched_chat_inputs(tokenizer, data["paraphrases"], "", device=device)
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            ) 
            paraphrase_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
            content_mask = get_instruction_mask(inputs, tokenizer, data["paraphrases"], paraphrase_sentence_embeddings, device=device) 
            if(args.mode):
                paraphrase_sentence_embeddings = get_average_embeddings(paraphrase_sentence_embeddings,content_mask=content_mask,normalize=normalize) 

            else:
                paraphrase_sentence_embeddings = get_last_token_embeddings(paraphrase_sentence_embeddings,content_mask=content_mask,normalize=normalize)

                
            #_________________________________________________________________________________________
            inputs = build_batched_chat_inputs(tokenizer, data["distractors"], "", device=device)
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            ) 
            distractor_sentence_embeddings = torch.stack([ret[layer_key].output[0] for layer_key in ret],dim=0)
            content_mask = get_instruction_mask(inputs, tokenizer, data["distractors"], distractor_sentence_embeddings, device=device) 
            if(args.mode):
                distractor_sentence_embeddings = get_average_embeddings(distractor_sentence_embeddings, content_mask=content_mask,normalize=normalize)
            else:
                distractor_sentence_embeddings = get_last_token_embeddings(distractor_sentence_embeddings, content_mask=content_mask,normalize=normalize)

    return anchor_sentence_embeddings,paraphrase_sentence_embeddings,distractor_sentence_embeddings
