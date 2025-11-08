from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch

def build_wikitext_loaders(
    tokenizer,
    seq_len: int = 50,
    batch_size: int = 32,
    split: str = "train",
    name: str = "wikitext-103-raw-v1",  # or "wikitext-2-raw-v1"
    num_workers: int = 2,
    streaming: bool = False,
    shuffle_buffer: int = 200,
    max_sequences: int = 2000,            # <-- cap number of sequences
) -> DataLoader:
    """
    Returns a DataLoader that yields dicts with keys:
      input_ids, labels, attention_mask
    Chunks token ids into fixed-length sequences.
    Caps total sequences to `max_sequences` without tokenizing the whole split.
    """
    from datasets import load_dataset

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ---------- Non-streaming: incrementally tokenize until cap is reached ----------
    if not streaming:
        ds = load_dataset("wikitext", name, split=split)

        cap_tokens = max_sequences * seq_len
        buf = []
        buf_len = 0

        for rec in ds:
            t = rec.get("text", "")
            if not t or not t.strip():
                continue
            # tokenize this record only
            tok = tokenizer(t, add_special_tokens=False).input_ids
            buf.extend(tok)
            buf_len += len(tok)
            if buf_len >= cap_tokens:
                break

        # trim to multiple of seq_len and view into [N, seq_len]
        total = (len(buf) // seq_len) * seq_len
        ids = torch.tensor(buf[:total], dtype=torch.long).view(-1, seq_len)

        class _FixedChunkDataset(Dataset):
            def __len__(self): return ids.size(0)
            def __getitem__(self, i):
                x = ids[i]
                return {
                    "input_ids": x,
                    "labels": x.clone(),
                    "attention_mask": torch.ones_like(x, dtype=torch.long),
                }

        return DataLoader(
            _FixedChunkDataset(),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # ---------- Streaming: chunk on the fly, stop after max_sequences ----------
    ds = load_dataset("wikitext", name, split=split, streaming=True)
    if shuffle_buffer and split == "train":
        ds = ds.shuffle(seed=42, buffer_size=shuffle_buffer)

    class _StreamChunks(IterableDataset):
        def __iter__(self):
            buf = []
            buf_len = 0
            yielded = 0
            for rec in ds:
                if yielded >= max_sequences:
                    break
                t = rec.get("text", "")
                if not t or not t.strip():
                    continue
                tok = tokenizer(t, add_special_tokens=False).input_ids
                buf.extend(tok)
                buf_len += len(tok)
                while buf_len >= seq_len and yielded < max_sequences:
                    chunk = buf[:seq_len]
                    buf = buf[seq_len:]
                    buf_len -= seq_len
                    x = torch.tensor(chunk, dtype=torch.long)
                    yielded += 1
                    yield {
                        "input_ids": x,
                        "labels": x.clone(),
                        "attention_mask": torch.ones_like(x, dtype=torch.long),
                    }

    def _collate(batch):
        x = torch.stack([b["input_ids"] for b in batch], dim=0)
        return {
            "input_ids": x,
            "labels": x.clone(),
            "attention_mask": torch.ones_like(x, dtype=torch.long),
        }

    return DataLoader(
        _StreamChunks(),
        batch_size=batch_size,
        shuffle=False,  # shuffling handled by dataset.shuffle(...)
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )
