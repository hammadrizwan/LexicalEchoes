import json
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader

# ---------- Dataset ----------

class SCPPDataset(Dataset):
    """
    Expects a JSON file that's a list of dicts, each with:
      {
        "caption": str,           # anchor
        "caption2": str,          # paraphrase (positive)
        "negative_caption": str   # distractor (negative)
      }
    """
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        self.data: List[Dict[str, str]] = []
        for record in dataset:
            # minimal validation + trimming
            anchor      = (record.get("caption") or "").strip()
            paraphrase  = (record.get("caption2") or "").strip()
            distractor  = (record.get("negative_caption") or "").strip()

            if not (anchor and paraphrase and distractor):
                continue  # skip malformed/empty rows

            self.data.append({
                "anchor": anchor,
                "paraphrase": paraphrase,
                "distractor": distractor,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Already in the format the SEM trainer expects (per-example dict of strings)
        return self.data[idx]

# ---------- Collate ----------

def sem_collate(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Turn a list of per-example dicts into a dict of lists:
      {
        "anchor":      [str, ...],
        "paraphrase":  [str, ...],
        "distractor":  [str, ...],
      }
    """
    return {
        "anchor":     [ex["anchor"] for ex in batch],
        "paraphrase": [ex["paraphrase"] for ex in batch],
        "distractor": [ex["distractor"] for ex in batch],
    }

# ---------- DataLoader builders ----------

def build_sem_loader_from_json(
    json_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    ds = SCPPDataset(json_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sem_collate,  # ensures dict-of-lists
        pin_memory=True,
    )
