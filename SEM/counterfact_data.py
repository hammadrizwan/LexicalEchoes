import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ---------- token helper (no entities) ----------
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def _tokens_no_entities(text: str) -> List[str]:
    """
    Very light tokenization, dropping simple 'entity-like' markers.
    Adjust if your dataset uses different markup.
    """
    if not isinstance(text, str):
        return []
    # strip angle/brace/bracketed spans (heuristic)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


# ---------- Dataset ----------
class CounterFactDatasetPenmeSEM(Dataset):
    """
    Reads a CounterFact-style JSONL where each line is a dict with keys like:
      - "edited_prompt": List[str]      (we take index 0 as anchor)
      - "edited_prompt_paraphrases_processed_testing": List[str] or str
      - "neighborhood_prompts_high_sim": List[str]
      - "neighborhood_prompts_low_sim":  List[str]

    Produces items:
      {"anchor": str, "paraphrase": str, "distractor": str}
    """

    def __init__(self, jsonl_path: str, max_examples: Optional[int] = 2000, choose_min_overlap: bool = False):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        records = [json.loads(line) for line in lines]
        if max_examples is not None:
            records = records[:max_examples]

        self.data: List[Dict[str, str]] = []
        self.choose_min_overlap = False

        for row in records:
            # anchor
            anchor_list = row.get("edited_prompt", [])
            if not anchor_list or not isinstance(anchor_list, list):
                continue
            anchor = (anchor_list[0] or "").strip()
            if not anchor:
                continue

            # paraphrase (ensure a single string)
            paraphrase_raw = row.get("edited_prompt_paraphrases_processed_testing", "")
            if isinstance(paraphrase_raw, list):
                # pick the first non-empty string
                paraphrase = next((p.strip() for p in paraphrase_raw if isinstance(p, str) and p.strip()), "")
            elif isinstance(paraphrase_raw, str):
                paraphrase = paraphrase_raw.strip()
            else:
                paraphrase = ""
            if not paraphrase:
                continue

            # candidate distractors (high + low sim neighborhoods)
            cands = []
            for k in ("neighborhood_prompts_high_sim", "neighborhood_prompts_low_sim"):
                lst = row.get(k, [])
                if isinstance(lst, list):
                    cands.extend([str(s).strip() for s in lst if isinstance(s, str) and s.strip()])
            if not cands:
                continue

            # pick distractor by min overlap (or switch to max overlap by flipping flag)
            if self.choose_min_overlap:
                distractor, _ = self.get_min_overlap(anchor, cands)
            else:
                distractor, _ = self.get_max_overlap(anchor, cands)  # optional alternative below
            if distractor is None or not distractor.strip():
                continue

            self.data.append({
                "anchor": anchor,
                "paraphrase": paraphrase,
                "distractor": distractor.strip(),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]

    # ---- overlap utilities ----
    def get_min_overlap(
        self,
        sentence: str,
        distractor_list: List[str]
    ) -> Tuple[Optional[str], Dict[str, float]]:
        query_tokens = Counter(_tokens_no_entities(sentence))
        best: Optional[str] = None
        # (overlap_count, jaccard) — minimize both
        best_key = (math.inf, math.inf)
        best_scores: Dict[str, float] = {
            "overlap_count": 0,
            "jaccard": 0.0,
            "candidate_len": 0,
            "query_len": sum(query_tokens.values()),
        }

        if not query_tokens:
            return None, best_scores

        for cand in distractor_list:
            cand_tokens = Counter(_tokens_no_entities(cand))
            if not cand_tokens:
                continue

            inter = query_tokens & cand_tokens
            union = query_tokens | cand_tokens
            overlap_count = sum(inter.values())
            union_size = sum(union.values())
            jaccard = overlap_count / union_size if union_size else 0.0

            key = (overlap_count, jaccard)
            if key < best_key:
                best_key = key
                best = cand
                best_scores = {
                    "overlap_count": overlap_count,
                    "jaccard": jaccard,
                    "candidate_len": sum(cand_tokens.values()),
                    "query_len": sum(query_tokens.values()),
                }
        return best, best_scores

    def get_max_overlap(
        self,
        sentence: str,
        distractor_list: List[str]
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Optional: pick most-overlapping distractor instead."""
        query_tokens = Counter(_tokens_no_entities(sentence))
        best: Optional[str] = None
        best_key = (-1, -1.0)  # maximize overlap_count, then jaccard
        best_scores: Dict[str, float] = {
            "overlap_count": 0,
            "jaccard": 0.0,
            "candidate_len": 0,
            "query_len": sum(query_tokens.values()),
        }

        if not query_tokens:
            return None, best_scores

        for cand in distractor_list:
            cand_tokens = Counter(_tokens_no_entities(cand))
            if not cand_tokens:
                continue
            inter = query_tokens & cand_tokens
            union = query_tokens | cand_tokens
            overlap_count = sum(inter.values())
            union_size = sum(union.values())
            jaccard = overlap_count / union_size if union_size else 0.0

            key = (overlap_count, jaccard)
            if key > best_key:
                best_key = key
                best = cand
                best_scores = {
                    "overlap_count": overlap_count,
                    "jaccard": jaccard,
                    "candidate_len": sum(cand_tokens.values()),
                    "query_len": sum(query_tokens.values()),
                }
        return best, best_scores


# ---------- Collate to dict-of-lists (SEM format) ----------
def sem_collate(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {
        "anchor":     [ex["anchor"] for ex in batch],
        "paraphrase": [ex["paraphrase"] for ex in batch],
        "distractor": [ex["distractor"] for ex in batch],
    }


# ---------- Loader builder ----------
def build_sem_loader_from_jsonl_counterfact(
    jsonl_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    max_examples: Optional[int] = 1000,
    choose_min_overlap: bool = False,
) -> DataLoader:
    ds = CounterFactDatasetPenmeSEM(
        jsonl_path=jsonl_path,
        max_examples=max_examples,
        choose_min_overlap=choose_min_overlap,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=sem_collate,
        pin_memory=True,
    )
