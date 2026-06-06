import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch
from talentmind.config import EMBEDDING_MODEL, LOCAL_MODEL_PATH

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        local = Path(LOCAL_MODEL_PATH)
        path = str(local) if local.exists() else EMBEDDING_MODEL
        # Set CPU threads to prevent thrashing and maximize performance on CPU
        torch.set_num_threads(4)
        _model = SentenceTransformer(path)
    return _model

def embed_texts(texts: list[str], batch_size: int = 512,
                show_progress: bool = True) -> np.ndarray:
    return _get_model().encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

def embed_single(text: str) -> np.ndarray:
    return embed_texts([text], show_progress=False)[0]

def cosine_top_k(query: np.ndarray, corpus: np.ndarray, k: int):
    k = min(k, len(corpus))
    scores = corpus @ query
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx, scores[idx]
