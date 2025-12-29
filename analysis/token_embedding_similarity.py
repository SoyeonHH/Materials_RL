"""Token embedding + similarity analysis (MatBERT vs RoBERTa).

목표:
- domain_token_frequencies.json의 유니크 토큰 42,525개에 대해
  - MatBERT 임베딩
  - RoBERTa-large 임베딩
  을 계산/캐시
- RoBERTa 코사인 유사도 Top-k (본인 제외) 이웃 탐색
- RoBERTa 유사도는 높지만 MatBERT 유사도는 낮은 쌍을 gap 기준으로 정렬
  (일반 도메인에서는 유사하지만 재료과학 도메인에서는 다른 의미의 토큰 쌍 발견)

주의:
- 컨텍스트 없는 단일 토큰 문자열을 임베딩하므로, 문장 임베딩 모델(SBERT 등) 대비 의미적 유사도는 제한적일 수 있습니다.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id_or_path: str
    tokenizer_id_or_path: Optional[str] = None
    cache_dir: Optional[str] = None
    max_length: int = 32
    batch_size: int = 256
    pooling: str = "mean_no_special"  # fixed for now


@dataclass(frozen=True)
class SimilarityConfig:
    topk: int = 20
    matmul_chunk_size: int = 512
    sim_roberta_min: float = 0.80  # RoBERTa에서 높은 유사도 (일반 도메인에서 유사)
    sim_matbert_max: float = 0.40  # MatBERT에서 낮은 유사도 (재료과학 도메인에서 다름)


_MATBERT_S3 = {
    # from https://github.com/lbnlp/MatBERT
    "cased": {
        "config.json": "https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/config.json",
        "vocab.txt": "https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/vocab.txt",
        "pytorch_model.bin": "https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_cased_30522_wd/pytorch_model.bin",
    },
    "uncased": {
        "config.json": "https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/config.json",
        "vocab.txt": "https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/vocab.txt",
        "pytorch_model.bin": "https://cedergroup-share.s3-us-west-2.amazonaws.com/public/MatBERT/model_2Mpapers_uncased_30522_wd/pytorch_model.bin",
    },
}


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted URL constants)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def ensure_matbert_local(analysis_dir: str | Path, *, variant: str = "cased") -> Path:
    """MatBERT 모델 파일을 로컬 폴더로 다운로드/검증 후 경로 반환."""
    analysis_dir = Path(analysis_dir)
    variant = variant.lower()
    if variant not in _MATBERT_S3:
        raise ValueError(f"Unknown MatBERT variant: {variant!r}. Use one of {sorted(_MATBERT_S3.keys())}")

    model_dir = analysis_dir / "models" / f"matbert-base-{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)

    required = _MATBERT_S3[variant]
    for fname, url in required.items():
        dst = model_dir / fname
        if not dst.exists() or dst.stat().st_size == 0:
            _download_file(url, dst)

    return model_dir


def load_token_frequencies(json_path: str | Path) -> Dict[str, int]:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        token_freq = json.load(f)
    if not isinstance(token_freq, dict):
        raise ValueError(f"Expected dict in {json_path}, got {type(token_freq)}")
    # ensure int values
    out: Dict[str, int] = {}
    for k, v in token_freq.items():
        if not isinstance(k, str):
            k = str(k)
        try:
            out[k] = int(v)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Invalid frequency for token={k!r}: {v!r}") from e
    return out


def build_token_list(token_freq: Dict[str, int], sort_tokens: bool = True) -> Tuple[List[str], np.ndarray]:
    tokens = list(token_freq.keys())
    if sort_tokens:
        tokens = sorted(tokens)
    freqs = np.asarray([token_freq[t] for t in tokens], dtype=np.int64)
    return tokens, freqs


def _pick_device(prefer: str = "auto") -> torch.device:
    prefer = (prefer or "auto").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x: (N, D)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


@torch.no_grad()
def embed_tokens(
    tokens: Sequence[str],
    cfg: EmbeddingConfig,
    device: torch.device,
    *,
    add_prefix_space_for_roberta: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """단일 토큰 문자열들을 임베딩합니다.

    pooling: special token 제외(mean pooling)
    반환: float32 numpy array (N, hidden_size)
    """

    tokenizer_id = cfg.tokenizer_id_or_path or cfg.model_id_or_path
    tok_kwargs = {}
    if add_prefix_space_for_roberta:
        tok_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast=True,
        cache_dir=cfg.cache_dir,
        **tok_kwargs,
    )
    model = AutoModel.from_pretrained(cfg.model_id_or_path, cache_dir=cfg.cache_dir)
    model.eval()
    model.to(device)

    # batching
    bs = int(cfg.batch_size)
    max_len = int(cfg.max_length)

    all_vecs: List[np.ndarray] = []
    it = range(0, len(tokens), bs)
    if show_progress:
        it = tqdm(it, total=math.ceil(len(tokens) / bs), desc=f"embed: {cfg.model_id_or_path}")

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    for start in it:
        batch = tokens[start : start + bs]

        # RoBERTa byte-BPE에서는 standalone 토큰이 문장 중간에 등장하는 것과 다르게 처리될 수 있어
        # 필요 시 선행 공백을 붙여 일관성을 맞춥니다.
        if add_prefix_space_for_roberta:
            batch = [" " + t if (t and not t.startswith(" ")) else t for t in batch]

        enc = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        use_amp = device.type == "cuda"
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = out.last_hidden_state  # (B, T, H)

        # special token 제외
        # mask: attention_mask == 1 이면서 special id가 아닌 위치만
        mask = attention_mask.bool()
        if special_ids:
            special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for sid in special_ids:
                special_mask |= input_ids.eq(sid)
            mask = mask & (~special_mask)

        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom

        pooled = pooled.float().detach().cpu().numpy().astype(np.float32)
        all_vecs.append(pooled)

    return np.concatenate(all_vecs, axis=0)


def _write_tokens_txt(path: Path, tokens: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t.replace("\n", " ") + "\n")


def _read_tokens_txt(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_or_compute_embeddings(
    *,
    tokens: Sequence[str],
    emb_dir: str | Path,
    cache_stem: str,
    cfg: EmbeddingConfig,
    device: torch.device,
    add_prefix_space_for_roberta: bool = False,
) -> np.ndarray:
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = emb_dir / "tokens.txt"
    emb_path = emb_dir / f"{cache_stem}.npy"
    meta_path = emb_dir / f"{cache_stem}.meta.json"

    if tokens_path.exists() and emb_path.exists() and meta_path.exists():
        cached_tokens = _read_tokens_txt(tokens_path)
        if list(tokens) == cached_tokens:
            arr = np.load(emb_path)
            return arr.astype(np.float32, copy=False)

    # compute
    arr = embed_tokens(
        tokens=tokens,
        cfg=cfg,
        device=device,
        add_prefix_space_for_roberta=add_prefix_space_for_roberta,
        show_progress=True,
    )

    _write_tokens_txt(tokens_path, tokens)
    np.save(emb_path, arr)
    meta = {
        "model_id_or_path": cfg.model_id_or_path,
        "tokenizer_id_or_path": cfg.tokenizer_id_or_path or cfg.model_id_or_path,
        "max_length": cfg.max_length,
        "batch_size": cfg.batch_size,
        "pooling": cfg.pooling,
        "add_prefix_space_for_roberta": bool(add_prefix_space_for_roberta),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return arr


def _maybe_move_to_device(x: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.device]:
    if device.type != "cuda":
        return x, device
    try:
        return x.to(device), device
    except torch.cuda.OutOfMemoryError:
        # fallback to CPU
        torch.cuda.empty_cache()
        return x.cpu(), torch.device("cpu")


def compute_topk_neighbors(
    emb: np.ndarray,
    *,
    topk: int = 20,
    chunk_size: int = 512,
    prefer_device: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """정규화 코사인 유사도 기준 Top-k(본인 제외) 이웃.

    반환:
      - neighbor_idx: (N, topk) int64
      - neighbor_sim: (N, topk) float32
    """

    if emb.ndim != 2:
        raise ValueError(f"Expected 2D emb, got shape={emb.shape}")

    N, D = emb.shape
    if topk >= N:
        raise ValueError(f"topk must be < N (topk={topk}, N={N})")

    emb_norm = _normalize_rows(emb.astype(np.float32, copy=False))
    E = torch.from_numpy(emb_norm)

    device = _pick_device(prefer_device)
    E_dev, device = _maybe_move_to_device(E, device)

    neighbor_idx = np.empty((N, topk), dtype=np.int64)
    neighbor_sim = np.empty((N, topk), dtype=np.float32)

    # compute in chunks
    for start in tqdm(range(0, N, chunk_size), desc="topk neighbors", total=math.ceil(N / chunk_size)):
        end = min(N, start + chunk_size)
        Q = E_dev[start:end]

        # similarity matrix for chunk: (B, N)
        S = Q @ E_dev.T

        # exclude self
        row = torch.arange(end - start, device=device)
        col = torch.arange(start, end, device=device)
        S[row, col] = -1e9

        vals, idx = torch.topk(S, k=topk, dim=1, largest=True, sorted=True)

        neighbor_idx[start:end] = idx.detach().cpu().numpy().astype(np.int64)
        neighbor_sim[start:end] = vals.detach().cpu().numpy().astype(np.float32)

    return neighbor_idx, neighbor_sim


def compute_sims_for_pairs(
    emb: np.ndarray,
    neighbor_idx: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """(i, neighbor_idx[i, j]) 쌍에 대한 코사인 유사도를 효율적으로 계산.

    반환: (N, K) float32
    """

    emb_norm = _normalize_rows(emb.astype(np.float32, copy=False))
    N, D = emb_norm.shape
    K = neighbor_idx.shape[1]

    out = np.empty((N, K), dtype=np.float32)

    for start in tqdm(range(0, N, chunk_size), desc="pair similarities", total=math.ceil(N / chunk_size)):
        end = min(N, start + chunk_size)
        a = emb_norm[start:end]  # (B, D)
        idx = neighbor_idx[start:end]  # (B, K)
        b = emb_norm[idx]  # (B, K, D)
        # dot
        out[start:end] = (a[:, None, :] * b).sum(axis=-1).astype(np.float32)

    return out


def build_topk_dataframe(
    *,
    tokens: Sequence[str],
    freqs: np.ndarray,
    neighbor_idx: np.ndarray,
    sim_matbert: np.ndarray,
    sim_roberta: np.ndarray,
) -> pd.DataFrame:
    N, K = neighbor_idx.shape
    tok_arr = np.asarray(tokens)

    token_col = np.repeat(tok_arr, K)
    neighbor_col = tok_arr[neighbor_idx.reshape(-1)]

    freq_token = np.repeat(freqs, K)
    freq_neighbor = freqs[neighbor_idx.reshape(-1)]

    df = pd.DataFrame(
        {
            "token": token_col,
            "neighbor": neighbor_col,
            "sim_matbert": sim_matbert.reshape(-1).astype(np.float32),
            "sim_roberta": sim_roberta.reshape(-1).astype(np.float32),
            "gap": (sim_matbert.reshape(-1) - sim_roberta.reshape(-1)).astype(np.float32),
            "freq_token": freq_token.astype(np.int64),
            "freq_neighbor": freq_neighbor.astype(np.int64),
        }
    )

    # convenience ordering (RoBERTa 유사도 기준으로 정렬)
    return df.sort_values(["token", "sim_roberta"], ascending=[True, False]).reset_index(drop=True)


def run_pipeline(
    *,
    analysis_dir: str | Path,
    matbert_model: str = "matbert-base-cased",
    roberta_model: str = "roberta-large",
    emb_subdir: str = "embeddings",
    topk: int = 20,
    matmul_chunk_size: int = 512,
    roberta_pair_chunk_size: int = 1024,
    prefer_device: str = "auto",
    sim_roberta_min: float = 0.80,  # RoBERTa에서 높은 유사도 (일반 도메인에서 유사)
    sim_matbert_max: Optional[float] = None,  # 변경: 기본은 고정 임계값 대신 분위수 사용
    sim_matbert_low_quantile: float = 0.10,   # 추가: MatBERT 하위 10%
) -> Tuple[Path, Path]:
    """전체 파이프라인 실행.

    RoBERTa 임베딩 기준으로 top-k 이웃을 찾고, 그 쌍들에 대해 MatBERT 유사도를 계산한다.
    RoBERTa는 높지만 MatBERT는 낮은 쌍을 찾아 일반 도메인에서는 유사하지만
    재료과학 도메인에서는 다른 의미를 가진 토큰 쌍을 발견한다.

    반환: (topk_csv_path, gap_csv_path)
    """

    analysis_dir = Path(analysis_dir)
    token_json = analysis_dir / "domain_token_frequencies.json"
    emb_dir = analysis_dir / emb_subdir
    hf_cache_dir = str(analysis_dir / ".hf_cache")
    Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)

    # MatBERT는 HF에 항상 공개되어 있지 않아(또는 ID가 다를 수 있어) 기본값은 S3에서 로컬로 내려받습니다.
    matbert_model_path: str
    if Path(matbert_model).exists():
        matbert_model_path = matbert_model
    elif matbert_model in {"matbert-base-cased", "matbert_cased", "cased"}:
        matbert_model_path = str(ensure_matbert_local(analysis_dir, variant="cased"))
    elif matbert_model in {"matbert-base-uncased", "matbert_uncased", "uncased"}:
        matbert_model_path = str(ensure_matbert_local(analysis_dir, variant="uncased"))
    else:
        # 사용자가 유효한 HF 모델 ID를 직접 넘기는 경우를 허용
        matbert_model_path = matbert_model

    token_freq = load_token_frequencies(token_json)
    tokens, freqs = build_token_list(token_freq, sort_tokens=True)

    device = _pick_device(prefer_device)

    mat_cfg = EmbeddingConfig(model_id_or_path=matbert_model_path, cache_dir=hf_cache_dir, max_length=32, batch_size=256)
    rob_cfg = EmbeddingConfig(model_id_or_path=roberta_model, cache_dir=hf_cache_dir, max_length=32, batch_size=128)

    mat_emb = load_or_compute_embeddings(
        tokens=tokens,
        emb_dir=emb_dir,
        cache_stem="matbert",
        cfg=mat_cfg,
        device=device,
        add_prefix_space_for_roberta=False,
    )

    rob_emb = load_or_compute_embeddings(
        tokens=tokens,
        emb_dir=emb_dir,
        cache_stem="roberta_large",
        cfg=rob_cfg,
        device=device,
        add_prefix_space_for_roberta=True,
    )

    # RoBERTa 기준으로 top-k 이웃 찾기 (일반 도메인에서 유사한 토큰)
    neigh_idx, sim_rob = compute_topk_neighbors(
        rob_emb,
        topk=topk,
        chunk_size=matmul_chunk_size,
        prefer_device=prefer_device,
    )

    # 그 쌍들에 대해 MatBERT 유사도 계산 (재료과학 도메인에서의 유사도)
    sim_mat = compute_sims_for_pairs(mat_emb, neigh_idx, chunk_size=roberta_pair_chunk_size)

    df_topk = build_topk_dataframe(
        tokens=tokens,
        freqs=freqs,
        neighbor_idx=neigh_idx,
        sim_matbert=sim_mat,
        sim_roberta=sim_rob,
    )

    topk_path = analysis_dir / "roberta_topk_neighbors.csv"
    df_topk.to_csv(topk_path, index=False)

    # RoBERTa는 높지만 MatBERT는 낮은 쌍 필터링 (일반 도메인에서는 유사하지만 재료과학에서는 다른 의미)
    df_hi_rob = df_topk[df_topk["sim_roberta"] >= sim_roberta_min].copy()

    if sim_matbert_max is None:
        matbert_thr = df_hi_rob["sim_matbert"].quantile(sim_matbert_low_quantile)
    else:
        matbert_thr = sim_matbert_max

    df_gap = df_hi_rob[df_hi_rob["sim_matbert"] <= matbert_thr].copy()
    df_gap = df_gap.sort_values(["gap", "sim_roberta"], ascending=[True, True]).reset_index(drop=True)

    gap_path = analysis_dir / "roberta_high_matbert_low_pairs.csv"
    df_gap.to_csv(gap_path, index=False)

    return topk_path, gap_path


if __name__ == "__main__":
    topk_path, gap_path = run_pipeline(analysis_dir=Path(__file__).resolve().parent)
    print(f"Wrote: {topk_path}")
    print(f"Wrote: {gap_path}")
