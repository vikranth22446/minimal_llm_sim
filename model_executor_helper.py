from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort


_prediction_cache = {}

def load_onnx_model(path):
    session = ort.InferenceSession(path)
    return session


def predict_lgbm_onnx(
    model, seq_lens: List[int], cached_context_lens: List[int], stage: str
) -> float:
    cache_key = (tuple(seq_lens), tuple(cached_context_lens), stage)
    if cache_key in _prediction_cache:
        return _prediction_cache[cache_key]
    
    x = preprocess_input_considering_seq_and_cached_len(
        seq_lens, cached_context_lens, stage
    )[None, :]
    y = model.run(None, {"input": x})[0]
    result = y[0].item()
    
    if len(_prediction_cache) >= 1000:
        oldest_key = next(iter(_prediction_cache))
        del _prediction_cache[oldest_key]
    
    _prediction_cache[cache_key] = result
    return result


class ModelExecutorHelper:
    @staticmethod
    def get_onnx_model_prefill_from_folder(folder_name, model_name):
        return Path(folder_name) / model_name / "prefill_model.onnx"

    def __init__(self, onnx_model_prefill, onnx_model_decode) -> None:
        self.prefill_model = load_onnx_model(onnx_model_prefill)
        self.decode_model = load_onnx_model(onnx_model_decode)

    def prefill(self, seq_lens, cached_context_lens) -> float:
        if cached_context_lens is None:
            cached_context_lens = [0] * len(seq_lens)
        return predict_lgbm_onnx(
            self.prefill_model, seq_lens, cached_context_lens, "prefill"
        )

    def decode(self, seq_lens, cached_context_lens) -> float:
        if cached_context_lens is None:
            cached_context_lens = [0] * len(seq_lens)
        return predict_lgbm_onnx(
            self.decode_model, seq_lens, cached_context_lens, "decode"
        )


def _percentile_from_sorted(sorted_arr, q):
    """
    Pre Sorts the percentiles for perf
    """
    n = sorted_arr.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_arr[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    if lo == hi:
        return float(sorted_arr[lo])
    frac = rank - lo
    return float((1.0 - frac) * sorted_arr[lo] + frac * sorted_arr[hi])


def preprocess_input_considering_seq_and_cached_len(
    seq_lens, cached_context_lens, stage
):
    """
    Build a single-sample rich feature vector for prediction without pandas.
    """
    if stage not in ("prefill", "decode"):
        raise ValueError("stage must be either 'prefill' or 'decode'")
    if len(seq_lens) != len(cached_context_lens):
        raise ValueError("seq_lens and cached_context_lens must have the same length")

    seq = np.asarray(seq_lens, dtype=np.float64)
    cached = np.asarray(cached_context_lens, dtype=np.float64)
    extend = np.maximum(0.0, seq - cached)

    batch_size = float(seq.size)
    total_token_length = float(np.sum(seq))

    seq_sorted = np.sort(seq)
    len_max = float(seq_sorted[-1])
    len_min = float(seq_sorted[0])
    len_std = float(np.std(seq_sorted))
    len_p90 = _percentile_from_sorted(seq_sorted, 90)
    len_p95 = _percentile_from_sorted(seq_sorted, 95)

    cached_sum = float(np.sum(cached))
    cached_max = float(np.max(cached))
    cached_ratio = cached_sum / max(1.0, total_token_length)

    extend_sum = float(np.sum(extend))
    extend_sorted = np.sort(extend)
    extend_max = float(extend_sorted[-1])
    extend_mean = float(np.mean(extend_sorted))
    extend_std = float(np.std(extend_sorted))
    extend_p90 = _percentile_from_sorted(extend_sorted, 90)

    imbalance = (len_max / len_min) if len_min > 0 else np.nan

    if stage == "prefill":
        num_new_tokens = extend_sum
        prod_ext_ctx = float(batch_size * (len_max**2))
    else:
        num_new_tokens = batch_size
        prod_ext_ctx = float(batch_size * len_max)
    num_context_tokens = float(batch_size * len_max)

    len_mean = float(np.mean(seq_sorted))
    mid = seq_sorted.size // 2
    if seq_sorted.size % 2 == 1:
        len_median = float(seq_sorted[mid])
    else:
        len_median = float((seq_sorted[mid - 1] + seq_sorted[mid]) / 2.0)
    len_range = len_max - len_min
    len_p99 = _percentile_from_sorted(seq_sorted, 99)
    len_cv = len_std / max(1.0, len_mean)

    extend_min = float(extend_sorted[0])
    mid_e = extend_sorted.size // 2
    if extend_sorted.size % 2 == 1:
        extend_median = float(extend_sorted[mid_e])
    else:
        extend_median = float((extend_sorted[mid_e - 1] + extend_sorted[mid_e]) / 2.0)
    extend_p99 = _percentile_from_sorted(extend_sorted, 99)
    extend_cv = extend_std / max(1.0, extend_mean) if extend_mean != 0.0 else np.nan

    prompt_ratio = extend_sum / max(1.0, total_token_length)
    cached_peak_ratio = cached_max / max(1.0, len_max)
    B_len_mean = float(batch_size * len_mean)
    B_len_max_sq = float(batch_size * (len_max**2))

    # Keep placeholders consistent with training
    skew = np.nan
    cache_percent = np.nan
    cache_len_prod = cache_percent * len_max if not np.isnan(cache_percent) else np.nan

    log_len_max = float(np.log1p(len_max))
    log_prod_ext_ctx = float(np.log1p(prod_ext_ctx))
    log_num_context_tokens = float(np.log1p(num_context_tokens))

    values = [
        num_new_tokens,
        prod_ext_ctx,
        num_context_tokens,
        len_max,
        len_min,
        len_std,
        len_p90,
        len_p95,
        cached_sum,
        cached_max,
        cached_ratio,
        extend_max,
        extend_mean,
        extend_std,
        extend_p90,
        batch_size,
        imbalance,
        skew,
        cache_percent,
        len_mean,
        len_median,
        len_range,
        len_p99,
        len_cv,
        extend_min,
        extend_median,
        extend_p99,
        extend_cv,
        prompt_ratio,
        cached_peak_ratio,
        B_len_mean,
        B_len_max_sq,
        cache_len_prod,
        log_len_max,
        log_prod_ext_ctx,
        log_num_context_tokens,
    ]
    return np.asarray(values, dtype=np.float32)
