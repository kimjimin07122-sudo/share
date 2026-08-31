"""
Lightweighting + on-device feasibility check for the LSTM-AE + Isolation
Forest baseline (Thursday plan, step: 경량화). No physical Jetson yet, so
this profiles on CPU as a proxy and reports numbers side-by-side with the
B-Spline paper's actual Jetson Nano measurements (93.6% acc, 42ms latency,
127MB memory, ~1.2MB LSTM model) for a rough feasibility argument.
"""
import os
import time

import joblib
import numpy as np
import torch

from config import Config
from networks.lstm_ae import LSTMAutoencoder

ONNX_PATH = os.path.join(Config.BASE_DIR, "lstm_ae.onnx")


def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


INPUT_DIM = Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL


def build_dummy_input(batch=1):
    return torch.randn(batch, Config.LSTM_AE_WINDOW, INPUT_DIM)


def export_onnx(model):
    model.eval()
    dummy = build_dummy_input(1)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["window"], output_names=["reconstruction"],
        dynamic_axes={"window": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported ONNX to {ONNX_PATH} ({model_size_mb(ONNX_PATH):.3f} MB)")


def benchmark_pytorch(model, device, n_runs=200):
    model.eval().to(device)
    x = build_dummy_input(1).to(device)
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1000  # ms per single-window inference


def benchmark_onnx(n_runs=200):
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    x = build_dummy_input(1).numpy()
    for _ in range(10):
        sess.run(None, {"window": x})
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {"window": x})
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1000


def main():
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    model = LSTMAutoencoder(
        input_dim=INPUT_DIM,
        hidden_size=Config.LSTM_AE_HIDDEN_SIZE,
        num_layers=Config.LSTM_AE_NUM_LAYERS,
        bottleneck_dim=Config.LSTM_AE_BOTTLENECK_DIM,
    )
    model.load_state_dict(torch.load(Config.LSTM_AE_MODEL_SAVE_PATH, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())

    print("=== Model footprint ===")
    print(f"LSTM-AE parameters: {n_params:,}")
    print(f"LSTM-AE .pth size: {model_size_mb(Config.LSTM_AE_MODEL_SAVE_PATH):.3f} MB (fp32)")
    print(f"Isolation Forest .joblib size: {model_size_mb(Config.IFOREST_MODEL_SAVE_PATH):.3f} MB "
          f"({Config.IFOREST_N_ESTIMATORS} estimators)")

    print("\n=== CPU latency (single-window inference, proxy for edge device) ===")
    lat_fp32_cpu = benchmark_pytorch(model, device_cpu)
    print(f"PyTorch fp32, CPU: {lat_fp32_cpu:.3f} ms/window")

    # Dynamic INT8 quantization (weights only, standard for LSTM/Linear on CPU)
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )
    quant_path = Config.LSTM_AE_MODEL_SAVE_PATH.replace(".pth", "_int8.pth")
    torch.save(quantized.state_dict(), quant_path)
    print(f"\nQuantized (dynamic INT8) .pth size: {model_size_mb(quant_path):.3f} MB "
          f"({model_size_mb(quant_path) / model_size_mb(Config.LSTM_AE_MODEL_SAVE_PATH) * 100:.1f}% of fp32)")
    lat_int8_cpu = benchmark_pytorch(quantized, device_cpu)
    print(f"PyTorch int8 (dynamic), CPU: {lat_int8_cpu:.3f} ms/window")

    print("\n=== ONNX export + CPU inference ===")
    export_onnx(model)
    lat_onnx_cpu = benchmark_onnx()
    print(f"ONNX Runtime, CPU: {lat_onnx_cpu:.3f} ms/window")

    print("\n=== Isolation Forest latency ===")
    iforest = joblib.load(Config.IFOREST_MODEL_SAVE_PATH)
    # v2: IF now takes compact (original + residual) summary stats, not the
    # raw flattened window -- 2 * (NUM_CHANNELS * FEATURES_PER_CHANNEL) dims.
    if_input_dim = 2 * Config.NUM_CHANNELS * Config.LSTM_AE_FEATURES_PER_CHANNEL
    x_flat = np.random.randn(1, if_input_dim).astype(np.float32)
    for _ in range(10):
        iforest.score_samples(x_flat)
    t0 = time.perf_counter()
    for _ in range(200):
        iforest.score_samples(x_flat)
    t1 = time.perf_counter()
    print(f"Isolation Forest score_samples: {(t1-t0)/200*1000:.3f} ms/window")

    print("\n=== Comparison with B-Spline paper's Jetson Nano numbers ===")
    print(f"{'metric':32s} {'paper (Jetson Nano)':22s} {'ours (CPU proxy)'}")
    print(f"{'LSTM model size':32s} {'~1.2 MB':22s} {model_size_mb(Config.LSTM_AE_MODEL_SAVE_PATH):.3f} MB (fp32) / "
          f"{model_size_mb(quant_path):.3f} MB (int8)")
    print(f"{'total latency (LSTM+IF)':32s} {'42 ms':22s} "
          f"{lat_onnx_cpu + (t1-t0)/200*1000:.3f} ms (onnx + iforest, unoptimized CPU)")


if __name__ == "__main__":
    main()
