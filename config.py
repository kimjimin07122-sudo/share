import os


class Config:
    # 1. Data parameters from the paper
    NUM_CHANNELS = 16
    SAMPLE_LENGTH = 8
    STRIDE = 4
    DATA_HZ = 10  # 10Hz data
    SCALING_METHOD = "standard"  # paper reports standardization performed best

    # 2. VAE hyperparameters from Table 7
    EPOCHS = 45
    LEARNING_RATE = 1.00e-05
    WARM_UP_PERIOD = 140.4  # KL weight warm-up length in epochs
    BATCH_SIZE = 64
    RANDOM_SEED = 42

    # 3. Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_DATA_ROOT = os.path.join(BASE_DIR, "datasets/original")
    ALT_DATA_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "drone", "datasets", "original"))
    if os.path.isdir(LOCAL_DATA_ROOT):
        DATA_ROOT = LOCAL_DATA_ROOT
    elif os.path.isdir(ALT_DATA_ROOT):
        DATA_ROOT = ALT_DATA_ROOT
    else:
        DATA_ROOT = LOCAL_DATA_ROOT
    TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "train")
    TEST_DATA_PATH = os.path.join(DATA_ROOT, "test")
    # Tried datasets/new/val/04_47_15_sensor_combined_0.csv as the missing Table-3 normal
    # test sortie (size matched 9.77MB), but its baro_altitude/temperature/mag stats don't
    # overlap x70_all.csv at all -> different flight session, not the paper's held-out sortie.
    # Using it inflated the threshold and tanked AUROC (0.91 -> 0.36). Reverted.
    NORMAL_TEST_DATA_PATH = None  # Set this if a genuine separate normal test sortie is found.
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_vae_model.pth")
    EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "results")

    # Fig. 7 fault-identification windowing. Options: full, threshold, peak, manual
    FIG7_MODE = "peak"
    FIG7_PEAK_WINDOW = 60
    FIG7_MANUAL_FILE = None
    FIG7_MANUAL_START_SEC = None
    FIG7_MANUAL_END_SEC = None
    CHANNEL_ATTRIBUTION_METHOD = "ratio"  # raw, excess, ratio

    # 4. GRU forecasting baseline (TiTAD-inspired, step 2 of the adoption plan)
    GRU_WINDOW = 32          # past steps used to predict the next step (3.2s @ 10Hz)
    GRU_HIDDEN_SIZE = 64
    GRU_NUM_LAYERS = 1
    GRU_STRIDE = 4           # window-building stride during training
    GRU_BATCH_SIZE = 64
    GRU_EPOCHS = 30
    GRU_LEARNING_RATE = 1e-3
    GRU_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_gru_model.pth")
    GRU_RESIDUAL_STATS_PATH = os.path.join(BASE_DIR, "gru_residual_stats.npz")

    # 5. LSTM-AE + Isolation Forest (UAV on-device baseline, inspired by
    #    "Edge-Based Real-Time Fault Detection in UAV Systems via B-Spline
    #    Telemetry Reconstruction and Lightweight Hybrid AI", Sensors 2025 --
    #    real PX4 flight logs, Jetson Nano deployment, no public code as of
    #    writing so reimplemented from the paper's described architecture:
    #    2-layer 32-unit LSTM autoencoder + IsolationForest(100 estimators),
    #    fused score = 0.65*LSTM + 0.35*IsolationForest.
    LSTM_AE_WINDOW = 32
    LSTM_AE_STRIDE = 4
    LSTM_AE_HIDDEN_SIZE = 32
    LSTM_AE_NUM_LAYERS = 2
    LSTM_AE_BOTTLENECK_DIM = 16
    LSTM_AE_FEATURES_PER_CHANNEL = 4  # raw, first_deriv, second_deriv, energy
    LSTM_AE_BATCH_SIZE = 64
    LSTM_AE_EPOCHS = 30
    LSTM_AE_LEARNING_RATE = 1e-3
    LSTM_AE_FUSION_ALPHA = 0.65  # weight on LSTM-AE score; (1-alpha) on Isolation Forest
    # 30, not sklearn's 100. On the Jetson the forest was 91% of the
    # detector's per-window budget (8.62 of 10.17 ms) while the ONNX network
    # was 0.49 ms, leaving only 2x headroom against the 20 ms window. A sweep
    # over 100/50/30/20/10/5 trees held AUROC and recall across five seeds at
    # every count -- and channel diagnosis is unaffected either way, since it
    # reads the LSTM-AE residuals and never touches the forest. 30 takes the
    # forest to ~2.9 ms (est.) for ~4.5x headroom while staying a real
    # ensemble; the smaller counts passed too but are too high-variance to
    # stake a flight system on. See experiment_iforest_lightweight.py.
    IFOREST_N_ESTIMATORS = 30
    LSTM_AE_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_lstm_ae_model.pth")
    IFOREST_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "iforest_model.joblib")
    LSTM_AE_STATS_PATH = os.path.join(BASE_DIR, "lstm_ae_stats.npz")
    LSTM_AE_FEATURE_SCALER_PATH = os.path.join(BASE_DIR, "lstm_ae_feature_scaler.joblib")

    # 6. Pipeline v4 (2026-08-20) -- fixes the three input-pipeline defects
    #    found by auditing against both papers (PROJECT_SUMMARY sec 3.9,
    #    quantified in eval_pipeline_fixes_ablation.py, variant "C"):
    #    every flight is resampled to RESAMPLE_HZ, normalized by its OWN
    #    statistics (Reis & Reis sec 6.1: "normalized per flight"), and
    #    windowed by real DURATION instead of a raw sample count -- so a
    #    window means the same amount of time on the ~991Hz training flight
    #    and the ~250Hz test sorties. See preprocessing.py.
    RESAMPLE_HZ = 50.0     # common rate every flight is resampled to
    # Causal normalization (v4.1): fit each flight's scaler on its first
    # WARMUP_SEC seconds only. Whole-flight statistics are NOT causal -- a
    # fault at t=100s measurably shifts the normalized values at t=10s (mean
    # |diff| 0.358), which fabricated lead time in the early-warning
    # experiment -- and they cannot run onboard at all, since future samples
    # do not exist yet at inference time. Mirrors the operational pattern:
    # calibrate over the first seconds of flight, then monitor forward.
    WARMUP_SEC = 30.0
    WINDOW_SEC = 1.0       # window length in real seconds (was 32 samples = 32-128ms)
    STRIDE_SEC = 0.04      # training stride in real seconds (2 samples @50Hz)
    EVAL_STRIDE_SEC = 0.02 # denser stride at evaluation (held-out normal is only ~19.8s)

    # Real-time fault/control windows for x70_20180611_x76_070620.csv.
    # The old "1043-1063s" burst constant was a pseudo-time artifact of
    # row_idx/DATA_HZ=10; in real seconds the fault is 42.19-42.99s of a
    # 45.71s flight, independently confirmed by mag_x's own file-wide max
    # deviation (z=7.55 at 42.65s). See PROJECT_SUMMARY sec 3.8.
    BURST_FILE = "x70_20180611_x76_070620.csv"
    BURST_REAL_SEC = (42.19, 42.99)
    CONTROL_REAL_SEC = [(8.01, 9.22), (20.03, 21.23), (36.46, 37.66)]

    CHANNEL_NAMES = [
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "gyro_integral_dt",
        "accel_timestamp_relative",
        "accel_x",
        "accel_y",
        "accel_z",
        "accel_integral_dt",
        "mag_timestamp_relative",
        "mag_x",
        "mag_y",
        "mag_z",
        "baro_timestamp_relative",
        "baro_altitude",
        "baro_temperature",
    ]
