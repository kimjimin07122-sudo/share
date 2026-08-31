"""
Raw data 기반 채널 진단 스크립트
- A단계: 정상(x70_all.csv) vs Test 데이터의 채널별 통계 비교
- B단계: 11/12/15번 채널의 time-series 시각화
- C단계: 채널 제외 옵션 테스트 (e.g., gyro/accel/mag/baro 센서값만 비교)
- D단계: 특정 시간 범위 지정 (e.g., 논문 Fig.7의 fault 구간)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from config import Config

# 플롯 스타일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_data(file_path: str, time_range=None, exclude_channels=None):
    """
    CSV 파일을 로드하고, 지정된 시간 범위와 제외할 채널을 적용합니다.
    
    Args:
        file_path: CSV 파일 경로
        time_range: [start_idx, end_idx] 형태의 시간 범위 (선택)
        exclude_channels: 제외할 채널 인덱스 리스트 (선택)
    
    Returns:
        pd.DataFrame or None
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found - {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, comment="#", header=None)
        # 숫자로 변환 시도
        df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
        original_shape = df.shape
        print(f"✓ Loaded {file_path}: shape {original_shape}")
        
        # drone 코드와 동일하게 첫 번째 열을 제외하고 16개 센서 채널만 사용
        if df.shape[1] == Config.NUM_CHANNELS + 1:
            df = df.iloc[:, 1: Config.NUM_CHANNELS + 1]
            print(f"  → Dropped first column, selected sensor channels 1:{Config.NUM_CHANNELS + 1}")
        elif df.shape[1] > Config.NUM_CHANNELS + 1:
            df = df.iloc[:, -Config.NUM_CHANNELS:]
            print(f"  → Unexpected column count; selected last {Config.NUM_CHANNELS} columns")
        elif df.shape[1] < Config.NUM_CHANNELS:
            raise ValueError(f"Expected at least {Config.NUM_CHANNELS} columns but got {df.shape[1]}")
        
        # D단계: 특정 시간 범위 지정 (e.g., 논문 Fig.7의 fault 구간)
        if time_range:
            df = df.iloc[time_range[0]:time_range[1]].reset_index(drop=True)
            print(f"  → Time range applied: [{time_range[0]}:{time_range[1]}]")
        
        # C단계: 채널 제외 옵션 (e.g., 실제 센서값만 비교)
        if exclude_channels:
            valid_exclude = [ch for ch in exclude_channels if ch < len(df.columns)]
            if valid_exclude:
                cols_to_drop = df.columns[valid_exclude]
                df = df.drop(columns=cols_to_drop)
                ch_names = [Config.CHANNEL_NAMES[i] for i in valid_exclude if i < len(Config.CHANNEL_NAMES)]
                print(f"  → Excluded channels: {valid_exclude} {ch_names}")
        
        return df
    
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")
        return None


def compare_channel_stats(normal_df, test_df, output_csv_path, channel_names=None):
    """
    A. 정상 데이터와 테스트 데이터의 채널별 통계를 비교하고 출력합니다.
    
    - Z-Score: 테스트 데이터가 정상 데이터 분포에서 얼마나 벗어났는지
    - Mean Diff: 두 데이터셋 간의 평균 차이
    - Peak Diff: 두 데이터셋 간의 최댓값 차이
    - Std Ratio: 테스트 std / 정상 std (변동성 증가 여부 판단)
    """
    print("\n" + "="*80)
    print("STEP A: Raw Channel Statistics Comparison")
    print("="*80)
    
    # NaN 처리
    normal_num = normal_df.fillna(normal_df.mean())
    test_num = test_df.fillna(test_df.mean())
    
    # 기본 통계
    normal_mean = normal_num.mean()
    normal_std = normal_num.std()
    normal_std = normal_std.replace(0, 1e-9)  # 분모 0 방지
    
    test_mean = test_num.mean()
    test_std = test_num.std()
    test_std = test_std.replace(0, 1e-9)
    
    # 메트릭 계산
    z_scores = np.abs((test_mean - normal_mean) / normal_std)
    mean_diff = np.abs(test_mean - normal_mean)
    peak_diff = np.abs(test_num.max() - normal_num.max())
    std_ratio = test_std / normal_std
    
    # 결과 데이터프레임
    stats_df = pd.DataFrame({
        'Channel': [f"{i}: {channel_names[i] if channel_names and i < len(channel_names) else 'CH'}" 
                   for i in range(len(normal_df.columns))],
        'Z-Score': z_scores.values,
        'Mean Diff': mean_diff.values,
        'Peak Diff': peak_diff.values,
        'Std Ratio': std_ratio.values,
        'Normal Mean': normal_mean.values,
        'Test Mean': test_mean.values,
    })
    
    # 1. Z-Score 기준 상위 10개
    print("\n[Top 10 by Z-Score (Mean deviation in units of normal std)]")
    top_z = stats_df.nlargest(10, 'Z-Score')[['Channel', 'Z-Score', 'Mean Diff']]
    print(top_z.to_string(index=False))
    
    # 2. Mean Diff 기준
    print("\n[Top 10 by Mean Diff (Absolute mean difference)]")
    top_mean = stats_df.nlargest(10, 'Mean Diff')[['Channel', 'Mean Diff', 'Z-Score']]
    print(top_mean.to_string(index=False))
    
    # 3. Peak Diff 기준
    print("\n[Top 10 by Peak Diff (Max value difference)]")
    top_peak = stats_df.nlargest(10, 'Peak Diff')[['Channel', 'Peak Diff', 'Mean Diff']]
    print(top_peak.to_string(index=False))
    
    # 4. Std Ratio 기준 (변동성 증가)
    print("\n[Top 10 by Std Ratio (Volatility increase)]")
    top_std = stats_df.nlargest(10, 'Std Ratio')[['Channel', 'Std Ratio', 'Z-Score']]
    print(top_std.to_string(index=False))
    
    # CSV 저장
    try:
        stats_df.to_csv(output_csv_path, index=False)
        print(f"\n✓ Statistics saved to '{output_csv_path}'")
    except Exception as e:
        print(f"ERROR saving stats: {e}")
    
    print("-" * 80)
    return stats_df


def plot_channel_timeseries(normal_df, test_df, channels_to_plot, output_filename, channel_names=None):
    """
    B. 지정된 채널의 time-series를 정상 vs 테스트로 비교 시각화합니다.
    """
    print("\n" + "="*80)
    print("STEP B: Channel Time-Series Comparison Visualization")
    print("="*80)
    
    num_channels = len(channels_to_plot)
    fig, axes = plt.subplots(num_channels, 1, figsize=(16, 5 * num_channels))
    if num_channels == 1:
        axes = [axes]
    
    print(f"Plotting channels: {channels_to_plot}")
    
    for i, ch_idx in enumerate(channels_to_plot):
        if ch_idx >= len(normal_df.columns):
            print(f"  WARNING: Channel {ch_idx} exceeds data columns ({len(normal_df.columns)})")
            continue
        
        ax = axes[i]
        ch_name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"CH{ch_idx}"
        
        # Normal과 Test 데이터 변환
        normal_vals = pd.to_numeric(normal_df.iloc[:, ch_idx], errors='coerce').fillna(0)
        test_vals = pd.to_numeric(test_df.iloc[:, ch_idx], errors='coerce').fillna(0)
        
        # 플로팅
        ax.plot(normal_vals.index, normal_vals, 'b-', label='Normal (x70_all)', linewidth=2, alpha=0.7)
        ax.plot(test_vals.index, test_vals, 'r--', label=f'Test ({os.path.basename(test_df.name)})', 
               linewidth=2, alpha=0.7)
        
        # 통계 표시
        normal_mean = normal_vals.mean()
        test_mean = test_vals.mean()
        ax.axhline(normal_mean, color='b', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(test_mean, color='r', linestyle=':', alpha=0.5, linewidth=1)
        
        ax.set_title(f'Channel {ch_idx}: {ch_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Index (samples @10Hz)')
    fig.suptitle('Raw Data Time-Series Comparison\n(Normal vs Test)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to '{output_filename}'")
    except Exception as e:
        print(f"ERROR saving plot: {e}")
    
    plt.close()
    print("-" * 80)


def plot_channel_distribution(normal_df, test_df, channels_to_plot, output_filename, channel_names=None):
    """
    보너스: 히스토그램으로 분포 비교
    """
    print("\nGenerating distribution comparison plots...")
    
    num_channels = len(channels_to_plot)
    fig, axes = plt.subplots(num_channels, 1, figsize=(16, 4 * num_channels))
    if num_channels == 1:
        axes = [axes]
    
    for i, ch_idx in enumerate(channels_to_plot):
        if ch_idx >= len(normal_df.columns):
            continue
        
        ax = axes[i]
        ch_name = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f"CH{ch_idx}"
        
        normal_vals = pd.to_numeric(normal_df.iloc[:, ch_idx], errors='coerce').dropna()
        test_vals = pd.to_numeric(test_df.iloc[:, ch_idx], errors='coerce').dropna()
        
        ax.hist(normal_vals, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(test_vals, bins=50, alpha=0.6, label='Test', color='red', density=True)
        
        ax.set_title(f'Distribution: Channel {ch_idx} - {ch_name}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Value')
    fig.suptitle('Data Distribution Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Distribution plot saved to '{output_filename}'")
    except Exception as e:
        print(f"ERROR: {e}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Channel Diagnostic Tool: Compare normal vs test data using raw statistics"
    )
    parser.add_argument('--normal_file', type=str, 
                       default=os.path.join(Config.TRAIN_DATA_PATH, 'x70_all.csv'),
                       help='Normal reference data (default: x70_all.csv from train set)')
    parser.add_argument('--test_file', type=str, required=True,
                       help=f'Test file name located in the test data directory ({Config.TEST_DATA_PATH})')
    parser.add_argument('--plot_channels', type=int, nargs='+', default=[11, 12, 15],
                       help='Channel indices to visualize (default: 11 12 15)')
    parser.add_argument('--output_file', type=str, default='raw_timeseries_comparison.png',
                       help='Output visualization filename')
    parser.add_argument('--stats_output', type=str, default='raw_channel_stats.csv',
                       help='Statistics output CSV filename')
    parser.add_argument('--dist_output', type=str, default='channel_distribution.png',
                       help='Distribution comparison plot filename')
    
    # C단계: 채널 제외 옵션 테스트. 실제 센서값(gyro/accel/mag/baro)만 비교하기 위함.
    parser.add_argument('--exclude_channels', type=int, nargs='+', default=[],
                       help='Channels to exclude from analysis (e.g., 4 5 9 10 14)')
    
    # D단계: 논문 Fig.7 시간 범위 지정. Fault가 명확한 시간 구간만 슬라이싱해서 비교.
    parser.add_argument('--time_range', type=int, nargs=2,
                       help='Time range [start end] for slicing data')
    
    parser.add_argument('--no_dist_plot', action='store_true',
                       help='Skip distribution plot generation')
    
    args = parser.parse_args()
    
    # 파일 경로 구성
    if os.path.isabs(args.normal_file):
        normal_file_path = args.normal_file
    else:
        normal_file_path = os.path.join(Config.BASE_DIR, args.normal_file)
        
    # config.py에 정의된 테스트 데이터 경로를 사용하여 파일 경로를 구성합니다.
    test_file_path = os.path.join(Config.TEST_DATA_PATH, args.test_file)
    print("\n" + "="*80)
    print("DRONE SENSOR CHANNEL DIAGNOSTIC TOOL")
    print("="*80)
    print(f"Normal reference: {normal_file_path}")
    print(f"Test data: {test_file_path}")
    print(f"Plot channels: {args.plot_channels}")
    if args.exclude_channels:
        print(f"Excluded channels: {args.exclude_channels}")
    print("="*80)
    
    # 데이터 로드 (두 가지 버전)
    # 1. 원본 데이터 (플롯용 - 채널 제외 안 함)
    original_normal_df = load_data(normal_file_path, args.time_range)
    original_test_df = load_data(test_file_path, args.time_range)
    
    if original_normal_df is None or original_test_df is None:
        print("ERROR: Failed to load data")
        return
    
    # 이름 저장 (나중에 플롯 레이블용)
    original_normal_df.name = os.path.basename(normal_file_path)
    original_test_df.name = args.test_file
    
    # 2. 분석용 데이터 (채널 제외 적용)
    analysis_normal_df = load_data(normal_file_path, args.time_range, args.exclude_channels)
    analysis_test_df = load_data(test_file_path, args.time_range, args.exclude_channels)
    
    if analysis_normal_df is None or analysis_test_df is None:
        print("ERROR: Failed to process data")
        return
    
    # ========== A단계: 통계 비교 ==========
    stats_df = compare_channel_stats(
        analysis_normal_df, 
        analysis_test_df,
        args.stats_output,
        channel_names=Config.CHANNEL_NAMES
    )
    
    # ========== B단계: Time-series 플롯 ==========
    plot_channel_timeseries(
        original_normal_df,
        original_test_df,
        args.plot_channels,
        args.output_file,
        channel_names=Config.CHANNEL_NAMES
    )
    
    # ========== 보너스: 분포 비교 ==========
    if not args.no_dist_plot:
        plot_channel_distribution(
            original_normal_df,
            original_test_df,
            args.plot_channels,
            args.dist_output,
            channel_names=Config.CHANNEL_NAMES
        )
    
    print("\n" + "="*80)
    print("✓ Diagnostic complete!")
    print("="*80)


if __name__ == '__main__':
    main()
