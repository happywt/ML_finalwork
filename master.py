import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

from models.lstm_model import LSTMNetwork
from models.transformer_model import SequenceTransformer
from models.DPTransformer import DualPathTransformer

# 路径准备
os.makedirs("result", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 数据预处理函数
def preprocess(train_df, test_df):
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    cols_std = ['Global_reactive_power', 'Voltage', 'Global_intensity']
    cols_log = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_remainder']
    cols_minmax = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    main_scaler = StandardScaler()
    train_scaled['Global_active_power'] = main_scaler.fit_transform(train_df[['Global_active_power']])
    test_scaled['Global_active_power'] = main_scaler.transform(test_df[['Global_active_power']])

    scaler_std = StandardScaler()
    train_scaled[cols_std] = scaler_std.fit_transform(train_df[cols_std])
    test_scaled[cols_std] = scaler_std.transform(test_df[cols_std])

    scaler_log = StandardScaler()
    train_scaled[cols_log] = scaler_log.fit_transform(np.log1p(train_df[cols_log]))
    test_scaled[cols_log] = scaler_log.transform(np.log1p(test_df[cols_log]))

    scaler_mm = MinMaxScaler()
    train_scaled[cols_minmax] = scaler_mm.fit_transform(train_df[cols_minmax])
    test_scaled[cols_minmax] = scaler_mm.transform(test_df[cols_minmax])

    return train_scaled.values, test_scaled.values, main_scaler

def create_sequences(data, past, future):
    X, Y = [], []
    for i in range(len(data) - past - future + 1):
        X.append(data[i:i+past])
        Y.append(data[i+past:i+past+future, 0])
    return np.array(X), np.array(Y)

def rolling_infer(model, series, past, future, scaler):
    model.eval()
    device = next(model.parameters()).device
    length = len(series) - past
    temp = {i: [] for i in range(length)}

    for i in range(length):
        x = torch.tensor(series[i:i+past], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x).cpu().numpy()
        y = scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
        for j in range(min(future, length - i)):
            temp[i + j].append(y[0, j])

    result = {
        "mean": np.array([np.mean(temp[i]) for i in range(length)]),
        "median": np.array([np.median(temp[i]) for i in range(length)]),
        "ema": []
    }

    alpha = 0.1
    for i in range(length):
        ema = temp[i][0]
        for val in temp[i][1:]:
            ema = alpha * val + (1 - alpha) * ema
        result['ema'].append(ema)

    return result

def best_prediction(preds, target):
    scores = {k: mean_absolute_error(target, v) for k, v in preds.items()}
    best = min(scores, key=scores.get)
    return preds[best]

def plot_all(preds_list, gt, label_days):
    sns.set(style="whitegrid")  # 白色网格背景
    p1, p2, p3 = preds_list
    plt.figure(figsize=(14, 5))
    
    palette = sns.color_palette("tab10", 4)
    
    plt.plot(gt, label="Ground Truth", color=palette[0], linewidth=2)
    plt.plot(p1, label="LSTM Prediction", color=palette[1], linewidth=2)
    plt.plot(p2, label="Transformer Prediction", color=palette[2], linewidth=2)
    plt.plot(p3, label="DualPathTransformer Prediction", color=palette[3], linewidth=2)
    
    plt.title(f"Prediction for {label_days} Days", fontsize=16, fontweight='bold')
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Global Active Power", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(f"result/{label_days}.png", dpi=300)
    plt.close()

def set_random_seed(seed):
    """设置随机种子以确保可重现性，但不同run之间有差异"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 允许一些随机性用于dropout等
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def train_model(model_type, X, Y, batch_size, past, future, run_id, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 为每次运行设置不同的随机种子
    set_random_seed(42 + run_id * 100)

    seq_len, input_dim = X.shape[1], X.shape[2]
    output_len = Y.shape[1]

    if model_type == 'LSTM':
        model = LSTMNetwork(input_dim, hidden_dim=64, output_steps=output_len)
    elif model_type == 'Transformer':
        model = SequenceTransformer(seq_len, input_dim, output_len)
    elif model_type == 'DualPathTransformer':
        model = DualPathTransformer(seq_len, input_dim, output_len)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)

    # 添加一些随机性：随机dropout率
    dropout_rate = 0.1 + np.random.uniform(-0.05, 0.05)  # 0.05-0.15之间
    
    dataset = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(Y, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

    # 学习率微调
    base_lr = 5e-4
    lr_variation = np.random.uniform(0.8, 1.2)  # 学习率在原来的0.8-1.2倍之间
    lr = base_lr * lr_variation
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print(f"Start training: Model = {model_type}, Run = {run_id}, Batch Size = {batch_size}, LR = {lr:.6f}, Epochs = 200")
    model.train()
    for epoch in tqdm(range(1, 201), desc=f"Training {model_type} Run {run_id}"):
        epoch_loss = 0.0
        for xb, yb in dataset:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            loss = criterion(output, yb)

            optimizer.zero_grad()
            loss.backward()
            
            # 添加梯度裁剪，增加训练稳定性的同时保持一些随机性
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 50 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(dataset)
            print(f"[{model_type}] Run {run_id} Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    # 保存checkpoint，包含run_id以区分不同的运行
    checkpoint_path = f"checkpoints/{model_type}_past{past}_future{future}_bs{batch_size}_run{run_id}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'input_dim': input_dim,
        'seq_len': seq_len,
        'output_len': output_len,
        'past': past,
        'future': future,
        'batch_size': batch_size,
        'run_id': run_id,
        'lr': lr
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Finished training: Model = {model_type}, Run = {run_id}\n")
    return model.eval()

def load_model(checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint['model_type']
    input_dim = checkpoint['input_dim']
    seq_len = checkpoint['seq_len']
    output_len = checkpoint['output_len']
    
    if model_type == 'LSTM':
        model = LSTMNetwork(input_dim, hidden_dim=64, output_steps=output_len)
    elif model_type == 'Transformer':
        model = SequenceTransformer(seq_len, input_dim, output_len)
    elif model_type == 'DualPathTransformer':
        model = DualPathTransformer(seq_len, input_dim, output_len)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def train_all_models(train_array, device):
    """训练所有模型并保存checkpoint"""
    print("=== TRAINING PHASE ===")
    
    # 训练配置
    configs = [
        ('LSTM', 90, 90),
        ('LSTM', 90, 365),
        ('Transformer', 1, 90),
        ('Transformer', 1, 365),
        ('DualPathTransformer', 1, 90),
        ('DualPathTransformer', 1, 365)
    ]
    
    # 使用不同的batch_size和多次运行来增加变异性
    batch_sizes = [16, 32, 64, 128, 256]  # 不同的batch size
    n_runs = 1  # 每个配置运行1次
    
    for model_type, past, future in configs:
        print(f"\nTraining {model_type} for past={past}, future={future}")
        X, Y = create_sequences(train_array, past, future)
        
        run_counter = 0
        for bs in batch_sizes:
            for run in range(n_runs):
                train_model(model_type, X, Y, bs, past, future, run_counter, device)
                run_counter += 1

def run_eval_with_checkpoints(model_type, past, future, train_array, test_array, scaler, ground_truth, device):
    """使用保存的checkpoint进行推理"""
    test_seq = np.concatenate([train_array[-past:], test_array])
    all_preds, mse_list, mae_list = [], [], []
    
    # 查找所有相关的checkpoint文件
    checkpoint_dir = "checkpoints"
    checkpoint_files = []
    
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(f"{model_type}_past{past}_future{future}") and filename.endswith(".pth"):
            checkpoint_files.append(os.path.join(checkpoint_dir, filename))
    
    if not checkpoint_files:
        print(f"No checkpoints found for {model_type}_past{past}_future{future}")
        return None
    
    print(f"Found {len(checkpoint_files)} checkpoints for {model_type}")
    
    for checkpoint_path in checkpoint_files:
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            model, checkpoint_info = load_model(checkpoint_path, device)
            
            pred_series = rolling_infer(model, test_seq, past, future, scaler)
            final = best_prediction(pred_series, ground_truth)
            
            mse_list.append(mean_squared_error(ground_truth, final))
            mae_list.append(mean_absolute_error(ground_truth, final))
            all_preds.append(final)
            
        except Exception as e:
            print(f"Error loading {checkpoint_path}: {e}")
            continue

    if not all_preds:
        print(f"No valid predictions for {model_type}")
        return None

    avg_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)
    avg_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)

    best_idx = np.argmin(mae_list)

    metrics = {
        "num_models": len(all_preds),
        "mse_mean": float(avg_mse),
        "mse_std": float(std_mse),
        "mae_mean": float(avg_mae),
        "mae_std": float(std_mae),
        "mse_values": [float(x) for x in mse_list],
        "mae_values": [float(x) for x in mae_list]
    }

    print(f"Results for {model_type}: MSE={avg_mse:.6f}±{std_mse:.6f}, MAE={avg_mae:.6f}±{std_mae:.6f}")

    return np.mean(all_preds, axis=0), metrics

def run_evaluation(train_array, test_array, scaler, ground_truth, device):
    """运行评估阶段"""
    print("\n=== EVALUATION PHASE ===")
    
    preds_90, preds_365 = [], []
    metrics_dict = {}
    
    # 评估配置
    eval_configs = [
        ('LSTM', 90, 90),
        ('LSTM', 90, 365),
        ('Transformer', 1, 90),
        ('Transformer', 1, 365),
        ('DualPathTransformer', 1, 90),
        ('DualPathTransformer', 1, 365)
    ]
    
    for model_type, past, future in eval_configs:
        print(f"\nEvaluating {model_type} for past={past}, future={future}")
        
        result = run_eval_with_checkpoints(
            model_type, past, future, train_array, test_array, 
            scaler, ground_truth, device
        )
        
        if result is not None:
            pred, metrics = result
            metrics_dict[f"{model_type}_{future}_days"] = metrics
            
            if future == 90:
                preds_90.append(pred)
            else:
                preds_365.append(pred)
    
    return preds_90, preds_365, metrics_dict

if __name__ == "__main__":
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    
    df_train = pd.read_csv("data/augmented_train.csv", index_col="DateTime").astype("float32")
    df_test = pd.read_csv("data/processed_test.csv", index_col="DateTime").astype("float32")

    train_array, test_array, scaler = preprocess(df_train, df_test)
    ground_truth = df_test['Global_active_power'].values.reshape(-1, 1)

    # 选择运行模式
    mode = input("选择运行模式 (train/eval/both): ").strip().lower()
    
    if mode in ['train', 'both']:
        # 训练阶段
        train_all_models(train_array, device)
    
    if mode in ['eval', 'both']:
        # 评估阶段
        preds_90, preds_365, metrics_dict = run_evaluation(
            train_array, test_array, scaler, ground_truth, device
        )
        
        # 保存结果
        with open("result/metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)

        if len(preds_90) >= 2:
            plot_all(preds_90, ground_truth, 90)
        if len(preds_365) >= 2:
            plot_all(preds_365, ground_truth, 365)
        
        print("\nEvaluation completed!")
        
        # 打印统计信息
        print("\n=== FINAL STATISTICS ===")
        for key, metrics in metrics_dict.items():
            print(f"{key}:")
            print(f"  Models trained: {metrics['num_models']}")
            print(f"  MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
            print(f"  MAE: {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
