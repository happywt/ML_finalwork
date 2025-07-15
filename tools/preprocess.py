import pandas as pd
import numpy as np
import os

# 加载数据并进行清洗处理
def preprocess_data(file_path):
    columns = [
        'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]
    
    has_header = 'train' in file_path
    data = pd.read_csv(
        file_path,
        names=columns,
        header=0 if has_header else None,
        index_col='DateTime',
        low_memory=False
    )

    # 转换索引为时间戳格式
    data.index = pd.to_datetime(data.index)

    # 缺失值处理：标记符替换为空值
    data.replace('?', np.nan, inplace=True)

    # 时间偏移填补缺失值
    for shift_val in [1440, 2880, -1440, -2880]:
        data.fillna(data.shift(shift_val), inplace=True)

    # 统一数据类型
    data = data.astype('float32')

    # 单位调整
    data['RR'] /= 10

    # 计算未计入的电量
    total_energy = (data['Global_active_power'] * 1000 / 60)
    known_subs = data['Sub_metering_1'] + data['Sub_metering_2'] + data['Sub_metering_3']
    residual_energy = total_energy - known_subs
    insert_loc = data.columns.get_loc('Sub_metering_3') + 1
    data.insert(insert_loc, 'Sub_metering_remainder', residual_energy)

    return data


# 将数据按天重构
def transform_to_daily(data):
    daily_data = {}

    # 按天汇总总和字段
    sum_cols = ["Global_active_power", "Global_reactive_power", "Sub_metering_1",
                "Sub_metering_2", "Sub_metering_3", "Sub_metering_remainder"]
    for col in sum_cols:
        daily_data[col] = data[col].resample("D").sum()

    # 计算日平均值
    mean_cols = ["Voltage", "Global_intensity"]
    for col in mean_cols:
        daily_data[col] = data[col].resample("D").mean()

    # 获取每日首个观测值
    first_cols = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
    for col in first_cols:
        daily_data[col] = data[col].resample("D").first()

    return pd.DataFrame(daily_data)


if __name__ == "__main__":
    # 读取与预处理数据
    train_raw = preprocess_data("/data/wt/study/machine_learning/dataset/train.csv")
    test_raw = preprocess_data("/data/wt/study/machine_learning/dataset/test.csv")

    # 数据转换，改为按天统计的数据集
    daily_train = transform_to_daily(train_raw)
    daily_test = transform_to_daily(test_raw)

    # 写入结果文件
    daily_train.to_csv("/data/wt/study/machine_learning/data/processed_train.csv")
    daily_test.to_csv("/data/wt/study/machine_learning/data/processed_test.csv")
