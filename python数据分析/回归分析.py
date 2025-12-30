import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
df = pd.read_csv('/Users/clinking/Desktop/python文件夹/python数据分析/ETH_插针记录.csv')

# 过滤掉双向插针，只保留单向插针（向上或向下）
single_pin_df = df[df['插针类型'] != '双向'].copy()

print(f"总数据量: {len(df)}")
print(f"单向插针数据量: {len(single_pin_df)}")
print(f"向上插针: {len(single_pin_df[single_pin_df['插针类型'] == '向上'])}")
print(f"向下插针: {len(single_pin_df[single_pin_df['插针类型'] == '向下'])}")

# 为不同类型插针添加标记
single_pin_df['插针类型编码'] = single_pin_df['插针类型'].apply(lambda x: 1 if x == '向上' else 0)

# 准备特征和目标变量
# 对于向上插针，我们关注上影线长度；对于向下插针，我们关注下影线长度
# 我们创建两个不同的数据集进行分析

# 1. 分析向上插针
up_pin_df = single_pin_df[single_pin_df['插针类型'] == '向上'].copy()
up_X = up_pin_df[['volume']].values  # 特征：成交量
up_y = up_pin_df['上影线长度'].values  # 目标：上影线长度

# 2. 分析向下插针
down_pin_df = single_pin_df[single_pin_df['插针类型'] == '向下'].copy()
down_X = down_pin_df[['volume']].values  # 特征：成交量
down_y = down_pin_df['下影线长度'].values  # 目标：下影线长度

print("\n=== 向上插针分析 ===")
print(f"样本数量: {len(up_pin_df)}")

if len(up_pin_df) > 1:
    # 创建并训练线性回归模型
    up_model = LinearRegression()
    up_model.fit(up_X, up_y)
    
    # 预测
    up_y_pred = up_model.predict(up_X)
    
    # 评估模型
    up_r2 = r2_score(up_y, up_y_pred)
    up_corr = up_pin_df['volume'].corr(up_pin_df['上影线长度'])
    
    print(f"R²分数: {up_r2:.4f}")
    print(f"相关系数: {up_corr:.4f}")
    print(f"系数 (斜率): {up_model.coef_[0]:.6f}")
    print(f"截距: {up_model.intercept_:.4f}")
    
    # 判断相关性强度
    if abs(up_corr) > 0.7:
        print("结论: 高度相关")
    elif abs(up_corr) > 0.4:
        print("结论: 中度相关")
    elif abs(up_corr) > 0.2:
        print("结论: 低度相关")
    else:
        print("结论: 几乎不相关")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(up_X, up_y, alpha=0.6, label='实际数据')
    plt.plot(up_X, up_y_pred, color='red', linewidth=2, label=f'回归线 (R²={up_r2:.3f})')
    plt.xlabel('成交量 (Volume)')
    plt.ylabel('上影线长度')
    plt.title('向上插针：成交量与上影线长度的关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("向上插针数据不足，无法进行分析")

print("\n=== 向下插针分析 ===")
print(f"样本数量: {len(down_pin_df)}")

if len(down_pin_df) > 1:
    # 创建并训练线性回归模型
    down_model = LinearRegression()
    down_model.fit(down_X, down_y)
    
    # 预测
    down_y_pred = down_model.predict(down_X)
    
    # 评估模型
    down_r2 = r2_score(down_y, down_y_pred)
    down_corr = down_pin_df['volume'].corr(down_pin_df['下影线长度'])
    
    print(f"R²分数: {down_r2:.4f}")
    print(f"相关系数: {down_corr:.4f}")
    print(f"系数 (斜率): {down_model.coef_[0]:.6f}")
    print(f"截距: {down_model.intercept_:.4f}")
    
    # 判断相关性强度
    if abs(down_corr) > 0.7:
        print("结论: 高度相关")
    elif abs(down_corr) > 0.4:
        print("结论: 中度相关")
    elif abs(down_corr) > 0.2:
        print("结论: 低度相关")
    else:
        print("结论: 几乎不相关")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(down_X, down_y, alpha=0.6, label='实际数据')
    plt.plot(down_X, down_y_pred, color='red', linewidth=2, label=f'回归线 (R²={down_r2:.3f})')
    plt.xlabel('成交量 (Volume)')
    plt.ylabel('下影线长度')
    plt.title('向下插针：成交量与下影线长度的关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("向下插针数据不足，无法进行分析")

# 3. 综合分析：将两种类型合并，使用虚拟变量
print("\n=== 综合分析 (包含插针类型) ===")

# 准备合并数据
combined_df = single_pin_df.copy()

# 创建目标变量：对于向上插针使用上影线长度，对于向下插针使用下影线长度
combined_df['影线长度'] = combined_df.apply(
    lambda row: row['上影线长度'] if row['插针类型'] == '向上' else row['下影线长度'], 
    axis=1
)

# 特征：成交量和插针类型编码
combined_X = combined_df[['volume', '插针类型编码']].values
combined_y = combined_df['影线长度'].values

if len(combined_df) > 2:
    # 创建并训练线性回归模型
    combined_model = LinearRegression()
    combined_model.fit(combined_X, combined_y)
    
    # 预测
    combined_y_pred = combined_model.predict(combined_X)
    
    # 评估模型
    combined_r2 = r2_score(combined_y, combined_y_pred)
    
    print(f"总样本数量: {len(combined_df)}")
    print(f"R²分数: {combined_r2:.4f}")
    print(f"成交量系数: {combined_model.coef_[0]:.6f}")
    print(f"插针类型系数: {combined_model.coef_[1]:.4f}")
    print(f"截距: {combined_model.intercept_:.4f}")
    
    # 计算整体相关性
    overall_corr = np.corrcoef(combined_df['volume'], combined_df['影线长度'])[0, 1]
    print(f"整体相关系数: {overall_corr:.4f}")
    
    # 判断整体相关性强度
    if abs(overall_corr) > 0.7:
        print("整体结论: 高度相关")
    elif abs(overall_corr) > 0.4:
        print("整体结论: 中度相关")
    elif abs(overall_corr) > 0.2:
        print("整体结论: 低度相关")
    else:
        print("整体结论: 几乎不相关")
    
    # 可视化：按类型着色
    plt.figure(figsize=(12, 8))
    
    # 向上插针
    up_mask = combined_df['插针类型'] == '向上'
    plt.scatter(combined_df.loc[up_mask, 'volume'], 
                combined_df.loc[up_mask, '影线长度'], 
                alpha=0.6, color='green', label='向上插针')
    
    # 向下插针
    down_mask = combined_df['插针类型'] == '向下'
    plt.scatter(combined_df.loc[down_mask, 'volume'], 
                combined_df.loc[down_mask, '影线长度'], 
                alpha=0.6, color='red', label='向下插针')
    
    # 为每种类型绘制回归线
    # 向上插针回归线
    if len(up_pin_df) > 1:
        up_X_sorted = np.sort(up_pin_df['volume'].values).reshape(-1, 1)
        up_y_pred_sorted = up_model.predict(up_X_sorted)
        plt.plot(up_X_sorted, up_y_pred_sorted, color='darkgreen', 
                linewidth=2, linestyle='--', label='向上插针回归线')
    
    # 向下插针回归线
    if len(down_pin_df) > 1:
        down_X_sorted = np.sort(down_pin_df['volume'].values).reshape(-1, 1)
        down_y_pred_sorted = down_model.predict(down_X_sorted)
        plt.plot(down_X_sorted, down_y_pred_sorted, color='darkred', 
                linewidth=2, linestyle='--', label='向下插针回归线')
    
    plt.xlabel('成交量 (Volume)')
    plt.ylabel('影线长度')
    plt.title('单向插针：成交量与影线长度的关系 (按类型)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 打印统计摘要
    print("\n=== 统计摘要 ===")
    print("向上插针:")
    print(f"  平均成交量: {up_pin_df['volume'].mean():.2f}")
    print(f"  平均上影线长度: {up_pin_df['上影线长度'].mean():.2f}")
    print(f"  成交量标准差: {up_pin_df['volume'].std():.2f}")
    
    print("\n向下插针:")
    print(f"  平均成交量: {down_pin_df['volume'].mean():.2f}")
    print(f"  平均下影线长度: {down_pin_df['下影线长度'].mean():.2f}")
    print(f"  成交量标准差: {down_pin_df['volume'].std():.2f}")
else:
    print("综合数据不足，无法进行分析")