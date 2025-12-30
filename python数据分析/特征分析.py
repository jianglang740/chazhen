import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（如果系统有中文字体）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('ETH_USDT_15m_20251227_164447.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# 计算实体长度、上下影线长度
df['实体长度'] = abs(df['close'] - df['open'])
df['上影线长度'] = df['high'] - np.maximum(df['open'], df['close'])
df['下影线长度'] = np.minimum(df['open'], df['close']) - df['low']

# 处理实体长度为0的情况（避免除以0）
df.loc[df['实体长度'] == 0, '实体长度'] = 0.001

# 计算影线比率
df['上影线比率'] = df['上影线长度'] / df['实体长度']
df['下影线比率'] = df['下影线长度'] / df['实体长度']

# 识别插针（比率 > 2）
df['向上插针'] = df['上影线比率'] > 2
df['向下插针'] = df['下影线比率'] > 2
df['插针类型'] = '无'
df.loc[df['向上插针'], '插针类型'] = '向上'
df.loc[df['向下插针'], '插针类型'] = '向下'
# 如果同时满足（理论上很少），标记为双向
df.loc[df['向上插针'] & df['向下插针'], '插针类型'] = '双向'

# 统计插针次数
total_candles = len(df)
up_pins = df['向上插针'].sum()
down_pins = df['向下插针'].sum()
total_pins = up_pins + down_pins

print(f"总K线数量: {total_candles}")
print(f"向上插针数量: {up_pins} ({up_pins/total_candles*100:.2f}%)")
print(f"向下插针数量: {down_pins} ({down_pins/total_candles*100:.2f}%)")
print(f"总插针数量: {total_pins} ({total_pins/total_candles*100:.2f}%)")

# 分析比率分布
bins = [(2, 3), (3, 4), (4, 5), (5, 10), (10, 20), (20, 50), (50, float('inf'))]
bin_labels = ['2-3', '3-4', '4-5', '5-10', '10-20', '20-50', '50+']

results = []
for i, (low, high) in enumerate(bins):
    if high == float('inf'):
        up_count = len(df[(df['上影线比率'] >= low) & df['向上插针']])
        down_count = len(df[(df['下影线比率'] >= low) & df['向下插针']])
    else:
        up_count = len(df[(df['上影线比率'] >= low) & (df['上影线比率'] < high) & df['向上插针']])
        down_count = len(df[(df['下影线比率'] >= low) & (df['下影线比率'] < high) & df['向下插针']])
    
    total_count = up_count + down_count
    total_percentage = total_count / total_pins * 100 if total_pins > 0 else 0
    
    results.append({
        '比率区间': bin_labels[i],
        '向上插针数量': up_count,
        '向下插针数量': down_count,
        '总插针数量': total_count,
        '占总插针比例(%)': total_percentage
    })

# 创建比率分布DataFrame
distribution_df = pd.DataFrame(results)
print("\n插针比率分布统计:")
print(distribution_df)

# 输出详细插针记录到CSV
pin_records = df[df['插针类型'] != '无'].copy()
pin_records = pin_records[[
    'datetime', 'open', 'high', 'low', 'close', 'volume',
    '实体长度', '上影线长度', '下影线长度', '上影线比率', '下影线比率', '插针类型'
]]

# 保存插针记录
pin_records.to_csv('ETH_插针记录.csv', index=False, encoding='utf-8-sig')
print(f"\n已保存插针记录到 'ETH_插针记录.csv'，共 {len(pin_records)} 条记录")

# 创建简单的可视化图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 插针类型分布
pin_types = pin_records['插针类型'].value_counts()
axes[0, 0].pie(pin_types.values, labels=pin_types.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('插针类型分布')

# 2. 上影线比率分布（直方图）
up_pins_only = pin_records[pin_records['插针类型'].isin(['向上', '双向'])]
if len(up_pins_only) > 0:
    axes[0, 1].hist(up_pins_only['上影线比率'], bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('上影线比率')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('上影线比率分布')
    axes[0, 1].axvline(x=up_pins_only['上影线比率'].median(), color='green', linestyle='--', label=f'中位数: {up_pins_only["上影线比率"].median():.2f}')
    axes[0, 1].legend()

# 3. 下影线比率分布（直方图）
down_pins_only = pin_records[pin_records['插针类型'].isin(['向下', '双向'])]
if len(down_pins_only) > 0:
    axes[1, 0].hist(down_pins_only['下影线比率'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('下影线比率')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_title('下影线比率分布')
    axes[1, 0].axvline(x=down_pins_only['下影线比率'].median(), color='green', linestyle='--', label=f'中位数: {down_pins_only["下影线比率"].median():.2f}')
    axes[1, 0].legend()

# 4. 比率区间分布（柱状图）
axes[1, 1].bar(range(len(distribution_df)), distribution_df['总插针数量'], color='orange', alpha=0.7)
axes[1, 1].set_xlabel('比率区间')
axes[1, 1].set_ylabel('插针数量')
axes[1, 1].set_title('插针比率区间分布')
axes[1, 1].set_xticks(range(len(distribution_df)))
axes[1, 1].set_xticklabels(distribution_df['比率区间'], rotation=45)

# 在柱状图上添加数值标签
for i, v in enumerate(distribution_df['总插针数量']):
    axes[1, 1].text(i, v + 0.5, str(v), ha='center')

plt.tight_layout()
plt.savefig('ETH_插针分析图表.png', dpi=150, bbox_inches='tight')
print(f"已保存分析图表到 'ETH_插针分析图表.png'")

# 显示基本统计信息
print("\n=== 基本统计信息 ===")
print(f"上影线比率中位数: {df['上影线比率'].median():.2f}")
print(f"下影线比率中位数: {df['下影线比率'].median():.2f}")
print(f"上影线比率最大值: {df['上影线比率'].max():.2f}")
print(f"下影线比率最大值: {df['下影线比率'].max():.2f}")

if len(up_pins_only) > 0:
    print(f"\n向上插针的上影线比率中位数: {up_pins_only['上影线比率'].median():.2f}")
    print(f"向上插针的上影线比率平均值: {up_pins_only['上影线比率'].mean():.2f}")

if len(down_pins_only) > 0:
    print(f"\n向下插针的下影线比率中位数: {down_pins_only['下影线比率'].median():.2f}")
    print(f"向下插针的下影线比率平均值: {down_pins_only['下影线比率'].mean():.2f}")

# 输出比率集中区间分析
print("\n=== 比率集中区间分析 ===")
# 找到占比最大的区间
max_row = distribution_df.loc[distribution_df['总插针数量'].idxmax()]
print(f"最集中的比率区间: {max_row['比率区间']}")
print(f"该区间插针数量: {max_row['总插针数量']} (占所有插针的{max_row['占总插针比例(%)']:.1f}%)")

# 计算累计占比
cumulative_percent = 0
print("\n累计占比分析:")
for i, row in distribution_df.iterrows():
    cumulative_percent += row['占总插针比例(%)']
    print(f"比率≤{row['比率区间'].split('-')[-1] if '+' not in row['比率区间'] else '50'}: {cumulative_percent:.1f}%")
    if cumulative_percent >= 80:
        print(f"★ 80%的插针集中在比率≤{row['比率区间'].split('-')[-1] if '+' not in row['比率区间'] else '50'}的区间内")
        break