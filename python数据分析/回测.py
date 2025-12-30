import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('ETH_插针记录.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# 筛选条件：插针比率在20～30之间，成交量在15000～35000 USDT之间
filtered_df = df[
    ((df['上影线比率'] >= 10) & (df['上影线比率'] <= 30)) | 
    ((df['下影线比率'] >= 10) & (df['下影线比率'] <= 30))
]
filtered_df = filtered_df[
    (filtered_df['volume'] >= 4000) & (filtered_df['volume'] <= 35000)
]
filtered_df = filtered_df[filtered_df['插针类型'].isin(['向上', '向下'])]

print(f"符合条件的插针数量: {len(filtered_df)}")
print(filtered_df[['datetime', '插针类型', '上影线比率', '下影线比率', 'volume']].head())

# 回测参数
initial_capital = 1000000  # 初始资金 100,000 USDT
position_size = 10  # 固定手数 10 ETH
take_profit_pct = 0.01  # 止盈 3%
stop_loss_pct = 0.035  # 止损 4.5%

# 回测结果存储
results = []
capital = initial_capital
trades = []  # 记录所有交易
current_position = None  # 当前持仓
position_entry_price = 0  # 持仓入场价格
position_entry_time = None  # 持仓入场时间
position_type = None  # 持仓类型: 'long' 或 'short'

# 净值记录
equity_curve = [{'datetime': df.iloc[0]['datetime'], 'equity': initial_capital}]

# 遍历所有K线进行回测
for i in range(len(df)):
    current_time = df.iloc[i]['datetime']
    current_open = df.iloc[i]['open']
    current_high = df.iloc[i]['high']
    current_low = df.iloc[i]['low']
    current_close = df.iloc[i]['close']
    
    # 检查是否有符合条件的插针信号
    signal_row = filtered_df[filtered_df['datetime'] == current_time]
    
    # 如果有持仓，检查是否达到止盈止损
    if current_position:
        if position_type == 'long':  # 多头持仓
            # 检查止盈止损
            take_profit_price = position_entry_price * (1 + take_profit_pct)
            stop_loss_price = position_entry_price * (1 - stop_loss_pct)
            
            # 检查是否在K线范围内触发了止盈止损
            hit_take_profit = current_high >= take_profit_price
            hit_stop_loss = current_low <= stop_loss_price
            
            if hit_take_profit or hit_stop_loss:
                # 确定平仓价格
                if hit_take_profit and hit_stop_loss:
                    # 两者都触发，取先达到的
                    high_ratio = (take_profit_price - position_entry_price) / (current_high - current_low) if current_high > current_low else 0.5
                    low_ratio = (position_entry_price - stop_loss_price) / (current_high - current_low) if current_high > current_low else 0.5
                    
                    if high_ratio <= low_ratio:  # 止盈先触发
                        exit_price = take_profit_price
                        exit_type = '止盈'
                    else:  # 止损先触发
                        exit_price = stop_loss_price
                        exit_type = '止损'
                elif hit_take_profit:
                    exit_price = take_profit_price
                    exit_type = '止盈'
                else:
                    exit_price = stop_loss_price
                    exit_type = '止损'
                
                # 计算盈亏
                pnl = position_size * (exit_price - position_entry_price)
                capital += pnl
                
                # 记录交易
                trades.append({
                    'entry_time': position_entry_time,
                    'exit_time': current_time,
                    'entry_price': position_entry_price,
                    'exit_price': exit_price,
                    'position_type': position_type,
                    'pnl': pnl,
                    'exit_type': exit_type
                })
                
                # 平仓
                current_position = None
                position_type = None
                print(f"{current_time}: 平仓多头, 入场价: {position_entry_price:.2f}, 平仓价: {exit_price:.2f}, 盈亏: {pnl:.2f}, {exit_type}")
        
        elif position_type == 'short':  # 空头持仓
            # 检查止盈止损
            take_profit_price = position_entry_price * (1 - take_profit_pct)
            stop_loss_price = position_entry_price * (1 + stop_loss_pct)
            
            # 检查是否在K线范围内触发了止盈止损
            hit_take_profit = current_low <= take_profit_price
            hit_stop_loss = current_high >= stop_loss_price
            
            if hit_take_profit or hit_stop_loss:
                # 确定平仓价格
                if hit_take_profit and hit_stop_loss:
                    # 两者都触发，取先达到的
                    low_ratio = (position_entry_price - take_profit_price) / (current_high - current_low) if current_high > current_low else 0.5
                    high_ratio = (stop_loss_price - position_entry_price) / (current_high - current_low) if current_high > current_low else 0.5
                    
                    if low_ratio <= high_ratio:  # 止盈先触发
                        exit_price = take_profit_price
                        exit_type = '止盈'
                    else:  # 止损先触发
                        exit_price = stop_loss_price
                        exit_type = '止损'
                elif hit_take_profit:
                    exit_price = take_profit_price
                    exit_type = '止盈'
                else:
                    exit_price = stop_loss_price
                    exit_type = '止损'
                
                # 计算盈亏
                pnl = position_size * (position_entry_price - exit_price)
                capital += pnl
                
                # 记录交易
                trades.append({
                    'entry_time': position_entry_time,
                    'exit_time': current_time,
                    'entry_price': position_entry_price,
                    'exit_price': exit_price,
                    'position_type': position_type,
                    'pnl': pnl,
                    'exit_type': exit_type
                })
                
                # 平仓
                current_position = None
                position_type = None
                print(f"{current_time}: 平仓空头, 入场价: {position_entry_price:.2f}, 平仓价: {exit_price:.2f}, 盈亏: {pnl:.2f}, {exit_type}")
    
    # 如果没有持仓，检查是否有新信号
    if not current_position and not signal_row.empty:
        signal = signal_row.iloc[0]
        signal_type = signal['插针类型']
        
        if signal_type == '向下':  # 下插针，做多
            entry_price = current_low  # 在底部做多
            position_type = 'long'
            current_position = position_size
            position_entry_price = entry_price
            position_entry_time = current_time
            
            # 记录开仓
            print(f"{current_time}: 开仓多头, 价格: {entry_price:.2f}, 资金: {capital:.2f}")
            
        elif signal_type == '向上':  # 上插针，做空
            entry_price = current_high  # 在顶部做空
            position_type = 'short'
            current_position = position_size
            position_entry_price = entry_price
            position_entry_time = current_time
            
            # 记录开仓
            print(f"{current_time}: 开仓空头, 价格: {entry_price:.2f}, 资金: {capital:.2f}")
    
    # 记录当前净值
    if current_position:
        if position_type == 'long':
            current_equity = capital + position_size * (current_close - position_entry_price)
        else:  # short
            current_equity = capital + position_size * (position_entry_price - current_close)
    else:
        current_equity = capital
    
    equity_curve.append({'datetime': current_time, 'equity': current_equity})

# 如果有持仓未平，强制平仓
if current_position:
    last_close = df.iloc[-1]['close']
    if position_type == 'long':
        pnl = position_size * (last_close - position_entry_price)
        exit_type = '强制平仓'
    else:  # short
        pnl = position_size * (position_entry_price - last_close)
        exit_type = '强制平仓'
    
    capital += pnl
    
    trades.append({
        'entry_time': position_entry_time,
        'exit_time': df.iloc[-1]['datetime'],
        'entry_price': position_entry_price,
        'exit_price': last_close,
        'position_type': position_type,
        'pnl': pnl,
        'exit_type': exit_type
    })
    
    print(f"强制平仓{position_type}, 入场价: {position_entry_price:.2f}, 平仓价: {last_close:.2f}, 盈亏: {pnl:.2f}")

# 分析结果
if trades:
    trades_df = pd.DataFrame(trades)
    
    # 计算总盈亏
    total_pnl = trades_df['pnl'].sum()
    total_return = (total_pnl / initial_capital) * 100
    
    # 计算胜率
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    
    # 计算平均盈亏
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    # 计算盈亏比
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf')
    
    print("\n" + "="*50)
    print("回测结果统计")
    print("="*50)
    print(f"初始资金: {initial_capital:.2f} USDT")
    print(f"最终资金: {capital:.2f} USDT")
    print(f"总盈亏: {total_pnl:.2f} USDT")
    print(f"总收益率: {total_return:.2f}%")
    print(f"交易次数: {len(trades_df)}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"盈利次数: {len(winning_trades)}")
    print(f"亏损次数: {len(losing_trades)}")
    print(f"平均盈利: {avg_win:.2f} USDT")
    print(f"平均亏损: {avg_loss:.2f} USDT")
    print(f"盈亏比: {profit_factor:.2f}")
    print(f"最大回撤: 待计算")
    
    # 按退出类型统计
    print("\n退出类型统计:")
    exit_stats = trades_df['exit_type'].value_counts()
    for exit_type, count in exit_stats.items():
        pnl_sum = trades_df[trades_df['exit_type'] == exit_type]['pnl'].sum()
        print(f"  {exit_type}: {count}次, 总盈亏: {pnl_sum:.2f}")
    
    # 按持仓类型统计
    print("\n持仓类型统计:")
    position_stats = trades_df['position_type'].value_counts()
    for pos_type, count in position_stats.items():
        pos_trades = trades_df[trades_df['position_type'] == pos_type]
        pos_pnl = pos_trades['pnl'].sum()
        pos_win_rate = len(pos_trades[pos_trades['pnl'] > 0]) / len(pos_trades) * 100 if len(pos_trades) > 0 else 0
        print(f"  {pos_type}: {count}次, 总盈亏: {pos_pnl:.2f}, 胜率: {pos_win_rate:.2f}%")
    
    # 显示交易明细
    print("\n交易明细:")
    for i, trade in enumerate(trades):
        print(f"{i+1}. {trade['position_type']} | 入场: {trade['entry_time']} @ {trade['entry_price']:.2f} | "
              f"出场: {trade['exit_time']} @ {trade['exit_price']:.2f} | "
              f"盈亏: {trade['pnl']:.2f} | {trade['exit_type']}")
else:
    print("没有执行任何交易")

# 绘制净值曲线
equity_df = pd.DataFrame(equity_curve)
equity_df = equity_df.drop_duplicates('datetime')

plt.figure(figsize=(14, 8))

# 净值曲线
plt.subplot(2, 1, 1)
plt.plot(equity_df['datetime'], equity_df['equity'], 'b-', linewidth=2, label='净值')
plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='初始资金')
plt.fill_between(equity_df['datetime'], initial_capital, equity_df['equity'], 
                 where=equity_df['equity'] >= initial_capital, 
                 facecolor='green', alpha=0.3)
plt.fill_between(equity_df['datetime'], initial_capital, equity_df['equity'], 
                 where=equity_df['equity'] < initial_capital, 
                 facecolor='red', alpha=0.3)
plt.title('ETH插针策略净值曲线', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('净值 (USDT)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 收益分布
plt.subplot(2, 2, 3)
if trades:
    pnl_values = [t['pnl'] for t in trades]
    colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
    plt.bar(range(len(pnl_values)), pnl_values, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('每笔交易盈亏分布', fontsize=12)
    plt.xlabel('交易序号', fontsize=10)
    plt.ylabel('盈亏 (USDT)', fontsize=10)
    plt.grid(True, alpha=0.3)

# 净值分布
plt.subplot(2, 2, 4)
equity_pct = ((equity_df['equity'] - initial_capital) / initial_capital * 100).values
plt.hist(equity_pct, bins=30, color='steelblue', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
plt.title('净值分布', fontsize=12)
plt.xlabel('收益率 (%)', fontsize=10)
plt.ylabel('频率', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 计算最大回撤
equity_series = equity_df.set_index('datetime')['equity']
rolling_max = equity_series.expanding().max()
drawdown = (equity_series - rolling_max) / rolling_max * 100
max_drawdown = drawdown.min()

print(f"\n最大回撤: {max_drawdown:.2f}%")
print(f"夏普比率: 待计算 (需要无风险收益率数据)")

# 保存结果到CSV
if trades:
    trades_df.to_csv('插针策略交易记录.csv', index=False, encoding='utf-8-sig')
    print("\n交易记录已保存到: 插针策略交易记录.csv")

equity_df.to_csv('插针策略净值曲线.csv', index=False, encoding='utf-8-sig')
print("净值曲线已保存到: 插针策略净值曲线.csv")