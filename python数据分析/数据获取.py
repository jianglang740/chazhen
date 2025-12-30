import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os

def fetch_eth_15m_data_with_proxy():
    """
    使用代理获取以太坊近三年的15分钟K线数据
    """
    # 配置代理（使用您的Clash端口7897）
    proxy_config = {
        'http': 'http://127.0.0.1:7897',
        'https': 'http://127.0.0.1:7897'
    }
    
    # 创建币安交易所实例并配置代理
    exchange = ccxt.binance({
        'enableRateLimit': True,  # 启用速率限制以避免被限制[citation:6]
        'proxies': proxy_config,
        'timeout': 30000,  # 30秒超时
    })
    
    # 交易对符号
    symbol = 'ETH/USDT'
    
    # 计算时间范围（近三年）
    end_time = datetime.now()
    start_time = end_time - timedelta(days=3*365)  # 三年
    
    print(f"开始获取数据，时间范围：{start_time} 到 {end_time}")
    print(f"交易对：{symbol}")
    print(f"时间间隔：15分钟")
    
    # 转换时间为毫秒时间戳
    since = int(start_time.timestamp() * 1000)
    current_time = int(end_time.timestamp() * 1000)
    
    all_ohlcv = []  # 存储所有数据
    
    # 分批次获取数据（每次获取1000根K线）
    batch_size = 1000  # 币安API每次最多返回1000条记录
    batch_count = 0
    
    while since < current_time:
        try:
            print(f"正在获取批次 {batch_count + 1}...")
            
            # 获取OHLCV数据
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='15m',
                since=since,
                limit=batch_size
            )
            
            if not ohlcv:
                print("没有更多数据，结束获取")
                break
            
            # 将数据添加到总列表
            all_ohlcv.extend(ohlcv)
            
            # 更新起始时间（使用最后一条数据的时间戳+1）
            since = ohlcv[-1][0] + 1  # 时间戳加1毫秒避免重复
            
            batch_count += 1
            
            # 打印进度
            last_time = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"  已获取 {len(ohlcv)} 条记录，最后时间：{last_time}")
            print(f"  总记录数：{len(all_ohlcv)}")
            
            # 遵守API速率限制
            time.sleep(exchange.rateLimit / 1000)  # 转换为秒
            
        except ccxt.NetworkError as e:
            print(f"网络错误: {e}, 等待10秒后重试...")
            time.sleep(10)
        except ccxt.ExchangeError as e:
            print(f"交易所错误: {e}, 等待30秒后重试...")
            time.sleep(30)
        except Exception as e:
            print(f"未知错误: {e}, 等待5秒后重试...")
            time.sleep(5)
    
    return all_ohlcv, symbol

def save_to_csv(ohlcv_data, symbol):
    """
    将OHLCV数据保存为CSV文件
    """
    if not ohlcv_data:
        print("没有数据可保存")
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 转换时间戳为可读格式
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 设置datetime为索引
    df.set_index('datetime', inplace=True)
    
    # 删除原始timestamp列（可选）
    df.drop('timestamp', axis=1, inplace=True)
    
    # 创建文件名
    safe_symbol = symbol.replace('/', '_')
    filename = f"ETH_USDT_15m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # 保存为CSV
    df.to_csv(filename, index=True)
    
    print(f"数据已保存到: {filename}")
    print(f"总记录数: {len(df)}")
    print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
    
    return filename, df

def main():
    """主函数"""
    print("=" * 60)
    print("以太坊15分钟K线数据下载器")
    print("=" * 60)
    
    try:
        # 获取数据
        ohlcv_data, symbol = fetch_eth_15m_data_with_proxy()
        
        if not ohlcv_data:
            print("未能获取到任何数据")
            return
        
        # 保存数据
        filename, df = save_to_csv(ohlcv_data, symbol)
        
        if df is not None:
            # 显示数据摘要
            print("\n数据摘要:")
            print("-" * 40)
            print(df.head())
            print(f"\n...\n")
            print(df.tail())
            
            print("\n基本统计信息:")
            print("-" * 40)
            print(f"开盘价统计: 均值={df['open'].mean():.2f}, 标准差={df['open'].std():.2f}")
            print(f"收盘价统计: 均值={df['close'].mean():.2f}, 标准差={df['close'].std():.2f}")
            print(f"成交量统计: 均值={df['volume'].mean():.2f}, 最大值={df['volume'].max():.2f}")
            
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()