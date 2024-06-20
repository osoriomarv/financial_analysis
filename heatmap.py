import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from io import StringIO
import os
import random
import seaborn as sns
import matplotlib.dates as mdates

# Get the list of S&P 500 stocks
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
table = pd.read_html(url)
sp500 = table[0]
tickers = sp500['Symbol'].tolist()

# Ensure the local directory for saving images exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Function to analyze a group of stocks
def analyze_group(stock_group, start_date="2019-01-01", end_date="2022-01-01"):
    portfolios = {}

    for ticker in stock_group:
        try:
            # Download historical data for the ticker
            ticker_data = yf.download(ticker, start=start_date, end=end_date)
            
            if ticker_data.empty:
                print(f"Data for {ticker} is empty. Skipping.")
                continue

            # Simple Moving Average Strategy
            ticker_data['SMA50'] = ticker_data['Close'].rolling(window=50).mean()
            ticker_data['SMA200'] = ticker_data['Close'].rolling(window=200).mean()
            ticker_data['Signal'] = 0.0
            ticker_data.loc[ticker_data.index[50:], 'Signal'] = np.where(ticker_data['SMA50'][50:] > ticker_data['SMA200'][50:], 1.0, 0.0)
            ticker_data['Position'] = ticker_data['Signal'].diff()

            # Backtest the strategy
            initial_capital = 100000.0
            positions = pd.DataFrame(index=ticker_data.index).fillna(0.0)
            positions[ticker] = ticker_data['Signal']
            portfolio = positions.multiply(ticker_data['Adj Close'], axis=0)
            pos_diff = positions.diff()
            portfolio['holdings'] = (positions.multiply(ticker_data['Adj Close'], axis=0)).sum(axis=1)
            portfolio['cash'] = initial_capital - (pos_diff.multiply(ticker_data['Adj Close'], axis=0)).sum(axis=1).cumsum()
            portfolio['total'] = portfolio['cash'] + portfolio['holdings']
            portfolio['returns'] = portfolio['total'].pct_change()

            # Store the portfolio in the dictionary
            portfolios[ticker] = portfolio

        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

    if portfolios:
        combined_portfolio = pd.concat(portfolios, axis=1)
        combined_portfolio['total'] = combined_portfolio.filter(like='total').sum(axis=1)
        combined_portfolio['returns'] = combined_portfolio['total'].pct_change()

        return combined_portfolio
    else:
        return None

# Number of groups and stocks per group
n_groups = 100
stocks_per_group_min = 10
stocks_per_group_max = 20

# Create random groups of stocks
groups = [random.sample(tickers, random.randint(stocks_per_group_min, stocks_per_group_max)) for _ in range(n_groups)]

# Analyze each group
group_performances = []

for i, group in enumerate(groups):
    print(f"Analyzing group {i + 1}/{n_groups}: {group}")
    portfolio = analyze_group(group)
    if portfolio is not None:
        group_performances.append((group, portfolio))

# Identify top performing groups (without Monte Carlo)
top_groups = sorted(group_performances, key=lambda x: x[1]['total'].iloc[-1], reverse=True)[:5]

# Function to plot heatmap of correlations
def plot_heatmap_of_correlations(portfolio, group_number):
    returns = portfolio.filter(like='returns').dropna()
    correlations = returns.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap for Group {group_number}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'plots/group_{group_number}_correlations.png')
    plt.close()

# Save the results to a CSV file
csv_buffer = StringIO()
results_df = pd.DataFrame([(str(group), portfolio['total'].iloc[-1]) for group, portfolio in top_groups],
                          columns=['Group', 'Total Portfolio Value'])
results_df.to_csv(csv_buffer)

# Upload CSV to S3
s3 = boto3.client('s3')
s3.put_object(Bucket='my-quant-trading-bucket', Key='top_groups.csv', Body=csv_buffer.getvalue())

# Upload heatmap images to S3
for i, (group, portfolio) in enumerate(top_groups):
    plot_heatmap_of_correlations(portfolio, i + 1)
    
    s3.upload_file(f'plots/group_{i + 1}_correlations.png', 'my-quant-trading-bucket', f'plots/group_{i + 1}_correlations.png')

print("Analysis complete and heatmap results saved to S3!")
