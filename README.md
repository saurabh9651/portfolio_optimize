# Portfolio Optimization Strategies
This project demonstrates the implementation of three different portfolio optimization strategies:

1. Maximize Sharpe Ratio
2. Minimize Variance
3. Equal Weighted

The strategies are applied to a given set of stocks with a specified rebalancing frequency, and their performance is compared over time.

## Dependencies
To run this project, you need to install the following Python libraries:
```
yfinance
pandas
numpy
scipy
matplotlib
```
You can install them using pip:
```python
pip install yfinance pandas numpy scipy matplotlib
```
## Usage
1. Clone the repository:
```python
git clone https://github.com/your_github_username/portfolio-optimization.git
```
2. Change the stock list, risk-free rate, start and end dates, and rebalancing frequency in the `potfolio-optimization.ipynb` notebook:
```python
stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
risk_free_rate = 0.02
start_date = "2019-01-01"
end_date = "2021-12-31"
rebal_freq = "M"
```
3. Run the notebook to create the graph and analysis

The script will download the historical stock data, apply the portfolio optimization strategies, and display the performance of each strategy over time. The final output will be a DataFrame with the optimized weights, shares, investment values, and total investment values for each strategy and each rebalancing period.

## License
This project is licensed under the MIT License.
