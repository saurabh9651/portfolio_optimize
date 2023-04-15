## ABSTRACT

An enchanting journey through diverse investment strategies, this study unveils the mystique behind maximizing Sharpe ratios, minimizing variances, and equal-weighting portfolios. Delving into a realm of captivating data, we unearth the triumphant equal-weighted strategy, emerging as the optimal risk-adjusted performer. Join us as we illuminate the path to investment wisdom, empowering investors to embrace simplicity in a world of complexity.

### INTRODUCTION

In recent years, portfolio management strategies have garnered significant attention in the financial literature, driven by increasing market volatility and the need for innovative investment approaches. This paper critically analyzes various portfolio management techniques, focusing on their efficacy in delivering optimal returns while managing risk. Drawing upon an extensive review of academic literature, including seminal works by Markowitz (1952), Sharpe (1964), and Black & Litterman (1991), we aim to provide a comprehensive understanding of modern portfolio theory, risk parity, and factor-based investing. This analysis facilitates the development of robust recommendations for investors seeking to enhance their portfolio performance in today's complex financial landscape.

- Sharpe, W.F. (1964), CAPITAL ASSET PRICES: A THEORY OF MARKET EQUILIBRIUM UNDER CONDITIONS OF RISK*. The Journal of Finance, 19: 425-442. https://doi.org/10.1111/j.1540-6261.1964.tb02865.x
- Markowitz, H. (1952), PORTFOLIO SELECTION*. The Journal of Finance, 7: 77-91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization. Financial Analysts Journal, 48(5), 28–43. http://www.jstor.org/stable/4479577


### PORTFOLIO OPTIMIZATION STRATEGIES

Portfolio optimization strategies aim to maximize returns while minimizing risk, typically through the allocation of assets in a portfolio. This section delves into three widely-adopted strategies: maximizing the Sharpe ratio, minimizing variance, and equal-weighted portfolio construction.

#### 1. Maximize Sharpe Ratio
The Sharpe ratio, introduced by Sharpe (1964), is a widely used risk-adjusted performance measure in portfolio management. It quantifies the excess return per unit of risk by comparing an investment's return to its standard deviation. A portfolio with a higher Sharpe ratio indicates better risk-adjusted performance. Optimization techniques, such as quadratic programming, can be employed to identify the asset allocation that maximizes the Sharpe ratio, thus yielding an efficient portfolio (Jagannathan & Ma, 2003).

#### 2. Minimum Variance
The minimum variance strategy aims to construct a portfolio that minimizes the total risk, as measured by the portfolio's variance or standard deviation. Markowitz's (1952) pioneering work on modern portfolio theory posits that investors can minimize risk by diversifying their investments across a range of assets with low correlations. This approach enables the construction of an efficient frontier, where each portfolio offers the highest possible return for a given level of risk. The global minimum variance (GMV) portfolio is located at the lowest point on this frontier, representing the optimal allocation with the least amount of risk (DeMiguel et al., 2009).

#### 3. Equal-Weighted Portfolio
The equal-weighted portfolio strategy involves allocating the same proportion of capital to each asset in a portfolio, disregarding individual asset characteristics such as market capitalization or expected returns. This approach offers several benefits, including simplicity and inherent diversification, as it avoids concentration risk (DeMiguel et al., 2009). Research by DeMiguel et al. (2009) suggests that equal-weighted portfolios can outperform more complex optimization techniques, particularly in the presence of estimation errors in the input parameters. However, equal-weighted portfolios may require frequent rebalancing and may underperform in certain market conditions (Clarke et al., 2002).

- Jagannathan, R. and Ma, T. (2003), Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. The Journal of Finance, 58: 1651-1683. https://doi.org/10.1111/1540-6261.00580
- Victor DeMiguel, Lorenzo Garlappi, Raman Uppal, Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?, The Review of Financial Studies, Volume 22, Issue 5, May 2009, Pages 1915–1953, https://doi.org/10.1093/rfs/hhm075


### EXPLORE

Python script is designed to analyze the performance of the three portfolio optimization strategies discussed earlier: maximizing the Sharpe ratio, minimizing variance, and equal-weighted portfolios. The script is divided into several sections, each with a specific purpose.

1. Import necessary libraries: 
The script imports necessary libraries such as NumPy, Pandas, yfinance, and others to perform the required calculations and fetch financial data.

2. Fetch historical data: 
The fetch_data and fetch_price_data functions use the Yahoo Finance API to fetch historical adjusted closing prices for a given list of stocks between specified dates.

3. Define optimization functions: 
The script defines three optimization functions: maximize_sharpe_ratio, minimize_variance, and get_weights_port_optimize. The first two functions receive mean returns, covariance matrix, and risk-free rate as inputs and return optimal asset allocations based on the respective strategies. The third function calculates the optimal weights for each strategy using historical price data, stock list, and trade date.

4. Annual performance metrics: 
The annual_sharpe_ratio and annual_metric functions calculate annualized performance metrics, such as the Sharpe ratio, annualized return, and annualized variation, based on daily returns.

5. Main script: 
The main part of the script executes the analysis. It fetches stock data for a specified list of stocks (e.g., "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA") and a date range. Then, it resamples the stock data to daily frequency, calculates rebalance periods, and fetches optimal weights for each strategy using the get_weights_port_optimize function. Next, the script calculates the investment value, remaining cash, and total amount for each strategy during the rebalance periods. Finally, the results can be visualized using Matplotlib or another plotting library.

This Python script provides a comprehensive framework for evaluating the performance of the three portfolio optimization strategies, allowing investors to gain valuable insights into their efficacy and potential application in real-world scenarios.

### ANALYZE

Our analysis of the data shows that the Sharpe ratios for the three investment strategies are as follows: (1) maximize Sharpe ratio (0.5339), (2) minimize variance (0.2472), and (3) equal-weighted (0.5924). Notably, the equal-weighted strategy demonstrates the highest Sharpe ratio, indicating that it offers the best risk-adjusted return. In terms of returns, the maximize Sharpe ratio strategy yields the highest return (0.6715), followed by the equal-weighted (0.6524) and the minimize variance strategies (0.2615). The variance, however, is lowest for the minimize variance strategy (0.2439), followed by the equal-weighted (0.2539) and the maximize Sharpe ratio strategies (0.2899). These results align with the findings of DeMiguel et al. (2009), who concluded that the equal-weighted portfolio often outperforms other sophisticated optimization strategies in terms of risk-adjusted 

![graph](https://user-images.githubusercontent.com/16968671/232250388-a4cc86ec-22d5-4c1c-8a22-24efa559eaaa.png)

### CONCLUSION
The equal-weighted investment strategy demonstrates the highest Sharpe ratio, indicating superior risk-adjusted performance compared to the maximize Sharpe ratio and minimize variance strategies. This reinforces the notion that a simple equal-weighted approach can often outperform more complex optimization techniques. Investors should carefully consider their risk tolerance and investment objectives when selecting a strategy, but our analysis suggests that an equal-weighted portfolio may be a compelling choice for many.

![results](https://user-images.githubusercontent.com/16968671/232250427-d09c8e01-9c21-4111-a022-b314ad3fbb71.png)

