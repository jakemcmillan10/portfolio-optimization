from flask import Flask, render_template, request, send_file, session
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Ensure you have a secret key for session management

# Function to validate tickers
def validate_tickers(tickers):
    valid_tickers = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')  # Check if ticker exists with 1-day data
        if data.empty:
            print(f"Ticker '{ticker}' does not exist.")
        else:
            valid_tickers.append(ticker)
    return valid_tickers

# Function to get stock data
def get_stock_data(tickers, years):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today().replace(year=datetime.today().year - years)).strftime('%Y-%m-%d')
    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if not data.empty:
            monthly_data = data['Close'].resample('MS').first()
            stock_data[ticker] = monthly_data
    return pd.DataFrame(stock_data)

# Function to calculate portfolio metrics and Sharpe ratio
def calculate_portfolio_metrics(stock_df, risk_free_rate):
    returns_df = stock_df.pct_change().dropna() * 100
    avg_annual_returns = returns_df.mean() * 12
    annual_volatilities = returns_df.std() * np.sqrt(12)
    cov_matrix = returns_df.cov() * 12
    sharpe_ratios = (avg_annual_returns - risk_free_rate) / annual_volatilities
    return avg_annual_returns, annual_volatilities, cov_matrix, sharpe_ratios

# Optimization function
def optimize_portfolio(avg_annual_returns, cov_matrix, risk_free_rate, short_selling_enabled):
    num_assets = len(avg_annual_returns)
    init_guess = np.ones(num_assets) / num_assets
    bounds = [(-1, 1) for _ in range(num_assets)] if short_selling_enabled else [(0, 1) for _ in range(num_assets)]
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    result = minimize(lambda w: -((np.sum(w * avg_annual_returns) - risk_free_rate) /
                                   np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))),
                      init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Home route - input form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and show results
@app.route('/results', methods=['POST'])
def results():
    # Get form data
    tickers = request.form['tickers'].split(',')
    tickers = [ticker.strip().upper() for ticker in tickers]  # Ensure uppercase ticker symbols
    years = int(request.form['years'])
    risk_free_rate = float(request.form['risk_free_rate'])
    short_selling_enabled = request.form['short_selling'] == 'yes'

    # Validate tickers
    valid_tickers = validate_tickers(tickers)
    if len(valid_tickers) != len(tickers):
        return render_template('index.html', error="One or more tickers were invalid. Please try again.")

    # Store valid tickers and years in session
    session['valid_tickers'] = valid_tickers
    session['years'] = years
    session['risk_free_rate'] = risk_free_rate  # Store risk-free rate in session

    # Get stock data and calculate metrics
    stock_df = get_stock_data(valid_tickers, years)
    avg_annual_returns, annual_volatilities, cov_matrix, sharpe_ratios = calculate_portfolio_metrics(stock_df, risk_free_rate)

    # Optimize portfolio
    optimal_weights = optimize_portfolio(avg_annual_returns, cov_matrix, risk_free_rate, short_selling_enabled)

    # Prepare results for optimized portfolio
    portfolio_return = np.sum(optimal_weights * avg_annual_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    results = {
        'tickers_weights': list(zip(valid_tickers, [round(weight * 100, 2) for weight in optimal_weights])),  # Combine tickers and weights
        'portfolio_return': f"{round(portfolio_return, 2)}",  # Format portfolio return without %
        'portfolio_volatility': f"{round(portfolio_volatility, 2)}",  # Format portfolio volatility without %
        'sharpe_ratio': round(sharpe_ratio, 2),
        'individual_sharpe_ratios': {ticker: round(sharpe, 2) for ticker, sharpe in sharpe_ratios.items()},  # Format Sharpe ratios
    }

    return render_template('results.html', results=results)

    # Store valid tickers and years in session
    session['valid_tickers'] = valid_tickers
    session['years'] = years
    session['risk_free_rate'] = risk_free_rate  # Store risk-free rate in session

    # Get stock data and calculate metrics
    stock_df = get_stock_data(valid_tickers, years)
    avg_annual_returns, annual_volatilities, cov_matrix, sharpe_ratios = calculate_portfolio_metrics(stock_df, risk_free_rate)

    # Optimize portfolio
    optimal_weights = optimize_portfolio(avg_annual_returns, cov_matrix, risk_free_rate, short_selling_enabled)

    # Prepare results for optimized portfolio
    portfolio_return = np.sum(optimal_weights * avg_annual_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    results = {
        'tickers': valid_tickers,  # Store valid tickers as a list
        'weights': [round(weight * 100, 2) for weight in optimal_weights],  # Store weights as a list of rounded values
        'portfolio_return': f"{round(portfolio_return, 2)}",  # Format portfolio return without %
        'portfolio_volatility': f"{round(portfolio_volatility, 2)}",  # Format portfolio volatility without %
        'sharpe_ratio': round(sharpe_ratio, 2),
        'individual_sharpe_ratios': {ticker: round(sharpe, 2) for ticker, sharpe in sharpe_ratios.items()},  # Format Sharpe ratios
    }

    return render_template('results.html', results=results)

# Route to export CSV
@app.route('/export', methods=['POST'])
def export_csv():
    # Get the tickers and years from the session
    valid_tickers = session.get('valid_tickers', [])
    years = session.get('years', 1)

    if not valid_tickers:
        return "No tickers to export", 400

    # Get stock data
    stock_df = get_stock_data(valid_tickers, years)

    # Save CSV
    filename = "stock_data.csv"
    stock_df.to_csv(filename)

    # Send CSV file as download
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)