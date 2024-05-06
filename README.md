# CrytoForecast
 Python app scrapes CoinMarketCap for cryptocurrency data, handles missing values, visualizes metrics in GUI with time series graphs, correlation matrix, growth rates, volatility. LSTM model forecasts prices. Uses pandas, scikit-learn, tensorflow, beautifulsoup4, matplotlib, tkinter.

## Cryptocurrency Data Scraper and Visualizer
This Python application allows you to scrape and visualize historical cryptocurrency data from CoinMarketCap. With an intuitive graphical user interface (GUI), you can effortlessly retrieve, process, and analyze various metrics for the top cryptocurrencies.
Features

### Data Scraping:
Scrape historical cryptocurrency data from CoinMarketCap, including coin name, symbol, market cap, price, circulating supply, trading volume, and percentage changes over different time periods.
### Data Cleaning and Preprocessing:
Clean and preprocess the scraped data, handling missing values and formatting issues to ensure data integrity.
### Data Display: 
View the scraped and processed data in a tabular format within the application.
### Time Series Visualization: 
Plot time series graphs for coin price, market cap, and 24-hour trading volume to observe trends over time.
### Correlation Matrix: 
Calculate and display the correlation matrix to understand the relationships between different metrics.
### Growth Rate Analysis: 
Analyze the growth rates for coin prices and market caps over the scraped period.
### Volatility Index: 
Calculate and display the volatility index for a specific cryptocurrency.
### Comparative Analysis: 
Compare the latest values of price, market cap, and trading volume across different cryptocurrencies.
### Predictive Analysis: 
Utilize machine learning techniques, such as Long Short-Term Memory (LSTM) models, to forecast future coin prices based on historical data.

## Getting Started

Clone the repository: git clone https://github.com/shreyashD0810/CrytoForecast.git
Install the required Python packages: pip install -r requirements.txt
Run the application: python main.py

## Dependencies
The application relies on the following Python packages:

tkinter: For creating the graphical user interface (GUI)
requests: For making HTTP requests to scrape data from CoinMarketCap
beautifulsoup4: For parsing HTML content from the scraped webpages
pandas: For data manipulation and analysis
matplotlib: For visualizing data and creating graphs
scikit-learn: For data preprocessing and machine learning techniques
tensorflow: For building and training machine learning models (used in predictive analysis)

The CoinMarketCap website for providing historical cryptocurrency data.
The developers of the Python packages used in this application.
