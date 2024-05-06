import tkinter as tk
from tkinter import ttk, simpledialog
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

class ScrapeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Scrape & Display App")
        self.geometry("800x600")
        self.configure(bg='#E5DDC5')  # Set the background color

        self.heading_label = tk.Label(self, text="Cryptocurrency Data Scraper", font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#333F4B')
        self.heading_label.pack(pady=20)

        self.output_frame = tk.Frame(self, bg='#D875C7')  # Set the background color for the output frame
        self.output_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.button_frame = tk.Frame(self, bg='#f0f0f0')  # Set the background color for the button frame
        self.button_frame.pack(pady=10)

        self.scrape_button = tk.Button(self.button_frame, text="Start Scraping", command=self.start_scraping, width=20, height=5, bg='#007acc', fg='white')  # Set the button colors
        self.scrape_button.pack(side=tk.LEFT, padx=10)

        self.display_button = tk.Button(self.button_frame, text="Display Data", command=self.display_data, width=20, height=5, bg='#007acc', fg='white')
        self.display_button.pack(side=tk.LEFT, padx=10)

        self.graph_button = tk.Button(self.button_frame, text="Time Series Graph", command=self.show_graph, width=20, height=5, bg='#007acc', fg='white')
        self.graph_button.pack(side=tk.LEFT, padx=10)

        self.correlation_button = tk.Button(self.button_frame, text="Correlation Matrix", command=self.show_correlation_matrix, width=20, height=5, bg='#007acc', fg='white')
        self.correlation_button.pack(side=tk.LEFT, padx=10)

        self.growth_button = tk.Button(self.button_frame, text="Growth Rates", command=self.show_growth_rates, width=20, height=5, bg='#007acc', fg='white')
        self.growth_button.pack(side=tk.LEFT, padx=10)

        self.volatility_button = tk.Button(self.button_frame, text="Volatility Index", command=self.show_volatility_index, width=20, height=5, bg='#007acc', fg='white')
        self.volatility_button.pack(side=tk.LEFT, padx=10)

        self.comparative_button = tk.Button(self.button_frame, text="Comparative Data", command=self.show_comparative_data, width=20, height=5, bg='#007acc', fg='white')
        self.comparative_button.pack(side=tk.LEFT, padx=10)

        self.prediction_button = tk.Button(self.button_frame, text="Predictive Analysis", command=self.show_predictive_analysis, width=20, height=5, bg='#007acc', fg='white')
        self.prediction_button.pack(side=tk.LEFT, padx=10)
        
        self.crypto_date_list = []
        self.crypto_name_list = []
        self.crypto_symbol_list = []
        self.crypto_market_cap_list = []
        self.crypto_price_list = []
        self.crypto_circulating_supply_list = []
        self.crypto_voulume_24hr_list = []
        self.crypto_pct_1hr_list = []
        self.crypto_pct_24hr_list = []
        self.crypto_pct_7day_list = []

        self.df = pd.DataFrame()
        self.scrape_date_list = []

    
        
    def clear_output(self):
        for widget in self.output_frame.winfo_children():
            widget.destroy()


    def start_scraping(self):
        # ... (rest of the start_scraping method remains the same)
        self.clear_output()
        def scrape_date():
            url = 'https://coinmarketcap.com/historical/'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            a_tags = soup.find_all('a', class_='historical-link cmc-link')
            for tag in a_tags:
                href = tag.get('href')
                self.scrape_date_list.append(href)
        start_date_str = simpledialog.askstring("Enter Start Date", "Enter the start date in YYYYMMDD format:")

        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y%m%d")
                scrape_date()

                # Find the index of the start date in the scrape_date_list
                start_index = None
                for i, date_str in enumerate(self.scrape_date_list):
                    date = datetime.strptime(date_str.split('/')[-2], "%Y%m%d")
                    if date >= start_date:
                        start_index = i
                        break

                if start_index is not None:
                    self.scrape_date_list = self.scrape_date_list[start_index:]
                    print(f'There are {len(self.scrape_date_list)} dates(Sundays) available for scraping from {start_date_str}.')
                else:
                    print("No data available for the entered start date.")
                    return

                # ... (rest of the start_scraping method remains the same)
            except ValueError:
                print("Invalid date format. Please enter the date in YYYYMMDD format.")
        else:
            print("No start date entered.")
        print('There are ' + str(len(self.scrape_date_list)) + ' dates(Sundays) available for scraping from CoinMarketCap historical data.')


        def scrape_data(date):
            url = 'https://coinmarketcap.com' + date
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            tr = soup.find_all('tr', attrs={'class': 'cmc-table-row'})
            count = 0
            for row in tr:
                if count == 10:
                    break
                count += 1

                try:
                    crypto_date = date
                except AttributeError:
                    crypto_date = None

                try:
                    name_column = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sticky cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__name'})
                    crypto_name = name_column.find('a', attrs={'class': 'cmc-table__column-name--name cmc-link'}).text.strip()
                except AttributeError:
                    crypto_name = None

                try:
                    crypto_symbol = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--left cmc-table__cell--sort-by__symbol'}).text.strip()
                except AttributeError:
                    crypto_symbol = None

                try:
                    crypto_market_cap = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__market-cap'}).text.strip()
                except AttributeError:
                    crypto_market_cap = None

                try:
                    crypto_price = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__price'}).text.strip()
                except AttributeError:
                    crypto_price = None

                try:
                    crypto_circulating_supply = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__circulating-supply'}).text.strip().split(' ')[0]
                except AttributeError:
                    crypto_circulating_supply = None

                try:
                    crypto_voulume_24hr_td = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__volume-24-h'})
                    crypto_voulume_24hr = crypto_voulume_24hr_td.find('a', attrs={'class': 'cmc-link'}).text.strip()
                except AttributeError:
                    crypto_voulume_24hr = None

                try:
                    crypto_pct_1hr = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__percent-change-1-h'}).text.strip()
                except AttributeError:
                    crypto_pct_1hr = None

                try:
                    crypto_pct_24hr = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__percent-change-24-h'}).text.strip()
                except AttributeError:
                    crypto_pct_24hr = None

                try:
                    crypto_pct_7day = row.find('td', attrs={'class': 'cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__percent-change-7-d'}).text.strip()
                except AttributeError:
                    crypto_pct_7day = None

                self.crypto_date_list.append(crypto_date)
                self.crypto_name_list.append(crypto_name)
                self.crypto_symbol_list.append(crypto_symbol)
                self.crypto_market_cap_list.append(crypto_market_cap)
                self.crypto_price_list.append(crypto_price)
                self.crypto_circulating_supply_list.append(crypto_circulating_supply)
                self.crypto_voulume_24hr_list.append(crypto_voulume_24hr)
                self.crypto_pct_1hr_list.append(crypto_pct_1hr)
                self.crypto_pct_24hr_list.append(crypto_pct_24hr)
                self.crypto_pct_7day_list.append(crypto_pct_7day)
        

        date_format = "%Y%m%d"

        # Split and convert the start date and end date
        start_date = datetime.strptime(self.scrape_date_list[0].split('/')[-2], date_format).strftime('%Y-%m-%d')
        end_date = datetime.strptime(self.scrape_date_list[-1].split('/')[-2], date_format).strftime('%Y-%m-%d')
        print('There are ' + str(len(self.scrape_date_list)) + ' dates(Sundays) between ' + start_date + ' and ' + end_date)


        for i in range(len(self.scrape_date_list)):
            scrape_data(self.scrape_date_list[i])
            print("completed: " + str(i+1) + " out of " + str(len(self.scrape_date_list)))
        self.df['Date'] = self.crypto_date_list
        self.df['Name'] = self.crypto_name_list
        self.df['Symbol'] = self.crypto_symbol_list
        self.df['Market Cap'] = self.crypto_market_cap_list
        self.df['Price'] = self.crypto_price_list
        self.df['Circulating Supply'] = self.crypto_circulating_supply_list
        self.df['Volume (24hr)'] = self.crypto_voulume_24hr_list
        self.df['% 1h'] = self.crypto_pct_1hr_list
        self.df['% 24h'] = self.crypto_pct_24hr_list
        self.df['% 7d'] = self.crypto_pct_7day_list

        # Extract the date component from the 'Date' column and convert it to a datetime data type
        self.df['Date'] = pd.to_datetime(self.df['Date'].str.split('/').str[-2], format='%Y%m%d')

        # Replace the dollar signs ($) and commas (,) from the 'Market Cap' and 'Price' columns
        self.df['Market Cap'] = self.df['Market Cap'].str.replace('[$,]', '', regex=True)
        self.df['Price'] = self.df['Price'].str.replace('[$,]', '', regex=True)

        # Replace the commas (,) from the 'Circulating Supply' column
        self.df['Circulating Supply'] = self.df['Circulating Supply'].str.replace(',', '')

        # Replace the dollar signs ($) and commas (,) from the 'Volume (24hr)' columns
        self.df['Volume (24hr)'] = self.df['Volume (24hr)'].str.replace('[$,]', '', regex=True)

        # Replace the unchange sign (--), the smaller sign (<), the larger sign (>) and percentage sign (%) from the '% 1h', '% 24h', and '% 7d' columns
        self.df['% 1h'] = self.df['% 1h'].str.replace('--', '0').str.lstrip('>').str.lstrip('<').str.rstrip('%')
        self.df['% 24h'] = self.df['% 24h'].str.replace('--', '0').str.lstrip('>').str.lstrip('<').str.rstrip('%')
        self.df['% 7d'] = self.df['% 7d'].str.replace('--', '0').str.lstrip('>').str.lstrip('<').str.rstrip('%')

        # Convert the numeric columns to appropriate data types, replacing invalid values with NaN
        numeric_cols = ['Market Cap', 'Price', 'Circulating Supply', 'Volume (24hr)', '% 1h', '% 24h', '% 7d']
        self.df[numeric_cols] = self.df[numeric_cols].apply(lambda x: pd.to_numeric(x))

        # Handle specific case of "<0.01" by replacing it with a small non-zero value, e.g., 0.005
        self.df.loc[self.df['% 1h'] < 0, '% 1h'] = 0.005

        # Set the display format for float and integer values
        pd.options.display.float_format = '{:.2f}'.format


        # Select numerical columns for imputation
        numeric_cols = ['Market Cap', 'Price', 'Circulating Supply', '% 1h', '% 24h', '% 7d', 'Volume (24hr)']

        # Normalization
        scaler = MinMaxScaler()
        df_normalized = self.df.copy()
        df_normalized[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

        # KNN Imputation
        imputer = KNNImputer(n_neighbors=3)
        df_imputed = df_normalized.copy()
        df_imputed[numeric_cols] = imputer.fit_transform(df_normalized[numeric_cols])

        # Reverse normalization
        df_imputed[numeric_cols] = scaler.inverse_transform(df_imputed[numeric_cols])

        self.df = df_imputed.copy()
        # Rename the columns in the DataFrame
        self.df.columns = ['Date', 'Name', 'Symbol', 'Market Cap', 'Price', 'Circulating Supply', 'Volume (24hr)', '% 1h', '% 24h', '% 7d']

    def display_data(self):
        self.clear_output()

        # Create a frame to hold the Text widget and the scrollbar
        data_frame = tk.Frame(self.output_frame, bg='#f0f0f0')  # Set the background color for the data frame
        data_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Text widget with a scrollbar
        data_text = tk.Text(data_frame, wrap=tk.NONE, bg='#f0f0f0', fg='#333F4B')  # Set the background and text color for the Text widget
        data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a vertical scrollbar
        scrollbar = tk.Scrollbar(data_frame, command=data_text.yview, bg='#d0d0d0')  # Set the background color for the scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the Text widget to use the scrollbar
        data_text.config(yscrollcommand=scrollbar.set)

        # Insert the data into the Text widget
        data_text.insert(tk.END, self.df.to_string())

    def show_graph(self):
        self.clear_output()
        import matplotlib.pyplot as plt
        coin = simpledialog.askstring("Coin Name", "Enter name of the coin (format: BTC, XRP)")
        # Convert 'Date' to datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Filter data for coin
        coin_data = self.df[self.df['Symbol'] == coin]

        # Set the style of the axes and the text color
        plt.rcParams['axes.edgecolor']='#333F4B'
        plt.rcParams['axes.linewidth']=0.8
        plt.rcParams['xtick.color']='#333F4B'
        plt.rcParams['ytick.color']='#333F4B'
        plt.rcParams['text.color']='#333F4B'

        import matplotlib.dates as mdates

        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex = True)

        # Plot 'Price' over time
        axs[0].plot(coin_data['Date'], coin_data['Price'], color='#007acc', alpha=0.7)
        axs[0].set_title(f'{coin} Price Over Time', fontsize=15, fontweight='black', color = '#333F4B')

        # Plot 'Market Cap' over time
        axs[1].plot(coin_data['Date'], coin_data['Market Cap'], color='#007acc', alpha=0.7)
        axs[1].set_title(f'{coin} Market Cap Over Time', fontsize=15, fontweight='black', color = '#333F4B')

        # Plot 'Volume (24hr)' over time
        axs[2].plot(coin_data['Date'], coin_data['Volume (24hr)'], color='#007acc', alpha=0.7)
        axs[2].set_title(f'{coin} 24hr Trading Volume Over Time', fontsize=15, fontweight='black', color = '#333F4B')

        # Formatting dates
        date_format = mdates.DateFormatter('%Y')
        for ax in axs:
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.subplots_adjust(hspace=0.5)
        plt.tight_layout()
        plt.show()

    def show_correlation_matrix(self):
        self.clear_output()
        correlation_matrix = self.df[['Market Cap', 'Price', 'Circulating Supply', 'Volume (24hr)', '% 1h', '% 24h', '% 7d']].corr()
        correlation_matrix_text = tk.Text(self.output_frame, wrap=tk.NONE)
        correlation_matrix_text.pack(fill=tk.BOTH, expand=True)
        correlation_matrix_text.insert(tk.END, correlation_matrix.to_string())

    def show_growth_rates(self):
        self.clear_output()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        data_sorted = self.df.sort_values(by=['Symbol', 'Date'])
        growth_rates = data_sorted.groupby('Symbol').apply(lambda x: ((x[['Price', 'Market Cap']].iloc[-1] - x[['Price', 'Market Cap']].iloc[0]) / x[['Price', 'Market Cap']].iloc[0]) * 100, include_groups=False)
        growth_rates_text = tk.Text(self.output_frame, wrap=tk.NONE)
        growth_rates_text.pack(fill=tk.BOTH, expand=True)
        growth_rates_text.insert(tk.END, growth_rates.to_string())

    def show_volatility_index(self):
        self.clear_output()
        data_sorted = self.df.sort_values(by=['Symbol', 'Date'])
        data_sorted['Price Change'] = data_sorted.groupby('Symbol')['Price'].pct_change()
        volatility = data_sorted.groupby('Symbol')['Price Change'].std()
        name = simpledialog.askstring("Volatility Index", "Enter the name to search for volatility:")
        if name and name in volatility.index:
            volatility_value = volatility.loc[name]
            volatility_text = tk.Text(self.output_frame, wrap=tk.NONE)
            volatility_text.pack(fill=tk.BOTH, expand=True)
            volatility_text.insert(tk.END, f"Volatility for {name}: {volatility_value:.2f}")
        else:
            tk.messagebox.showerror("Error", "No such coin")

    def show_comparative_data(self):
        self.clear_output()
        data_sorted = self.df.sort_values(by=['Symbol', 'Date'])
        most_recent_data = data_sorted.groupby('Symbol').last()
        comparative_data = most_recent_data[['Price', 'Market Cap', 'Volume (24hr)']]
        comparative_data_text = tk.Text(self.output_frame, wrap=tk.NONE)
        comparative_data_text.pack(fill=tk.BOTH, expand=True)
        comparative_data_text.insert(tk.END, comparative_data.to_string())

    def show_predictive_analysis(self):
        self.clear_output()
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        import matplotlib.pyplot as plt

        name = simpledialog.askstring("Predictive Analysis", "Enter name of the coin price you want to predict:")
        if name:
            coin_data = self.df[self.df['Name'] == name]
            if not coin_data.empty and coin_data['Price'].notna().any():
                price_data = coin_data['Price'].values
                scaler = MinMaxScaler()
                price_data_normalized = scaler.fit_transform(price_data.reshape(-1, 1))

                n_steps = 30
                X = []
                y = []
                for i in range(n_steps, len(price_data_normalized)):
                    X.append(price_data_normalized[i - n_steps:i])
                    y.append(price_data_normalized[i])
                X = np.array(X)
                y = np.array(y)

                train_size = int(0.8 * len(X))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                model = Sequential()
                model.add(LSTM(units=64, activation='relu', input_shape=(n_steps, 1)))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                model.fit(X_train, y_train, epochs=50, batch_size=32)

                last_sequence = X_test[-1]
                forecast = []
                for _ in range(60):
                    next_prediction = model.predict(last_sequence.reshape(1, n_steps, 1))
                    forecast.append(next_prediction[0, 0])
                    last_sequence = np.append(last_sequence[1:], next_prediction[0, 0])

                forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

                last_date = coin_data['Date'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60)

                fig, ax = plt.subplots(figsize=(12, 6))

                canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                ax.plot(coin_data['Date'], price_data, label='Original')
                ax.plot(future_dates, forecast, label='Extended Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Coin Price')
                ax.legend()
                plt.show()
            else:
                tk.messagebox.showerror("Error", "No data available for the entered coin name or the 'Price' column contains invalid values.")

            
if __name__ == "__main__":
    app = ScrapeApp()
    app.mainloop() 
