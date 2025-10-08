import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
import joblib
import os
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

class StockPredictor:
    def _init_(self, symbol):
        self.symbol = symbol.upper()
        self.data = None
        self.company_name = None
        self.scaler = None
        self.scaled_data = None
        self.lgmregression_model = None
        self.lgmclassification_model = None
        self.model_path = f"models/{self.symbol}"
        
        # Ensure directory exists for this stock's models
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # Alpha Vantage API Key - Replace with your key
        self.api_key = 'YOUR_API_KEY'  # ← Replace with your API key
    
    def get_company_name(self):
        """Get the company name for the stock symbol"""
        if self.company_name:
            return self.company_name
            
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'OVERVIEW',
                'symbol': self.symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Name' in data:
                self.company_name = data['Name']
            else:
                # Map common stock symbols to company names as fallback
                company_map = {
                    'AAPL': 'Apple Inc.',
                    'MSFT': 'Microsoft Corporation',
                    'GOOGL': 'Alphabet Inc.',
                    'GOOG': 'Alphabet Inc.',
                    'AMZN': 'Amazon.com Inc.',
                    'META': 'Meta Platforms Inc.',
                    'TSLA': 'Tesla Inc.',
                    'NVDA': 'NVIDIA Corporation',
                    'JPM': 'JPMorgan Chase & Co.',
                    'V': 'Visa Inc.'
                }
                self.company_name = company_map.get(self.symbol, self.symbol)
                
        except Exception as e:
            print(f"Error fetching company name: {str(e)}")
            self.company_name = self.symbol  # Use symbol as fallback
            
        return self.company_name
    
    def fetch_data(self):
        """Fetch stock data from Alpha Vantage API"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': self.symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                df = df[['4. close']].astype(float)
                df.columns = ['Close']
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Check if we have enough data (at least 60 days)
                if len(df) < 60:
                    raise ValueError(f"Not enough data available for {self.symbol}. Need at least 60 days of historical data.")
                
                self.data = df
                self._preprocess_data()
                return df
            elif 'Note' in data:
                # API rate limit reached
                raise ValueError(f"API limit reached: {data['Note']}. Please try again in a minute.")
            elif 'Error Message' in data:
                # Invalid API call
                raise ValueError(f"API Error: {data['Error Message']}. Check if '{self.symbol}' is a valid stock symbol.")
            else:
                # General error
                raise ValueError(f"Unable to retrieve data for symbol '{self.symbol}'. Please verify it's a valid stock symbol.")
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error when fetching data: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing stock data: {str(e)}")
    
    def get_latest_price(self):
        """Return the most recent closing price"""
        if self.data is not None and not self.data.empty:
            return self.data['Close'].iloc[-1]
        return None
    
    def _calculate_rsi(self, window=14):
        """Calculate RSI indicator"""
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, np.finfo(float).eps)
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _preprocess_data(self):
        """Preprocess data for model training"""
        self.data['RSI_14'] = self._calculate_rsi(window=14)
        self.data.dropna(inplace=True)

        close_prices = self.data['Close'].values.astype('float32')
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(close_prices.reshape(-1, 1))

        self.X, self.y = [], []
        for i in range(60, len(self.scaled_data)):
            self.X.append(self.scaled_data[i-60:i, 0])
            self.y.append(self.scaled_data[i, 0])

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        # Make sure we have enough data for training
        if len(self.X) < 10 or len(self.y) < 10:
            raise ValueError(f"Not enough data to train models for {self.symbol}")
    
    def train_models(self):
        """Train predictive models"""
        # Check if models already exist for this stock
        regression_model_path = os.path.join(self.model_path, "lgmregression_model.pkl")
        classification_model_path = os.path.join(self.model_path, "lgmclassification_model.pkl")
        
        if os.path.exists(regression_model_path) and os.path.exists(classification_model_path):
            # Load existing models
            try:
                self.lgmregression_model = joblib.load(regression_model_path)
                self.lgmclassification_model = joblib.load(classification_model_path)
                print(f"Loaded existing models for {self.symbol}")
                return
            except Exception as e:
                print(f"Error loading existing models, will train new ones: {str(e)}")
                # Continue to train new models
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, shuffle=False)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            
            # Feature names
            feature_names = [f'f{i}' for i in range(X_train_flat.shape[1])]
            
            # Train regression model
            self.lgmregression_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
            self.lgmregression_model.fit(
                pd.DataFrame(X_train_flat, columns=feature_names), 
                y_train
            )
            
            # Create classification labels
            three_class_labels = pd.cut(self.y, bins=[-np.inf, -0.05, 0.05, np.inf], labels=['Down', 'Flat', 'Up'])
            three_class_labels_numeric = three_class_labels.codes
            
            # Split classification labels
            y_train_class_labels = three_class_labels_numeric[:len(X_train)]
            
            # Train classification model
            self.lgmclassification_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
            self.lgmclassification_model.fit(
                pd.DataFrame(X_train_flat, columns=feature_names), 
                y_train_class_labels
            )
            
            # Save models
            joblib.dump(self.lgmregression_model, regression_model_path)
            joblib.dump(self.lgmclassification_model, classification_model_path)
            print(f"Successfully trained and saved models for {self.symbol}")
            
        except Exception as e:
            raise ValueError(f"Error training models: {str(e)}")
    
    def predict_future(self, days):
        """Predict future prices"""
        if self.lgmregression_model is None:
            raise ValueError("Model not trained. Call train_models() first")
            
        try:
            current_batch = self.scaled_data[-60:].reshape(1, 60)
            feature_names = [f'f{i}' for i in range(60)]

            future_predictions = []
            for _ in range(days):
                df_batch = pd.DataFrame(current_batch, columns=feature_names)
                pred = self.lgmregression_model.predict(df_batch)[0]
                future_predictions.append(pred)
                current_batch = np.append(current_batch[:, 1:], [[pred]], axis=1)

            predicted_prices = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=days)
            return pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted_Close'])
        
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")


