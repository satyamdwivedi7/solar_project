"""
LSTM Neural Network for Solar Energy Forecasting
Deep learning approach for accurate multi-step ahead prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SolarLSTMForecaster:
    def __init__(self, config_path='../../config/ml_hyperparameters.yaml'):
        """Initialize LSTM model with configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['solar_forecasting']['lstm']
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.history = None
        
    def create_sequences(self, X, y, seq_length):
        """Create sequences for LSTM input"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape):
        """Build LSTM architecture"""
        model = keras.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(
                self.config['hidden_units'][0],
                return_sequences=True,
                input_shape=input_shape
            ),
            layers.Dropout(self.config['dropout']),
            
            # Second LSTM layer
            layers.LSTM(
                self.config['hidden_units'][1],
                return_sequences=True
            ),
            layers.Dropout(self.config['dropout']),
            
            # Third LSTM layer
            layers.LSTM(
                self.config['hidden_units'][2],
                return_sequences=False
            ),
            layers.Dropout(self.config['dropout']),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Output layer
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, df):
        """Prepare data for LSTM training"""
        # Feature engineering
        features = [
            'ghi', 'dni', 'dhi', 'temp_air', 'temp_module',
            'wind_speed', 'pressure', 'humidity',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]
        
        X = df[features].values
        y = df['energy_kwh'].values.reshape(-1, 1)
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(
            X_scaled, 
            y_scaled, 
            self.config['sequence_length']
        )
        
        return X_seq, y_seq
    
    def train(self, train_df, val_df):
        """Train LSTM model"""
        print("🧠 Training LSTM Solar Forecasting Model...")
        
        # Prepare data
        X_train, y_train = self.prepare_data(train_df)
        X_val, y_val = self.prepare_data(val_df)
        
        print(f"   Training sequences: {X_train.shape}")
        print(f"   Validation sequences: {X_val.shape}")
        
        # Build model
        self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("✅ Training complete!")
        return self.history
    
    def predict(self, df):
        """Make predictions"""
        X, _ = self.prepare_data(df)
        y_pred_scaled = self.model.predict(X, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred.flatten()
    
    def evaluate(self, test_df):
        """Evaluate model performance"""
        print("\n📊 Evaluating LSTM Model...")
        
        X_test, y_test_scaled = self.prepare_data(test_df)
        y_test = self.scaler_y.inverse_transform(y_test_scaled).flatten()
        
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        print(f"   MAE:  {mae:.4f} kWh")
        print(f"   RMSE: {rmse:.4f} kWh")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def save(self, path):
        """Save model and scalers"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # Save scalers
        import joblib
        joblib.dump(self.scaler_X, path.replace('.h5', '_scaler_X.pkl'))
        joblib.dump(self.scaler_y, path.replace('.h5', '_scaler_y.pkl'))
        
        print(f"💾 Model saved to: {path}")
    
    def load(self, path):
        """Load model and scalers"""
        self.model = keras.models.load_model(path)
        
        import joblib
        self.scaler_X = joblib.load(path.replace('.h5', '_scaler_X.pkl'))
        self.scaler_y = joblib.load(path.replace('.h5', '_scaler_y.pkl'))
        
        print(f"📂 Model loaded from: {path}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('LSTM Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Train MAE')
        ax2.plot(self.history.history['val_mae'], label='Val MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('LSTM Training MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved: {save_path}")
        
        plt.show()

def main():
    """Train and evaluate LSTM model"""
    print("="*70)
    print(" 🧠 LSTM SOLAR ENERGY FORECASTING")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv('../../data/processed/splits/train.csv')
    val_df = pd.read_csv('../../data/processed/splits/val.csv')
    test_df = pd.read_csv('../../data/processed/splits/test.csv')
    
    print(f"\n📂 Data loaded:")
    print(f"   Train: {len(train_df)} records")
    print(f"   Val:   {len(val_df)} records")
    print(f"   Test:  {len(test_df)} records")
    
    # Initialize and train model
    forecaster = SolarLSTMForecaster()
    forecaster.train(train_df, val_df)
    
    # Plot training history
    forecaster.plot_training_history(
        save_path='../../reports/ml_performance/lstm_training_history.png'
    )
    
    # Evaluate
    results = forecaster.evaluate(test_df)
    
    # Save model
    forecaster.save('../../models/solar_forecast/lstm_model.h5')
    
    # Save results
    os.makedirs('../../reports/ml_performance/', exist_ok=True)
    results_df = pd.DataFrame({
        'actual': results['actual'],
        'predicted': results['predictions']
    })
    results_df.to_csv('../../reports/ml_performance/lstm_predictions.csv', index=False)
    
    print("\n✅ LSTM training complete!")
    print("="*70)

if __name__ == "__main__":
    main()
