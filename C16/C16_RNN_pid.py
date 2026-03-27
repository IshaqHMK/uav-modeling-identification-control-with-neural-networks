import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


class ErrorToControlRNNController:
    def __init__(self, sequence_length=10, error_features=4, control_features=4):
        """
        Initialize the RNN controller trainer for Error -> Control mapping

        Args:
            sequence_length: Number of timesteps to look back
            error_features: Number of error signals (e.g., x_error, y_error, z_error, yaw_error)
            control_features: Number of control outputs (4 motor commands for quadcopter)
        """
        self.sequence_length = sequence_length
        self.error_features = error_features
        self.control_features = control_features
        self.model = None
        self.error_scaler = StandardScaler()  # Errors can be positive/negative
        self.control_scaler = MinMaxScaler()  # Control signals usually positive

    def load_and_preprocess_data(self, data_file=None, data_array=None):
        """
        Load and preprocess the PID controller data (Error -> Control)

        Expected data format:
        - Columns: [timestamp, x_error, y_error, z_error, yaw_error,
                   motor1_cmd, motor2_cmd, motor3_cmd, motor4_cmd]
        Or:
        - Columns: [timestamp, position_error, velocity_error, integral_error, derivative_error,
                   motor1_cmd, motor2_cmd, motor3_cmd, motor4_cmd]
        """
        if data_file is not None:
            # Load from CSV file
            data = pd.read_csv(data_file)
        elif data_array is not None:
            # Use provided numpy array
            if isinstance(data_array, np.ndarray):
                # Create column names if array provided
                error_cols = [f'error_{i}' for i in range(self.error_features)]
                control_cols = [f'motor_{i + 1}_cmd' for i in range(self.control_features)]
                columns = ['timestamp'] + error_cols + control_cols
                data = pd.DataFrame(data_array, columns=columns[:data_array.shape[1]])
            else:
                data = pd.DataFrame(data_array)
        else:
            # Generate synthetic data for demonstration
            print("No data provided, generating synthetic PID-like data...")
            data = self._generate_synthetic_pid_data()

        # Define expected column patterns
        error_patterns = ['error', 'err', 'e_']
        control_patterns = ['motor', 'cmd', 'control', 'output', 'pwm']

        # Auto-detect error and control columns
        error_cols = []
        control_cols = []

        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in error_patterns) and 'motor' not in col_lower:
                error_cols.append(col)
            elif any(pattern in col_lower for pattern in control_patterns):
                control_cols.append(col)

        # Fallback: assume first N columns after timestamp are errors, last M are controls
        if len(error_cols) == 0 or len(control_cols) == 0:
            print("Auto-detection failed, using column position...")
            all_cols = [col for col in data.columns if 'time' not in col.lower()]
            error_cols = all_cols[:self.error_features]
            control_cols = all_cols[-self.control_features:]

        print(f"Error columns: {error_cols}")
        print(f"Control columns: {control_cols}")

        # Update features based on actual data
        self.error_features = len(error_cols)
        self.control_features = len(control_cols)

        # Extract data
        errors = data[error_cols].values
        controls = data[control_cols].values

        # Remove any rows with NaN or infinite values
        valid_idx = np.isfinite(errors).all(axis=1) & np.isfinite(controls).all(axis=1)
        errors = errors[valid_idx]
        controls = controls[valid_idx]

        print(f"Data shape - Errors: {errors.shape}, Controls: {controls.shape}")
        print(f"Error range: [{errors.min():.3f}, {errors.max():.3f}]")
        print(f"Control range: [{controls.min():.3f}, {controls.max():.3f}]")

        # Normalize the data
        errors_scaled = self.error_scaler.fit_transform(errors)
        controls_scaled = self.control_scaler.fit_transform(controls)

        return self._create_sequences(errors_scaled, controls_scaled)

    def _generate_synthetic_pid_data(self, n_samples=15000):
        """Generate synthetic PID controller data (Error -> Control)"""
        np.random.seed(42)
        t = np.linspace(0, 150, n_samples)
        dt = t[1] - t[0]

        # Simulate various error signals that a quadcopter might experience

        # 1. Position errors (step responses, ramps, oscillations)
        x_error = np.zeros(n_samples)
        y_error = np.zeros(n_samples)
        z_error = np.zeros(n_samples)
        yaw_error = np.zeros(n_samples)

        # Add different types of errors throughout the flight
        for i in range(n_samples):
            # Step inputs
            if 1000 < i < 3000:
                x_error[i] = 2.0
            elif 5000 < i < 7000:
                y_error[i] = -1.5
            elif 9000 < i < 11000:
                z_error[i] = 1.0

            # Sinusoidal tracking errors
            x_error[i] += 0.5 * np.sin(0.01 * i) * np.exp(-0.0001 * i)
            y_error[i] += 0.3 * np.cos(0.015 * i) * np.exp(-0.0001 * i)
            z_error[i] += 0.2 * np.sin(0.008 * i) * np.exp(-0.0001 * i)
            yaw_error[i] += 0.1 * np.sin(0.005 * i) * np.exp(-0.0001 * i)

            # Add noise
            x_error[i] += np.random.normal(0, 0.05)
            y_error[i] += np.random.normal(0, 0.05)
            z_error[i] += np.random.normal(0, 0.03)
            yaw_error[i] += np.random.normal(0, 0.02)

        # PID parameters (typical values)
        # Position PID gains
        kp_pos = 3.0
        ki_pos = 0.5
        kd_pos = 1.2

        # Yaw PID gains
        kp_yaw = 2.0
        ki_yaw = 0.1
        kd_yaw = 0.8

        # Initialize PID variables
        integral_x = integral_y = integral_z = integral_yaw = 0
        prev_x = prev_y = prev_z = prev_yaw = 0

        # Base motor command (hovering)
        base_cmd = 1500

        # Initialize motor commands
        motor1_cmd = np.zeros(n_samples)
        motor2_cmd = np.zeros(n_samples)
        motor3_cmd = np.zeros(n_samples)
        motor4_cmd = np.zeros(n_samples)

        # Simulate PID controller response
        for i in range(n_samples):
            # Current errors
            ex, ey, ez, eyaw = x_error[i], y_error[i], z_error[i], yaw_error[i]

            # Integral terms (with windup protection)
            integral_x = np.clip(integral_x + ex * dt, -10, 10)
            integral_y = np.clip(integral_y + ey * dt, -10, 10)
            integral_z = np.clip(integral_z + ez * dt, -10, 10)
            integral_yaw = np.clip(integral_yaw + eyaw * dt, -5, 5)

            # Derivative terms
            if i > 0:
                deriv_x = (ex - prev_x) / dt
                deriv_y = (ey - prev_y) / dt
                deriv_z = (ez - prev_z) / dt
                deriv_yaw = (eyaw - prev_yaw) / dt
            else:
                deriv_x = deriv_y = deriv_z = deriv_yaw = 0

            # PID control outputs
            pid_x = kp_pos * ex + ki_pos * integral_x + kd_pos * deriv_x
            pid_y = kp_pos * ey + ki_pos * integral_y + kd_pos * deriv_y
            pid_z = kp_pos * ez + ki_pos * integral_z + kd_pos * deriv_z
            pid_yaw = kp_yaw * eyaw + ki_yaw * integral_yaw + kd_yaw * deriv_yaw

            # Convert PID outputs to motor commands (quadcopter mixing)
            # Motor layout: 1=front-right, 2=back-left, 3=front-left, 4=back-right
            motor1_cmd[i] = base_cmd + pid_z - pid_x + pid_y + pid_yaw
            motor2_cmd[i] = base_cmd + pid_z + pid_x - pid_y + pid_yaw
            motor3_cmd[i] = base_cmd + pid_z + pid_x + pid_y - pid_yaw
            motor4_cmd[i] = base_cmd + pid_z - pid_x - pid_y - pid_yaw

            # Add some noise to motor commands
            motor1_cmd[i] += np.random.normal(0, 5)
            motor2_cmd[i] += np.random.normal(0, 5)
            motor3_cmd[i] += np.random.normal(0, 5)
            motor4_cmd[i] += np.random.normal(0, 5)

            # Limit motor commands to realistic range
            motor1_cmd[i] = np.clip(motor1_cmd[i], 1000, 2000)
            motor2_cmd[i] = np.clip(motor2_cmd[i], 1000, 2000)
            motor3_cmd[i] = np.clip(motor3_cmd[i], 1000, 2000)
            motor4_cmd[i] = np.clip(motor4_cmd[i], 1000, 2000)

            # Store previous errors for derivative calculation
            prev_x, prev_y, prev_z, prev_yaw = ex, ey, ez, eyaw

        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': t,
            'x_error': x_error,
            'y_error': y_error,
            'z_error': z_error,
            'yaw_error': yaw_error,
            'motor1_cmd': motor1_cmd,
            'motor2_cmd': motor2_cmd,
            'motor3_cmd': motor3_cmd,
            'motor4_cmd': motor4_cmd
        })

        return data

    def _create_sequences(self, errors, controls):
        """Create sequences for RNN training"""
        X_seq, y_seq = [], []

        for i in range(len(errors) - self.sequence_length):
            X_seq.append(errors[i:(i + self.sequence_length)])
            y_seq.append(controls[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, rnn_type='LSTM', hidden_units=64, num_layers=2, dropout_rate=0.2):
        """
        Build the RNN model for Error -> Control mapping
        """
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(self.sequence_length, self.error_features)))

        # RNN layers
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)

            if rnn_type == 'LSTM':
                model.add(layers.LSTM(hidden_units,
                                      return_sequences=return_sequences,
                                      dropout=dropout_rate,
                                      recurrent_dropout=dropout_rate))
            elif rnn_type == 'GRU':
                model.add(layers.GRU(hidden_units,
                                     return_sequences=return_sequences,
                                     dropout=dropout_rate,
                                     recurrent_dropout=dropout_rate))
            else:  # SimpleRNN
                model.add(layers.SimpleRNN(hidden_units,
                                           return_sequences=return_sequences,
                                           dropout=dropout_rate))

        # Dense layers for control output
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.control_features, activation='linear'))

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, verbose=1):
        """Train the model"""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict_control(self, error_sequence):
        """
        Predict control commands from error sequence

        Args:
            error_sequence: Array of shape (sequence_length, error_features) or
                           (batch_size, sequence_length, error_features)

        Returns:
            control_commands: Denormalized control commands
        """
        if error_sequence.ndim == 2:
            # Single sequence
            error_sequence = error_sequence.reshape(1, -1, self.error_features)

        # Normalize errors
        batch_size, seq_len, features = error_sequence.shape
        error_flat = error_sequence.reshape(-1, features)
        error_scaled = self.error_scaler.transform(error_flat)
        error_scaled = error_scaled.reshape(batch_size, seq_len, features)

        # Predict
        control_scaled = self.model.predict(error_scaled, verbose=0)

        # Denormalize controls
        control_commands = self.control_scaler.inverse_transform(control_scaled)

        return control_commands

    def real_time_predict(self, current_errors, error_history):
        """
        Real-time prediction for deployment

        Args:
            current_errors: Current error values [x_err, y_err, z_err, yaw_err]
            error_history: List/array of previous errors (sequence_length-1 previous steps)

        Returns:
            motor_commands: [motor1, motor2, motor3, motor4]
        """
        # Build sequence
        if len(error_history) >= self.sequence_length - 1:
            sequence = np.array(error_history[-(self.sequence_length - 1):] + [current_errors])
        else:
            # Pad with zeros if not enough history
            padding = np.zeros((self.sequence_length - 1 - len(error_history), self.error_features))
            sequence = np.vstack([padding, error_history, current_errors])

        # Predict
        control_commands = self.predict_control(sequence)

        return control_commands[0]  # Return single prediction

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.control_scaler.inverse_transform(y_pred_scaled)
        y_test_orig = self.control_scaler.inverse_transform(y_test)

        # Calculate metrics
        mse = np.mean((y_pred - y_test_orig) ** 2)
        mae = np.mean(np.abs(y_pred - y_test_orig))
        rmse = np.sqrt(mse)

        # Per-motor metrics
        motor_mse = np.mean((y_pred - y_test_orig) ** 2, axis=0)
        motor_mae = np.mean(np.abs(y_pred - y_test_orig), axis=0)

        print(f"Overall - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        for i in range(self.control_features):
            print(f"Motor {i + 1} - MSE: {motor_mse[i]:.4f}, MAE: {motor_mae[i]:.4f}")

        return {
            'mse': mse, 'mae': mae, 'rmse': rmse,
            'motor_mse': motor_mse, 'motor_mae': motor_mae
        }

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True)

        # MAE plot
        ax2.plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, X_test, y_test, num_samples=200):
        """Plot predictions vs actual values"""
        y_pred_scaled = self.model.predict(X_test[:num_samples])
        y_pred = self.control_scaler.inverse_transform(y_pred_scaled)
        y_test_orig = self.control_scaler.inverse_transform(y_test[:num_samples])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        motor_names = ['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4']

        for i in range(min(4, self.control_features)):
            axes[i].plot(y_test_orig[:, i], label='Actual (PID)', alpha=0.8, linewidth=1.5)
            axes[i].plot(y_pred[:, i], label='Predicted (RNN)', alpha=0.8, linewidth=1.5)
            axes[i].set_title(f'{motor_names[i]} Command')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('PWM Command')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """Save the trained model and scalers"""
        self.model.save(filepath)

        # Save scalers
        import joblib
        joblib.dump(self.error_scaler, filepath.replace('.h5', '_error_scaler.pkl'))
        joblib.dump(self.control_scaler, filepath.replace('.h5', '_control_scaler.pkl'))

        print(f"Model and scalers saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model and scalers"""
        self.model = keras.models.load_model(filepath)

        # Load scalers
        import joblib
        self.error_scaler = joblib.load(filepath.replace('.h5', '_error_scaler.pkl'))
        self.control_scaler = joblib.load(filepath.replace('.h5', '_control_scaler.pkl'))

        print(f"Model and scalers loaded from {filepath}")


# Example usage and training script
def main():
    """Main training and evaluation script"""
    print("=== RNN Controller Training (Error -> Control) ===\n")

    # Initialize the controller
    controller = ErrorToControlRNNController(
        sequence_length=15,  # Look back 15 time steps
        error_features=4,  # x, y, z, yaw errors
        control_features=4  # 4 motor commands
    )

    # Load and preprocess data
    print("Loading and preprocessing data...")
    # For your real data: X, y = controller.load_and_preprocess_data('your_pid_data.csv')
    X, y = controller.load_and_preprocess_data()  # Uses synthetic data for demo

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(f"Training sequences: {X_train.shape[0]}")
    print(f"Validation sequences: {X_val.shape[0]}")
    print(f"Test sequences: {X_test.shape[0]}")
    print(f"Sequence length: {X_train.shape[1]}")
    print(f"Error features: {X_train.shape[2]}")
    print(f"Control outputs: {y_train.shape[1]}\n")

    # Build and compile model
    print("Building model...")
    model = controller.build_model(
        rnn_type='LSTM',
        hidden_units=64,
        num_layers=2,
        dropout_rate=0.2
    )
    controller.compile_model(learning_rate=0.001)

    print("\nModel Summary:")
    model.summary()

    # Train the model
    print("\nTraining the model...")
    history = controller.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )

    # Plot training history
    controller.plot_training_history(history)

    # Evaluate
    print("\nEvaluating the model...")
    metrics = controller.evaluate_model(X_test, y_test)

    # Plot predictions
    print("\nPlotting predictions...")
    controller.plot_predictions(X_test, y_test)

    # Save the model
    controller.save_model('error_to_control_rnn.h5')

    # Demonstrate real-time usage
    print("\n=== Real-time Usage Example ===")

    # Simulate some error values
    current_errors = [0.5, -0.3, 0.1, 0.05]  # [x_err, y_err, z_err, yaw_err]
    error_history = [[0.4, -0.2, 0.0, 0.0], [0.45, -0.25, 0.05, 0.02]]  # Previous errors

    motor_commands = controller.real_time_predict(current_errors, error_history)
    print(f"Input errors: {current_errors}")
    print(f"Output motor commands: {motor_commands}")

    return controller, history, metrics


if __name__ == "__main__":
    controller, history, metrics = main()