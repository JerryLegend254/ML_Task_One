import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data.csv")

X = df['SIZE'].values.reshape(-1, 1)
y = df['PRICE'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()


def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def gradient_descent(X, y, m, c, learning_rate=0.01, momentum=0.9, epochs=10):
    n = len(X)
    errors = []

    v_m = 0
    v_c = 0

    best_error = float('inf')
    best_params = (m, c)

    for epoch in range(epochs):
        y_pred = m * X + c

        error = compute_mse(y, y_pred)
        errors.append(error)

        if error < best_error:
            best_error = error
            best_params = (m, c)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Error: {error:.4f}")

        dm = (-2/n) * np.sum(X * (y - y_pred))
        dc = (-2/n) * np.sum(y - y_pred)

        v_m = momentum * v_m - learning_rate * dm
        v_c = momentum * v_c - learning_rate * dc

        m = m + v_m
        c = c + v_c

    return best_params[0], best_params[1], errors


initial_m = 0
initial_c = 0

final_m, final_c, errors = gradient_descent(
    X_scaled.ravel(), y_scaled, initial_m, initial_c)



def convert_params_to_original_scale(m, c, scaler_X, scaler_y):
    m_original = m * (scaler_y.scale_[0] / scaler_X.scale_[0])

    c_original = (c * scaler_y.scale_[0] + scaler_y.mean_[0] -
                  m_original * scaler_X.mean_[0])

    return m_original, c_original


m_original, c_original = convert_params_to_original_scale(
    final_m, final_c, scaler_X, scaler_y)


plt.figure(figsize=(15, 5))


plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
X_line = np.array([min(X), max(X)])
y_line = m_original * X_line + c_original
plt.plot(X_line, y_line.ravel(), color='red', label='Line of best fit')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Price')
plt.title('Nairobi Office Prices: Linear Regression')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(range(1, len(errors) + 1), errors, alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (Scaled)')
plt.title('Error vs. Epoch')
plt.grid(True)

plt.tight_layout()
plt.show()


y_pred = m_original * X + c_original
final_mse = compute_mse(y, y_pred)
r2 = 1 - (np.sum((y - y_pred.ravel())**2) / np.sum((y - np.mean(y))**2))

print("\nModel Performance:")
print(f"Final MSE: {final_mse:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"\nFinal Parameters:")
print(f"Slope (m): {m_original:.4f}")
print(f"Intercept (c): {c_original:.4f}")

# Predict price for 100 sq. ft.
prediction_size = 100
predicted_price = m_original * prediction_size + c_original
print(f"\nPrediction for {prediction_size} sq. ft.:")
print(f"Predicted price: {predicted_price:.2f}")

# Create a summary DataFrame with residuals
summary_df = pd.DataFrame({
    'Actual Size': X.ravel(),
    'Actual Price': y,
    'Predicted Price': y_pred.ravel(),
    'Residual': y - y_pred.ravel()
})
print("\nResidual Summary:")
print(summary_df.describe())
