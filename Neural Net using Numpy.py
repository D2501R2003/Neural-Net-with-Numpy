import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  # x is the sigmoid ACTIVATION (not the input to sigmoid)
    return x * (1 - x)

import numpy as np

def parse_numbers(line):
    """Clean a line: replace commas → spaces, remove brackets, split, convert to float"""
    cleaned = line.replace(',', ' ').replace('[', '').replace(']', '').strip()
    parts = cleaned.split()
    numbers = []
    for p in parts:
        if p.strip():  # skip empty
            try:
                numbers.append(float(p))
            except ValueError:
                pass  # ignore invalid tokens
    return numbers

# ────────────────────────────────────────────────
print("Let's create your training data!\n")

# Ask for dimensions
n_samples = int(input("How many examples / rows do you want? (e.g. 5)   : ").strip())
n_inputs  = int(input("How many input features per example? (e.g. 4)  : ").strip())
n_outputs = int(input("How many output values per example? (e.g. 2)   : ").strip())

print(f"\nOkay → collecting {n_samples} examples")
print(f"  → each with {n_inputs} input values")
print(f"  → each with {n_outputs} output values\n")

# ────────────────────────────────────────────────
print("=== INPUT DATA ===")
print(f"Enter {n_samples} lines — each should contain {n_inputs} numbers")
print("   (spaces or commas between numbers — extra spaces are okay)\n")

X_rows = []
for i in range(n_samples):
    while True:
        line = input(f"Example {i+1}/{n_samples}  → input: ").strip()
        nums = parse_numbers(line)
        
        if len(nums) == n_inputs:
            X_rows.append(nums)
            break
        else:
            print(f"   → Got {len(nums)} numbers, but expected {n_inputs}. Try again.")

X = np.array(X_rows)

# ────────────────────────────────────────────────
print("\n=== OUTPUT DATA ===")
print(f"Enter {n_samples} lines — each should contain {n_outputs} numbers\n")

y_rows = []
for i in range(n_samples):
    while True:
        line = input(f"Example {i+1}/{n_samples}  → output: ").strip()
        nums = parse_numbers(line)
        
        if len(nums) == n_outputs:
            y_rows.append(nums)
            break
        else:
            print(f"   → Got {len(nums)} numbers, but expected {n_outputs}. Try again.")

y = np.array(y_rows)

# Optional: normalize outputs (common in regression problems)
normalize_outputs = input("\nDivide outputs by 10? (y/n): ").strip().lower().startswith('y')
if normalize_outputs:
    y = y / 10.0

# ────────────────────────────────────────────────
print("\n" + "═" * 50)
training_input = np.array([row.tolist() for row in X])

training_output = np.array([row.tolist() for row in y])

print("\nYour training input:")
print(training_input)
print("\nYour training output:")
print(training_output)

np.random.seed(42)

input_size   = int(training_input.shape[1])
output_size  = int(training_output.shape[1])

# 1. Normalize inputs (very important!)
X_mean = training_input.mean(axis=0)
X_std  = training_input.std(axis=0) + 1e-8     # avoid div by zero
X      = (training_input - X_mean) / X_std

# 2. Bigger & better hidden layer + better init
hidden_size = 32                               # ← much more capacity

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)   # He init
b1 = np.zeros((1, hidden_size))

# 3. Linear output layer ← most important change for regression
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.03                           # smaller & safer
epochs = 15000

print("\nStarting training (normalized inputs + linear output layer)...\n")

for epoch in range(epochs):
    # Forward
    hidden = sigmoid(np.dot(X, W1) + b1)
    output = np.dot(hidden, W2) + b2            # ← no sigmoid here

    # Error
    error = training_output - output
    loss  = np.mean(np.abs(error))

    # Backward
    d_output = error                            # ← no sigmoid derivative on output

    error_hidden = np.dot(d_output, W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden)

    # Updates
    W2 += learning_rate * np.dot(hidden.T, d_output)
    b2 += learning_rate * np.sum(d_output, axis=0, keepdims=True)

    W1 += learning_rate * np.dot(X.T, d_hidden)
    b1 += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    if (epoch + 1) % 3000 == 0 or epoch == epochs-1:
        print(f"Epoch {epoch+1:6d} | MAE = {loss:.6f}")

# Final predictions
hidden_final = sigmoid(np.dot(X, W1) + b1)
predictions = np.dot(hidden_final, W2) + b2

print("\nFinal predictions (×10):")
print(np.round(predictions * 10, 2))

print("\nTargets (×10):")
print(training_output * 10)

mae_scaled = np.mean(np.abs(predictions - training_output)) * 10
print(f"\nMean absolute error (scaled): {mae_scaled:.3f}")