import numpy as np
import pandas as pd


import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# Hyperparameters grid
lam = [0.1,0.01, 0.001, 0.0001, 0.00001]
learning_rate = [0.1, 0.01, 0.001]
num_iters = 1000


def read_data(file_path):
    """Read CSV data into a pandas DataFrame."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data: separate features and target, normalize features, and split into training and validation sets."""
    y = df.pop('Revenue_Growth')

    x = df

    feature_names = x.columns.tolist()
    x = jnp.array(x)
    y = jnp.array(y)

    # Normalization
    mean_x = x.mean(0)
    std_x = x.std(0)

    x_norm = (x - mean_x) / std_x
    x_norm = jnp.column_stack([jnp.ones(len(x)), x_norm])

    x_train , x_val, y_train, y_val = train_test_split(x_norm, y, test_size=0.2, random_state=42)
    
    return x_norm, y, x_train, x_val, y_train, y_val, feature_names


def model(X, teeta):
    return X @ teeta

@jax.jit
def Loss(teeta, X, y, lam):
    """Mean squared error with L1 regularization (Lasso)."""
    prediction = model(X, teeta)
    error = (prediction - y) ** 2
    mse = jnp.mean(error)
    # L1 regularization (exclude bias term teeta[0])
    reg = lam * jnp.sum(jnp.abs(teeta[1:]))
    return mse + reg

@jax.jit
def gradient_descent(teeta, X, y, lam, learning_rate):
    """Perform a single step of gradient descent using JAX's automatic differentiation."""
    grad_loss = jax.grad(Loss)(teeta, X, y, lam)
    teeta = teeta - learning_rate * grad_loss
    return teeta

@jax.jit
def evaluate_model(teeta, X, y):
    """Evaluate MSE using JAX ops (safe for JAX arrays)."""
    preds = model(X, teeta)
    mse = jnp.mean((preds - y) ** 2)
    return mse


def sample(N, n, X, y):
    """Randomly sample n data points from the dataset."""
    if n > N:
        raise ValueError("Sample size n cannot be greater than population size N.")
    idx = np.random.choice(N, n)
    return X[idx], y[idx]


def train_model(teeta, X, y, lam, learning_rate, n, num_iters=1000):
    """Train the model using mini-batch gradient descent."""
    N = X.shape[0]
    for i in range(num_iters):
        xi, yi = sample(N, n, X, y)
        teeta = gradient_descent(teeta, xi, yi, lam, learning_rate)
        if i % 500 == 0:
            current_loss = Loss(teeta, X, y, lam)
            print('iter', i, 'loss:', float(current_loss))
    return teeta


def pick_random_grid(grid_size, lam, learning_rate):
    """Pick a random subset of hyperparameter combinations."""
    if grid_size > len(lam) * len(learning_rate):
        raise ValueError("Grid size exceeds the number of unique hyperparameter combinations.")
    
    chosen_params = set()
    while len(chosen_params) < grid_size:
        l = np.random.choice(lam)
        lr = np.random.choice(learning_rate)
        chosen_params.add((l, lr))
    return list(chosen_params)

def hyperparameter_tuning(l, lr, x_train, y_train, x_val, y_val, num_iters=1000):
    """Tune hyperparameters and return validation MSE."""
    print("Lambda:", l, "Learning Rate:", lr)
    # initialize theta as a JAX array and match dimensionality of X_train
    teeta = jnp.zeros(x_train.shape[1])
    # Train model (bias/intercept column kept in X_norm; we excluded it from reg)
    trained_theta = train_model(teeta, x_train, y_train, l, lr, 1000 ,num_iters)

    return l, lr, evaluate_model(trained_theta, x_val, y_val), trained_theta



def grid_search(lam, learning_rate, x_train, y_train, x_val, y_val, num_iters=1000):
    """Perform a full grid search over hyperparameters."""
    results = []
    for l in lam:
        for lr in learning_rate:
            print("Lambda:", l, "Learning Rate:", lr)
            # initialize theta as a JAX array and match dimensionality of X_train
            teeta = jnp.zeros(x_train.shape[1])
            # Train model (bias/intercept column kept in X_norm; we excluded it from reg)
            trained_theta = train_model(teeta, x_train, y_train, l, lr, 1000 ,num_iters)
            val_mse = evaluate_model(trained_theta, x_val, y_val)
            train_mse = evaluate_model(trained_theta, x_train, y_train)
            results.append({
                "Lambda": l,
                "Learning Rate": lr,
                "Validation MSE": val_mse,
                "Training MSE": train_mse,
                "Trained Theta": trained_theta
            })
    return pd.DataFrame(results)


def combined_grid_search(x_train, y_train, x_val, y_val, lam, learning_rate, grid_size=5, num_iters=1000):
    """Combine random grid search and narrowed full grid search."""

    # Stage 1: Random grid search using hyperparameter_tuning()
    random_combos = pick_random_grid(grid_size, lam, learning_rate)
    stage1_results = []

    for l, lr in random_combos:
        l, lr, val_mse, trained_theta = hyperparameter_tuning(l, lr, x_train, y_train, x_val, y_val, num_iters)
        train_mse = evaluate_model(trained_theta, x_train, y_train)
        stage1_results.append({
            "Lambda": l,
            "Learning Rate": lr,
            "Validation MSE": val_mse,
            "Training MSE": train_mse,
            "Trained Theta": trained_theta
        })

    stage1_df = pd.DataFrame(stage1_results)
    best_row = stage1_df.loc[stage1_df["Validation MSE"].idxmin()]
    best_lam, best_lr = best_row["Lambda"], best_row["Learning Rate"]

    # Stage 2: Narrowed full grid search using grid_search()
    def narrow(values, center, factor=10):
        return sorted(set([
            center,
            center / factor,
            center * factor
        ]) & set(values))  # keep only values in original list

    narrowed_lam = narrow(lam, best_lam)
    narrowed_lr = narrow(learning_rate, best_lr)

    print(f"\n[Stage 2] Narrowed Lambda: {narrowed_lam}, Learning Rate: {narrowed_lr}")
    final_df = grid_search(narrowed_lam, narrowed_lr, x_train, y_train, x_val, y_val, num_iters)

    return final_df

def cross_validate_evaluation(x_data, y_data, trained_theta, k=5):
    """Perform k-fold cross-validation to evaluate model stability."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(kf.split(x_data)):
        print(f"\nFold {fold + 1}/{k}")

        x_val_fold = x_data[val_index]
        y_val_fold = y_data[val_index]
        val_mse = evaluate_model(trained_theta, x_val_fold, y_val_fold)

        fold_results.append({
            "Fold": fold + 1,
            "Validation MSE": float(val_mse)
        })

    return fold_results

def test_data_read(file_path):
    """Read and preprocess test data."""
    df = pd.read_csv(file_path)
    y = df.pop('Revenue_Growth')
    x = df
    x = jnp.array(x)
    y = jnp.array(y)
    mean_x = x.mean(0)
    std_x = x.std(0)
    x_norm = (x - mean_x) / std_x
    x_norm = jnp.column_stack([jnp.ones(len(x)), x_norm])
    return x_norm

def test_prediction(x_test, trained_theta):
    """Generate predictions on test data."""
    predictions = model(x_test, trained_theta)
    
    return pd.DataFrame(predictions, columns=["Revenue_Growth"])

def output_predictions(predictions, filename='predictions.csv'):
    """Output predictions to a CSV file."""
    predictions.to_csv(filename, index=True)

if __name__ == "__main__":
    # Load and preprocess data  
    file_path = 'train.csv'
    df = read_data(file_path)
    x_norm, y, x_train, x_val, y_train, y_val, feature_names = preprocess_data(df)

    # Hyperparameter tuning and model training
    tuned_df = combined_grid_search(x_train, y_train, x_val, y_val, lam=lam, learning_rate=learning_rate, grid_size=6)
    best_row = tuned_df.loc[tuned_df["Validation MSE"].idxmin()]
    print(f"\nBest Hyperparameters from Combined Search: Lambda: {best_row['Lambda']} Learning Rate: {best_row['Learning Rate']} with Validation MSE: {best_row['Validation MSE']}")
    trained_theta = best_row['Trained Theta']

    fold_results = cross_validate_evaluation(x_norm, y, trained_theta, k=5)
    print("Cross-Validation Results: to verify model stability")
    print(fold_results)

    x_test = test_data_read('test.csv')
    prediction = test_prediction(x_test, trained_theta)
    output_predictions(prediction, filename='predictions.csv')

    
