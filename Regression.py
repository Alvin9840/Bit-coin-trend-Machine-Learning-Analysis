import pandas as pd
import os
import warnings
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm


warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "12"
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 5)
RANDOM = 5805
CACHE_PATH = "agg_df_cache.parquet"

FEATURES = [
    'volatility',
    'volume',
    'number_of_trades',
    'taker_buy_quote_asset_volume'
]


def regression_feature_engineering(df, lags=1):
    df = df.copy()
    base_features = FEATURES

    for col in base_features:
        df[f'{col}_lag1'] = df[col].shift(1)

    # Drop rows with missing lag values
    lag_cols = [f'{col}_lag1' for col in base_features]
    df = df.dropna(subset=lag_cols)

    # Feature columns: all lag-1 features
    feature_cols = lag_cols

    # Standardization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, feature_cols


def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print("Add  {:30} with p-value {:.6f}".format(best_feature, best_pval))

        # Backward step
        if included:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print("Drop {:30} with p-value {:.6f}".format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def regression(agg_df):
    print("\n===== Phase II: Regression Analysis =====\n")

    agg_df_reg, feature_cols = regression_feature_engineering(agg_df, lags=1)
    X = agg_df_reg[feature_cols]
    y = agg_df_reg['close']

    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Stepwise feature selection
    selected_features = stepwise_selection(X_train, y_train, verbose=True)
    print('\nSelected features after stepwise regression:', selected_features)

    # Fit OLS model with selected features
    X_train_sel = sm.add_constant(X_train[selected_features])
    X_test_sel = sm.add_constant(X_test[selected_features])
    model = sm.OLS(y_train, X_train_sel).fit()
    y_train_pred = model.predict(X_train_sel)
    y_test_pred = model.predict(X_test_sel)

    # Model summary (t-test, F-test, R2, adj-R2, AIC, BIC)
    print(model.summary())

    # Train/Test MSE
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"\nTrain MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train.values, label='Train Actual', alpha=0.7)
    plt.plot(y_train.index, y_train_pred, label='Train Predicted', alpha=0.7, linestyle='--')
    plt.plot(y_test.index, y_test.values, label='Test Actual', alpha=0.7)
    plt.plot(y_test.index, y_test_pred, label='Test Predicted', alpha=0.7, linestyle='--')
    plt.xlabel('Timestamp')
    plt.ylabel('Close Price')
    plt.title('Linear Regression: Actual vs Predicted (Lag-1 Features)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 95% confidence intervals
    print("\n95% Confidence Intervals for regression coefficients:\n")
    print(round(model.conf_int(), 4))

    return model



if __name__ == '__main__':
    df = pd.read_csv("BTCUSDT.csv", low_memory=False)

    from Feature_Engineer import  load_or_create_agg_df
    agg_df = load_or_create_agg_df(df)

    # Phase II: Regression Analysis
    model = regression(agg_df)

    print("\n===== Analysis Complete =====")