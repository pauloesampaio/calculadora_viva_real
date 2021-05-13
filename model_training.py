from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from utils.utils import load_config, mape, create_summary
import pandas as pd
import shap

# Loads configuration file
config = load_config()

# Loads model input data
df = pd.read_csv(config["paths"]["model_input"])

# Set x and y
X = df[config["model"]["numerical_features"] + config["model"]["categorical_features"]]
y = df[config["model"]["target"]]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["model"]["test_size"], random_state=1234
)

# Instantiates model
model = CatBoostRegressor(
    random_state=1234,
    cat_features=config["model"]["categorical_features"],
    verbose=False,
)

# If on configuration file there's the grid search flag
# uses grid search to find best parameters for model
# or else get pre-defined parameters from config file
if config["model"]["grid_search"]:
    print("Running grid search")
    gs = GridSearchCV(
        estimator=model,
        param_grid=config["model"]["param_grid"],
        cv=3,
        refit=True,
        verbose=2,
    )
    gs.fit(X_train, y_train)
    print(f"Best parameters: {gs.best_params_}")
    model = model.set_params(**gs.best_params_)
    print(f"Saving report to {config['paths']['grid_search_report']}")
    pd.DataFrame(gs.cv_results_).to_csv(
        config["paths"]["grid_search_report"], index=False
    )
else:
    print("Training model without grid search")
    model = model.set_params(**config["model"]["model_params"])

# Trains model
model.fit(X_train, y_train)
# Predicts
y_pred = model.predict(X_test)

# Calculate metrics (mape and r2)
mape_test = mape(y_test, y_pred)
mape_train = mape(y_train, model.predict(X_train))
r2_test = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, model.predict(X_train))
metrics_df = pd.DataFrame(
    [
        ["mape_test", mape_test],
        ["mape_train", mape_train],
        ["r2_test", r2_test],
        ["r2_train", r2_train],
    ],
    columns=["metric", "result"],
)

# Calculate shape values for explainability
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Gets feature importances
feature_importances = model.get_feature_importance(prettified=True)

# Creates model summary
summary = create_summary(y_test, y_pred, shap_values, feature_importances)

# Prints metrics
print(f"Test set mape: {mape_test:.4f}, train set mape: {mape_train:.4f}")
print(f"Test set r2: {r2_test:.4f}, train set r2: {r2_train:.4f}")

# Save reports
print(f"Saving summary report to {config['paths']['summary_report']}")
summary.savefig(config["paths"]["summary_report"])
print(f"Saving metrics report to {config['paths']['metrics_report']}")
metrics_df.to_csv(config["paths"]["metrics_report"], index=False)

# Saves model
print(f"Saving model to {config['paths']['model']}")
model.save_model(config["paths"]["model"])
