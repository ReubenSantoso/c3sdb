import pickle
import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from c3sdb.ml.data import C3SD
from c3sdb.ml.kmcm import kmcm_p_grid, KMCMulti
from c3sdb.ml.metrics import compute_metrics_train_test, train_test_summary_figure
 
def print_debug(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Record the start time
start_time = time.time()

_SRC_TAGS = [
    "zhou1016",
    "zhou0817",
    "zhen0917",
    "pagl0314",
    "righ0218",
    "nich1118",
    "may_0114",
    "moll0218",
    "hine1217",
    "hine0217",
    "hine0817",
    "groe0815",
    "bijl0517",
    "stow0817",
    "hine0119",
    "leap0219",
    "blaz0818",
    "tsug0220",
    "lian0118",
    "teja0918",
    "pola0620",
    "dodd0220",
    "celm1120",
    "belo0321",
    "ross0422",
    "baker0524", 
    "mull_1223", 
    "palm_0424", 
]

# Initialize the dataset
print_debug("Initializing the dataset.")
print_debug(f"Fetching these src tags: {_SRC_TAGS}")
data = C3SD("C3S_V2.db", datasets= _SRC_TAGS, )

print_debug("Assembling features.")

# Or handle_nan="drop"
# "impute" fills in mean of from the column
data.assemble_features(encoded_adduct=True, mqn_indices="all", handle_nan="drop")  


print_debug("Splitting the dataset into training and testing sets.")
data.train_test_split("ccs")

print_debug("Centering and scaling the data.")
data.center_and_scale()

# Save the encoder and scaler
print_debug("Saving encoder and scaler.")
data.save_encoder_and_scaler()

kmcm_svr_p_grid = kmcm_p_grid([4, 5], {"C": [1000, 10000], "gamma": [0.001, 0.1]}) 
kmcm_svr_gs = GridSearchCV(KMCMulti(n_clusters=4, seed=2345, use_estimator=SVR(cache_size=1024, tol=1e-3)),
                           param_grid=kmcm_svr_p_grid, n_jobs=-1, cv=3, scoring="neg_mean_squared_error",
                           verbose=3)

kmcm_svr_gs.fit(data.X_train_ss_, data.y_train_)
kmcm_svr_best = kmcm_svr_gs.best_estimator_

# Save the trained model
print_debug("Saving the trained model to 'c3sdb_kmcm_svr.pkl'.")
with open("c3sdb_kmcm_svr.pkl", "wb") as pf:
    pickle.dump(kmcm_svr_best, pf)

# Compute metrics for the trained model
print_debug("Computing predictions and metrics for the training data.")
y_pred_train = kmcm_svr_best.predict(data.X_train_ss_)
print_debug("Computing predictions and metrics for the testing data.")
y_pred_test = kmcm_svr_best.predict(data.X_test_ss_)

print_debug("Calculating performance metrics.")
summary = compute_metrics_train_test(data.y_train_, data.y_test_, y_pred_train, y_pred_test)

# Generate summary figure
print_debug("Generating summary figure and saving it as 'metrics_clean.png'.")
train_test_summary_figure(summary, "metrics.png")

# Record the end time
end_time = time.time()

# Calculate the total runtime
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)
print_debug(f"Total runtime: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.")

print_debug("Process completed successfully.")
