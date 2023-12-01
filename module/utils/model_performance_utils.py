import statistics

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def check_out_cross_val_score(estimator, cap_x_df, y_df, cv=5):
    scores = cross_val_score(
        estimator,
        cap_x_df,
        y_df.values.ravel(),
        cv=cv
    )
    print(f"Cross Validation Scores: {scores}")
    print(f"mean: {scores.mean()}")
    print(f"standard deviation: {statistics.stdev(scores)}")


def check_out_permutation_importance(estimator, cap_x_df, y_df, permutation_importance_random_state):
    print("\nPermutation importance:")
    r_multi = permutation_importance(
        estimator,
        cap_x_df,
        y_df,
        n_repeats=10,
        random_state=permutation_importance_random_state,
        scoring=['neg_mean_squared_error']
    )

    for metric in r_multi:
        temp_metric = metric
        if metric == 'neg_mean_squared_error':
            temp_metric = 'sqrt_' + metric
        print(f"\nmetric: {temp_metric}")
        r = r_multi[metric]
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                feature_name = cap_x_df.columns[i]
                mean = r.importances_mean[i]
                std_dev = r.importances_std[i]
                if metric == 'neg_mean_squared_error':
                    mean = np.sqrt(mean)
                    std_dev = np.sqrt(std_dev)
                print(
                    f"    {feature_name:<8}"
                    f" {mean:.3f}"
                    f" +/- {std_dev:.3f}"
                )


def check_out_error_types(estimator, cap_x_df, y_df):
    print("\nCheck out type I and type II errors")
    y_pred = estimator.predict(cap_x_df)
    tn, fp, fn, tp = confusion_matrix(y_df, y_pred).ravel()
    print(f"type I error (false positive): {fp}")
    print(f"type II error (false negative): {fn}")
