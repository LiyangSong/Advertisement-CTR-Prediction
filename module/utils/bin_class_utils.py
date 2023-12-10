from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

import module.utils.data_prepare_utils as data_prepare_utils
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score

from sklearn.inspection import permutation_importance
import seaborn as sns


def build_preprocessing_pipeline(numerical_attr_list, categorical_attr_list, attrs_to_drop, target_type, target_encoding_random_state):

    numerical_transformer = Pipeline([
        ("column_dropper", data_prepare_utils.DropColumnsTransformer(attrs_to_drop)),
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("column_dropper", data_prepare_utils.DropColumnsTransformer(attrs_to_drop)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target_encoder", TargetEncoder(target_type=target_type, random_state=target_encoding_random_state)),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_transformer, numerical_attr_list),
        ("categorical", categorical_transformer, categorical_attr_list)
    ])

    return preprocessor


def get_default_model(estimator_name, model_random_state):
    estimator_list = {
        "SGDClassifier": SGDClassifier(
            loss='log_loss',
            class_weight='balanced',
            max_iter=10000,
            random_state=model_random_state
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            criterion='log_loss',
            class_weight='balanced',
            random_state=model_random_state
        ),
        "RandomForestClassifier": DecisionTreeClassifier(
            class_weight='balanced',
            random_state=model_random_state
        ),
        "AdaBoostClassifier": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(criterion='log_loss',
                                             class_weight='balanced',
                                             random_state=model_random_state
                                             ),
            random_state=model_random_state
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=model_random_state
        )
    }

    return estimator_list[estimator_name]


def eval_class(cap_x_df, y_df, trained_estimator, data_set_name, cvs_scoring_list, threshold=0.50):

    print(f'Evaluate the trained estimator performance on {data_set_name} set')
    y_proba = trained_estimator.predict_proba(cap_x_df)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print('Check accuracy score')
    print(f'{data_set_name} set accuracy score: {accuracy_score(y_df, y_pred)}')

    print('\nCheck classification report')
    print(classification_report(y_df, y_pred))

    print('\nCheck confusion matrix')
    cm = confusion_matrix(y_df, y_pred)
    print(f'{data_set_name} set confusion matrix: \n{cm}')
    print('True Positives = ', cm[0,0])
    print('True Negatives = ', cm[1,1])
    print('False Positives(Type I error) = ', cm[0,1])
    print('False Negatives(Type II error) = ', cm[1,0])

    print('\nCheck cross validation score')
    for scoring in cvs_scoring_list:
        scores = cross_val_score(
            trained_estimator,
            cap_x_df,
            y_df.values.ravel(),
            scoring=scoring,
            cv=5,
            n_jobs=-1
        )
        print(f'\n{scoring} scores: {scores}')
        print(f'np.mean(scores): {np.mean(scores)}')
        print(f'np.std(scores, ddof=1): {np.std(scores, ddof=1)}')

    print('\nCheck the ROC Curve and AUC')
    roc_auc = roc_auc_score(y_df, y_pred)
    fpr, tpr, _ = roc_curve(y_df, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
    plt.xlabel('False Positive Rate (Sensitivity)')
    plt.ylabel('True Positive Rate (Specificity)')
    plt.title(f'{data_set_name} Set Roc Curve\n'
              f'roc_auc_score = {round(roc_auc, 4)}')
    plt.grid()
    plt.show()

    print('\nCheck Precision-Recall Curve and Average Precision Score')
    precision, recall, _ = precision_recall_curve(y_df, y_proba)
    ave_precision = average_precision_score(y_df, y_proba)
    
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{data_set_name} Set Precision-Recall Curve\n'
              f'ave_precision_score = {round(ave_precision, 4)}')
    plt.grid()
    plt.show()


def check_out_permutation_importance(estimator, cap_x_df, y_df, permutation_importance_random_state, permutation_scoring_list):
    print("\nCheck out permutation importance:")
    r_multi = permutation_importance(
        estimator,
        cap_x_df,
        y_df,
        n_repeats=10,
        random_state=permutation_importance_random_state,
        scoring=permutation_scoring_list
    )

    results = []
    for metric in r_multi:

        temp_metric = metric
        if metric == 'neg_mean_squared_error':
            temp_metric = 'sqrt_' + metric

        r = r_multi[metric]
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:

                feature_name = cap_x_df.columns[i]
                mean = r.importances_mean[i]
                std_dev = r.importances_std[i]

                if metric == 'neg_mean_squared_error':
                    mean = np.sqrt(mean)
                    std_dev = np.sqrt(std_dev)

                results.append({
                    "metric_name": temp_metric,
                    "feature_name": feature_name,
                    "metric_mean": mean,
                    "metric_std_dev": std_dev
                })

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by=["metric_name", "metric_mean"], ascending=[True, False])


# perf_dict_list = []
#
# for class_weight_name, class_weight_option in class_weight_options_dict.items():
#     estimator = SGDClassifier(loss=loss, random_state=model_random_state, class_weight=class_weight_option)
#
#     composite_estimator = Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])
#     print(f'', '*' * 60, sep='')
#     print(f'\nclass_weight {class_weight_name}')
#     roc_auc, ave_precision = \
#         bin_class_utils.eval_class(train_cap_x_df, train_y_df, composite_estimator, 'train sample')
#     row_dict = {
#         'class_weight_name': class_weight_name,
#         'class imbalance class 0': train_y_df.value_counts(normalize=True).loc[0],
#         'class imbalance class 1': train_y_df.value_counts(normalize=True).loc[1],
#         'roc_curve_auc': roc_auc,
#         'ave_precision_score': ave_precision,
#         'data_set': 'train'
#     }
#     perf_dict_list.append(row_dict)
#
# perf_dict_df = pd.DataFrame(perf_dict_list)
# perf_dict_df


# # compare the performance on the train and validation set
# uncompared_columns = ['class imbalance class 0', 'class imbalance class 1']
# best_perf_dict_df = perf_dict_df[perf_dict_df['class_weight_name'] == 'None'].drop(columns = uncompared_columns)
# pd.concat([best_perf_dict_df, val_perf_dict_df], ignore_index=True)


def drop_least_important_attrs(results_df, threshold_percent):
    attrs_below_threshold = {}

    for metric, group in results_df.groupby('metric_name'):
        max_importance = group['metric_mean'].max()
        importance_threshold = threshold_percent * max_importance

        # Find attrs below this threshold
        attrs_below = group[group['metric_mean'] < importance_threshold]['feature_name'].tolist()
        attrs_below_threshold[metric] = set(attrs_below)

    # Find common attrs across all metrics
    common_attrs = set.intersection(*attrs_below_threshold.values())

    return list(common_attrs)


def balance_class_weight(y_df):
    balanced = y_df.shape[0] / (y_df.nunique()*np.bincount(y_df))
    balanced_dict = \
    dict(
        zip(
            y_df.unique(),
            balanced
        )
    )
    balanced_and_normalized_dict = \
        dict(
            zip(
                y_df.unique(),
                balanced/sum(balanced)
                )
            )
    class_weight_options = [None,'balanced',balanced_dict,balanced_and_normalized_dict]

    return class_weight_options


def tune_hyperparameters(cap_x_df, y_df, pipe, param_grid):
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='average_precision',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(cap_x_df, y_df.values.ravel())

    print("Best estimator hyper parameters:\n", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    return best_model


def tune_hyperparameters_sgd(cap_x_df, y_df, pipe):

    param_grid = {
        'preprocessor__numerical__imputer__strategy': ['mean', 'median'],
        'preprocessor__categorical__target_encoder__smooth': ['auto'],
        'estimator__loss': ['log_loss'],
        'estimator__penalty': ['l2', 'l1', 'elasticnet'],
        'estimator__alpha': [0.0001, 0.001, 0.01],
        'estimator__max_iter': [1000, 10000],
        'estimator__l1_ratio': [0.15, 0.3, 0.5],
        'estimator__n_jobs': [None, -1],
    }

    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(cap_x_df, y_df.values.ravel())
    print("Best estimator:\n", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    return best_model
        

def avoiding_false_discoveries_class_helper(best_model, train_cap_x_df, train_y_df, validation_cap_x_df,
                                            validation_y_df, num_samples):

    # get bootstrap sample - fit and eval on bootstrap samples
    bs_results_df = get_bootstrap_sample_class(num_samples, best_model, train_cap_x_df, train_y_df,
                                               validation_cap_x_df, validation_y_df)

    # get randomized target sample - fit and eval on randomized target samples + kde on sample
    rt_results_df = get_randomized_target_sample_class(num_samples, best_model, train_cap_x_df, train_y_df,
                                                       validation_cap_x_df, validation_y_df)

    # combine the results
    results_df = pd.concat([bs_results_df, rt_results_df], axis=0)

    # plot the histogram
    plot_title = f'false discovery check'
    plot_null_and_alt_dist(data=results_df.drop(columns='roc_auc_score_'), x='ave_precision_score', hue='distribution',
                           x_label= ' ave_precision_score', y_label='counts', title=plot_title,
                           kde=False)
    plot_null_and_alt_dist(data=results_df.drop(columns='ave_precision_score'), x='roc_auc_score', hue='distribution',
                           x_label='roc_auc_score', y_label='counts', title=plot_title, kde=False)


def get_bootstrap_sample_class(num_samples, best_model, train_cap_x_df, train_y_df, validation_cap_x_df, validation_y_df):

    df_row_dict_list = []
    for i in range(0, num_samples):

        bs_cap_x_df, bs_y_df = get_bootstrap_sample(train_cap_x_df, train_y_df, i)
        best_model.fit(bs_cap_x_df, bs_y_df)
        y_pred_proba = best_model.predict_proba(validation_cap_x_df)[:, 1]

        roc_auc = roc_auc_score(validation_y_df, y_pred_proba)
        ave_precision = average_precision_score(validation_y_df, y_pred_proba)

        df_row_dict_list.append({
            'distribution': 'bootstrap_sample',
            'ave_precision_score': ave_precision,
            'roc_auc_score': roc_auc
        })

    bs_results_df = pd.DataFrame(df_row_dict_list)
    return bs_results_df


def get_bootstrap_sample(cap_x_df, y_df, random_state):

    bs_df = pd.concat([cap_x_df, y_df], axis=1).sample(
        # n=None,
        frac=1.0,
        replace=True,
        weights=None,
        random_state=random_state,
        axis=0,
        ignore_index=False
    )

    bs_cap_x_df, bs_y_df = bs_df.iloc[:, :-1], bs_df.iloc[:, -1]

    return bs_cap_x_df, bs_y_df


def get_randomized_target_sample_class(num_samples, best_model, train_cap_x_df, train_y_df, validation_cap_x_df,
                                       validation_y_df):
    df_row_dict_list = []
    for i in range(0, num_samples):
        rt_cap_x_df, rt_y_df = get_randomized_target_sample(train_cap_x_df, train_y_df, i)
        best_model.fit(rt_cap_x_df, rt_y_df)
        y_pred_proba = best_model.predict_proba(validation_cap_x_df)[:, 1]

        roc_auc = roc_auc_score(validation_y_df, y_pred_proba)
        ave_precision = average_precision_score(validation_y_df, y_pred_proba)

        df_row_dict_list.append({
            'distribution': 'randomized_target_sample',
            'ave_precision_score': ave_precision,
            'roc_auc_score': roc_auc
        })

    rt_results_df = pd.DataFrame(df_row_dict_list)
    return rt_results_df


def get_randomized_target_sample(cap_x_df, y_df, random_state):

    rt_df = pd.concat([cap_x_df, y_df], axis=1)
    target_name = rt_df.columns[-1]
    np.random.seed(random_state)
    rt_df[target_name] = np.random.permutation(rt_df[target_name])
    rt_cap_x_df, rt_y_df = rt_df.iloc[:, :-1], rt_df.iloc[:, -1]

    return rt_cap_x_df, rt_y_df


def plot_null_and_alt_dist(data, x, hue, x_label='', y_label='', title='', kde=True):

    print('\n', '*' * 50, sep='')
    print(f'means of the distributions:')
    print(data.groupby('distribution').mean())

    sns.histplot(data=data, x=x, hue=hue, bins=40, kde=kde)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def print_classification_metrics_at_thresholds(cap_x_df, y_df, trained_estimator, data_set_name, cvs_scoring_list, thresholds):

    # Evaluate the model at each threshold and print metrics
    for threshold in thresholds:
        y_proba = trained_estimator.predict_proba(cap_x_df)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        print("\nClassification Report at Threshold {:.2f}:\n".format(threshold))
        print(classification_report(y_df, y_pred))


def plot_errors_to_threshold(best_estimator, cap_x_df, y_df, data_set_name):
    df_row_dict_list = []
    for class_threshold in np.arange(0, 1.01, 0.01):
        class_1_proba_preds = best_estimator.predict_proba(cap_x_df)[:, 1]
        class_preds = np.where(class_1_proba_preds > class_threshold, 1, 0)

        conf_matrix = confusion_matrix(y_df, class_preds)

        df_row_dict_list.append({
            'tn': conf_matrix[0, 0],
            'fp': conf_matrix[0, 1],
            'fn': conf_matrix[1, 0],
            'tp': conf_matrix[1, 1],
            'class_threshold': class_threshold
        })

    results_df = pd.DataFrame(df_row_dict_list)

    sns.lineplot(data=results_df, x='class_threshold', y='fn', label='fn')
    sns.lineplot(data=results_df, x='class_threshold', y='fp', label='fp')
    plt.ylabel('fp and fn')
    plt.legend()
    plt.title(f'{data_set_name} fp and fn errors as a function of threshold')
    plt.grid()
    plt.show()