from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import module.utils.data_prepare_utils as data_prepare_utils
from sklearn.model_selection import GridSearchCV, train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

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

    preprocessor

    return preprocessor

def eval_class(x_df, y_df, composite_estimator, type):
    log_pipe = composite_estimator.fit(
        x_df,
        y_df.values.ravel()
    )

    y_pred = log_pipe.predict(x_df)
    y_proba = log_pipe.predict_proba(x_df)[:, 1]

    print('Check accuracy score')
    print(f'{type} set accuracy score: {accuracy_score(y_df, y_pred)}')

    print('\nCheck confusion matrix')
    cm = confusion_matrix(y_df, y_pred)
    print(f'{type} set confusion matrix: \n{cm}')
    print('True Positives = ', cm[0,0])
    print('True Negatives = ', cm[1,1])
    print('False Positives(Type I error) = ', cm[0,1])
    print('False Negatives(Type II error) = ', cm[1,0])

    print('\nCheck classification report')
    print(classification_report(y_df, y_pred))
    
    print('\nCheck the ROC Curve and AUC')
    roc_auc = roc_auc_score(y_df, y_pred)
    fpr, tpr, _ = roc_curve(y_df, log_pipe.predict_proba(x_df)[:,1])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate (Sensitivity)')
    plt.ylabel('True Positive Rate (Specificity)')
    plt.title('Curve')
    plt.show()

    print('\nCheck Precision-Recall Curve and Average Precision Score')
    precision, recall, _ = precision_recall_curve(y_df, y_proba)
    ave_precision = average_precision_score(y_df, y_proba)
    
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()    

    print('The roc_auc_score: ', roc_auc)
    print('Average Precision Score: ', ave_precision)

    return roc_auc, ave_precision

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

    permutation_df = pd.DataFrame(columns=['Feature', 'sqrt_neg_mean_squared_error'])

    for metric in r_multi:

        r = r_multi[metric]
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                feature_name = cap_x_df.columns[i]
                mean = r.importances_mean[i]
                std_dev = r.importances_std[i]
                if metric == 'neg_mean_squared_error':
                    mean = np.sqrt(mean)
                    std_dev = np.sqrt(std_dev)
                
                sqrt_neg_mean_squared_error = f"{mean:.3f} +/- {std_dev:.3f}"

                row_data = {
                    'Feature': feature_name,
                    'sqrt_neg_mean_squared_error': sqrt_neg_mean_squared_error
                }
                permutation_df.loc[len(permutation_df)] = row_data
        return permutation_df

def print_permutation_importance_all(estimator, cap_x_df, y_df, permutation_importance_random_state, score):
    print("\nPermutation importance:")
    r_multi = permutation_importance(
        estimator,
        cap_x_df,
        y_df,
        n_repeats=10,
        random_state=permutation_importance_random_state,
        scoring=[score]
    )
    for metric in r_multi:

        r = r_multi[metric]
        for i in r.importances_mean.argsort()[::-1]:
            mean_minus_two_std = r.importances_mean[i] - 2 * r.importances_std[i]
            if mean_minus_two_std > 0:

                feature_name = cap_x_df.columns[i]
                mean = r.importances_mean[i]
                std_dev = r.importances_std[i]
                
                print(
                    f"    {feature_name:<8}"
                    f" {mean:.3f}"
                    f" +/- {std_dev:.3f}"
                )

def tune_hyperparameters_dt(x_df, y_df, pipe):

    param_grid = {
        'preprocessor__numerical__imputer__strategy': ['mean', 'median'],
        'preprocessor__categorical__target_encoder__smooth': ['auto'],
        'estimator__criterion': ['gini', 'entropy', 'log_loss'],
        'estimator__max_depth': [None, 5, 10],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2],
        'estimator__max_features': [None, 'sqrt', 0.8, 1.0],
        'estimator__splitter': ['best']
    }

    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(x_df, y_df.values.ravel())
    print("Best estimator:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    return best_model

def tune_hyperparameters_sgd(x_df, y_df, pipe):

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
    grid_search.fit(x_df, y_df.values.ravel())
    print("Best estimator:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    return best_model
        

def avoiding_false_discoveries(best_model, train_cap_x_df, train_y_df, validation_cap_x_df,
                               validation_y_df, num_samples, data_type):
    # get bootstrap sample - fit and eval on bootstrap samples
    bs_results_df = get_bootstrap_sample_class(num_samples, best_model, train_cap_x_df, train_y_df,
                                               validation_cap_x_df, validation_y_df, data_type)

    # get randomized target sample - fit and eval on randomized target samples
    rt_results_df = get_randomized_target_sample_class(num_samples, best_model, train_cap_x_df, train_y_df,
                                                       validation_cap_x_df, validation_y_df, data_type)

    # combine the results
    results_df = pd.concat([bs_results_df, rt_results_df], axis=0)

    # plot the histogram
    plot_title = f'False Discovery Check using the {data_type} data set'
    plot_null_and_alt_dist(data=results_df.drop(columns='roc_auc_score_'), x='ave_precision_score',
                           hue='distribution', x_label=f'{data_type} ave_precision_score', y_label='Counts',
                           title=plot_title, kde=False)
    plot_null_and_alt_dist(data=results_df.drop(columns='ave_precision_score'), x='roc_auc_score_',
                           hue='distribution', x_label=f'{data_type} roc_auc_score_', y_label='Counts',
                           title=plot_title, kde=False)

def get_bootstrap_sample(cap_x_df, y_df, random_state):
    bs_df = pd.concat([cap_x_df, y_df], axis=1).sample(frac=1.0, replace=True, random_state=random_state)
    bs_cap_x_df, bs_y_df = bs_df.iloc[:, :-1], bs_df.iloc[:, -1]
    return bs_cap_x_df, bs_y_df

def get_bootstrap_sample_class(num_samples, best_model, train_cap_x_df, train_y_df, validation_cap_x_df,
                               validation_y_df, data_type):
    df_row_dict_list = []
    for i in range(num_samples):
        bs_cap_x_df, bs_y_df = get_bootstrap_sample(train_cap_x_df, train_y_df, i)
        
        # Split the bootstrap sample for training and validation
        bs_train_x, bs_val_x, bs_train_y, bs_val_y = train_test_split(bs_cap_x_df, bs_y_df, test_size=0.2, random_state=i)
        
        # Fit the model on the training set
        bs_train = best_model.fit(bs_train_x, bs_train_y)
        
        # Evaluate on the validation set
        roc_auc, ave_precision, y_pred = eval_class_sim(bs_val_x, bs_val_y, best_model, data_type)

        df_row_dict_list.append({
            'distribution': 'bootstrap_sample',
            'ave_precision_score': ave_precision,
            'roc_auc_score_': roc_auc
        })

    bs_results_df = pd.DataFrame(df_row_dict_list)

    return bs_results_df

def get_randomized_target_sample(cap_x_df, y_df, random_state):
    rt_df = pd.concat([cap_x_df, y_df], axis=1)
    target_name = rt_df.columns[-1]
    np.random.seed(random_state)
    rt_df[target_name] = np.random.permutation(rt_df[target_name])
    rt_cap_x_df, rt_y_df = rt_df.iloc[:, :-1], rt_df.iloc[:, -1]
    return rt_cap_x_df, rt_y_df

def get_randomized_target_sample_class(num_samples, best_model, train_cap_x_df, train_y_df, validation_cap_x_df,
                                       validation_y_df, data_type):
    df_row_dict_list = []
    for i in range(num_samples):
        rt_cap_x_df, rt_y_df = get_randomized_target_sample(train_cap_x_df, train_y_df, i)
        
        # Fit the model on the randomized target sample
        best_model.fit(rt_cap_x_df, rt_y_df)
        
        # Evaluate on the validation set
        roc_auc, ave_precision, y_pred = eval_class_sim(validation_cap_x_df, validation_y_df, best_model, data_type)

        df_row_dict_list.append({
            'distribution': 'randomized_target_sample',
            'ave_precision_score': ave_precision,
            'roc_auc_score_': roc_auc
        })

    rt_results_df = pd.DataFrame(df_row_dict_list)

    return rt_results_df

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

def eval_class_sim(cap_x_df, y_df, model, data_type):
    # Assuming you have a function to evaluate your model on the validation set
    # and return ROC AUC and average precision scores
    # Modify this part based on your actual evaluation function
    
    # Assuming model has predict_proba method for calculating probabilities
    y_pred_proba = model.predict_proba(cap_x_df)[:, 1]
    
    # Calculate ROC AUC and average precision
    roc_auc = roc_auc_score(y_df, y_pred_proba)
    ave_precision = average_precision_score(y_df, y_pred_proba)

    # Assuming you want to make binary predictions based on a threshold (e.g., 0.5)
    threshold = 0.5
    y_pred = (y_pred_proba >= threshold).astype(int)

    return roc_auc, ave_precision, y_pred

def print_classification_metrics_at_thresholds(model, cap_x_df, y_df, thresholds):

    # Evaluate the model at each threshold and print metrics
    for threshold in thresholds:
        y_pred_prob = model.predict_proba(cap_x_df)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)

        print("\nClassification Report at Threshold {:.2f}:\n".format(threshold))
        print(classification_report(y_df, y_pred))