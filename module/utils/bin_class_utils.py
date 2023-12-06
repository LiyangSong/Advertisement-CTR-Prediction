from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import module.utils.data_prepare_utils as data_prepare_utils
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

from sklearn.inspection import permutation_importance



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
    
def eval_class_tuned(x_df, y_df, best_model, type):

    y_pred = best_model.predict(x_df)
    y_proba = best_model.predict_proba(x_df)[:, 1]

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
    fpr, tpr, _ = roc_curve(y_df, best_model.predict_proba(x_df)[:,1])
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

def tune_hyperparameters_dt(x_df, y_df, pipe):

    param_grid = {
        'preprocessor__numerical__imputer__strategy': ['mean', 'median'],
        'preprocessor__categorical__target_encoder__smooth': ['auto'],
        'estimator__criterion': ['gini', 'entropy'],
        'estimator__max_depth': [None, 5],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2],
        'estimator__max_features': [None, 'sqrt']
    }

    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(x_df, y_df.values.ravel())
    print("Best estimator:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_df)

    accuracy = accuracy_score(y_df, y_pred)
    print("accuracy:", accuracy)

    ave_precision = average_precision_score(y_df, y_pred)
    print("ave_precision_score:", ave_precision)

    roc_auc = roc_auc_score(y_df, y_pred)
    print("roc_curve_auc:", roc_auc)

    tune_perf_dict = {
        'accuracy': accuracy,
        'roc_curve_auc': roc_auc,
        'ave_precision_score': ave_precision
    }

    return best_model

def tune_hyperparameters_sgd(x_df, y_df, pipe):

    param_grid = {
        'preprocessor__numerical__imputer__strategy': ['mean', 'median'],
        'preprocessor__categorical__target_encoder__smooth': ['auto'],
        'estimator__loss': ['log_loss'],
        'estimator__penalty': ['l2', 'l1'],
        'estimator__alpha': [0.0001, 0.001],
        'estimator__max_iter': [1000, 10000],
    }

    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(x_df, y_df.values.ravel())
    print("Best estimator:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_df)

    accuracy = accuracy_score(y_df, y_pred)
    print("accuracy:", accuracy)

    ave_precision = average_precision_score(y_df, y_pred)
    print("ave_precision_score:", ave_precision)

    roc_auc = roc_auc_score(y_df, y_pred)
    print("roc_curve_auc:", roc_auc)

    tune_perf_dict = {
        'accuracy': accuracy,
        'roc_curve_auc': roc_auc,
        'ave_precision_score': ave_precision
    }

    return best_model
    