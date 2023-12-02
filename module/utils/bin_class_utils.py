from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import module.utils.data_prepare_utils as data_prepare_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
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

def get_feature_importance_df(model, features, estimator_name):
    final_estimator = model.steps[-1][1]

    if estimator_name in ['LogisticRegression', 'SGDClassifier']:
        importance = final_estimator.coef_[0]
        feature_importance_df = pd.DataFrame({'Feature': features, 'Coefficient': importance})
        feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
        feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)
    else:
        importance = final_estimator.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

def eval_class(x_df, y_df, composite_estimator, type):
    log_pipe = composite_estimator.fit(
        x_df,
        y_df.values.ravel()
    )

    y_pred = log_pipe.predict(x_df)

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
    print('The area under the curve (AUC): ', roc_auc)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate (Sensitivity)')
    plt.ylabel('True Positive Rate (Specificity)')
    plt.title('Curve')
    plt.show()

def feature_select_drop(importance_list, threshold, estimator_name):
    if estimator_name in ['LogisticRegression', 'SGDClassifier']:
        feature_names = importance_list['Feature']
        feature_importance = importance_list['Abs_Coefficient']
        selected_features = [feature for feature, importance in zip(feature_names, feature_importance) if importance < threshold]
    else:
        feature_names = importance_list['Feature']
        feature_importance = importance_list['Importance']
        selected_features = [feature for feature, importance in zip(feature_names, feature_importance) if importance < threshold]
    return selected_features