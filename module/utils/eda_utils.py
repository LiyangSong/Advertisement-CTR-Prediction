from IPython.core.display_functions import display
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def check_out_general_info(df):
    print("\nGeneral information of DataFrame:")
    print(f"df.shape:\n{df.shape}")
    print("df.head():")
    display(df.head())
    print("df.info:")
    display(df.info())
    print("df.describe: ")
    display(df.describe())


def check_out_missing_target(df, target):
    print("\nCheck out observations with missing target:")
    drop_miss_tar_df = df.dropna(subset=target)
    print("df.shape: ", df.shape)
    print("drop_miss_tar_df.shape: ", drop_miss_tar_df.shape)
    if drop_miss_tar_df.shape[0] < df.shape[0]:
        print("Caution: data set contains observations with missing target!!!")
    else:
        print("No missing-target observations observed in data set.")


def check_out_duplicate_obs(df):
    print("\nCheck out duplicate observations:")
    drop_dup_df = df.drop_duplicates()
    print("df.shape: ", df.shape)
    print("drop_dup_df.shape: ", drop_dup_df.shape)
    if drop_dup_df.shape[0] < df.shape[0]:
        print("Caution: data set contains duplicate observations!!!")
    else:
        print("No duplicate observations observed in data set.")


def check_out_missingness(df, sample_size_threshold=250):
    print("\nCheck out missingness:")
    if df.isna().sum().sum() > 0:
        print("Caution: found missing values in data set!!!")
        print("Use missingno to understand pattern of missingness:")

        sample_size = df.shape[0] if df.shape[0] < sample_size_threshold else sample_size_threshold
        print(f"df.shape[0]: {df.shape[0]}")
        print(f"missingno sample_size: {sample_size}")

        msno.matrix(df.sample(sample_size, random_state=42))
        plt.title("Matrix visualizing nullity of DataFrame")
        plt.show()
        msno.heatmap(df.sample(sample_size, random_state=42))
        plt.title("Heatmap visualizing nullity of DataFrame")
        plt.show()
    else:
        print("No missing values in data set.")


def attr_unique_counts(df):
    result = []
    for attr in df.columns:
        result.append({
            "attribute": attr,
            "unique counts": len(df[attr].unique())
        })
    result_df = pd.DataFrame(result)
    display(result_df)


def split_numerical_categorical_attr(df, target):
    print("\nSplit numerical and categorical attributes:")

    cap_x_df = df.drop(target, axis=1)
    numerical_attr_list = cap_x_df.select_dtypes(include=['number']).columns.tolist()
    categorical_attr_list = cap_x_df.select_dtypes(exclude=['number']).columns.tolist()

    print(f"numerical_attr_list: \n{numerical_attr_list}")
    print(f"categorical_attr_list: \n{categorical_attr_list}")

    return numerical_attr_list, categorical_attr_list


def box_plot_for_numerical(df, numerical_attr_list, n_cols=5):
    n_attrs = len(numerical_attr_list)
    n_rows = (n_attrs // n_cols) + (n_attrs % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    for i, attr in enumerate(numerical_attr_list):
        row = i // n_cols
        col = i % n_cols
        df.boxplot(column=attr, ax=axes[row, col])
    plt.tight_layout()
    plt.show()


def tukeys_method(df, attr):
    q1 = df[attr].quantile(0.25)
    q3 = df[attr].quantile(0.75)
    iqr = q3 - q1
    inner_fence = 1.5 * iqr
    outer_fence = 3 * iqr

    # inner fence lower and upper end
    inner_fence_le = q1 - inner_fence
    inner_fence_ue = q3 + inner_fence

    # outer fence lower and upper end
    outer_fence_le = q1 - outer_fence
    outer_fence_ue = q3 + outer_fence

    outliers_prob = df[(df[attr] <= outer_fence_le) | (df[attr] >= outer_fence_ue)].index.tolist()
    outliers_poss = df[(df[attr] <= inner_fence_le) | (df[attr] >= inner_fence_ue)].index.tolist()

    return outliers_prob, outliers_poss


def tukeys_method_for_numerical(df, numerical_attr_list):
    print(f"\nImplement Tukey\'s fences to identify outliers based on the Inter Quartile Range (IQR) method:")
    results = []
    for attr in numerical_attr_list:
        outliers_prob, outliers_poss = tukeys_method(df, attr)
        results.append({
            'Attribute': attr,
            'Outliers Prob Count': len(outliers_prob),
            'Outliers Prob Fraction': len(outliers_prob) / len(df[attr]),
            'Outliers Poss Count': len(outliers_poss),
            'Outliers Poss Fraction': len(outliers_poss) / len(df[attr])
        })
    results_df = pd.DataFrame(results)
    display(results_df)


def hist_plot_for_numerical(df, numerical_attr_list, bins=10, n_cols=5):
    print("\nHistogram plots for numerical attributes:")
    n_attrs = len(numerical_attr_list)
    n_rows = (n_attrs // n_cols) + (n_attrs % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2 * n_rows))
    for i, attr in enumerate(numerical_attr_list):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        sns.histplot(df[attr], ax=ax, kde=True, bins=bins)
        ax.lines[0].set_color('crimson')
    plt.tight_layout()
    plt.show()


def corr_for_numerical(df, numerical_attr_list, threshold=0.5):
    print("\nHeatmap visualizing correlation between numerical attributes:")

    result = df[numerical_attr_list].corr().abs()
    sns.heatmap(result)
    plt.show()

    corr_df = (
        result.where(
            np.triu(np.ones(result.shape), k=1)
            .astype(bool)
        )
        .stack()
        .to_frame(name='correlation')
    )
    new_index = [i + ' with ' + j for i, j in corr_df.index]
    corr_df.index = new_index
    corr_df = corr_df.sort_values('correlation', ascending=False)
    corr_df = corr_df[corr_df['correlation'] > threshold]
    print(f"\nMatrix visualizing correlation (>{threshold}) between numerical attributes:")
    display(corr_df)


def corr_target_for_numerical(df, target, numerical_attr_list):
    print("\nCalculate the correlation between each attribute and the target:")
    corr_tar = pd.DataFrame(df[numerical_attr_list].corrwith(df[target]), columns=["correlation"])
    corr_tar = corr_tar.sort_values('correlation')
    display(corr_tar)


def variance_for_numerical(df, numerical_attr_list):
    print("\nCalculate the variance of each attribute:")
    var_attr = pd.DataFrame(np.var(df[numerical_attr_list]), columns=["variance"])
    var_attr = var_attr.sort_values('variance')
    display(var_attr)


def prep_data_for_vif_calc(df, numerical_attr_list):
    print("Prepare DataFrame for vif calculation:")
    if df.isna().sum().sum() > 0:
        print(f"Found observations with nans - pre obs. drop a_df.shape: {df.shape}")
        df = df.dropna(axis=0, how='any')
        print(f"Post obs. drop a_df.shape: {df.shape}")

    design_matrix = None
    bias_attr = None
    for attr in df[numerical_attr_list]:
        if df[attr].nunique() == 1 and df[attr].iloc[0] == 1:  # found the bias attribute
            design_matrix = df[numerical_attr_list]
            bias_attr = attr
            print("Found the bias term - no need to add one")
            break

    if design_matrix is None:
        design_matrix = sm.add_constant(df[numerical_attr_list])
        bias_attr = 'const'
        numerical_attr_list = [bias_attr] + numerical_attr_list
        print("Added a bias term to the data frame to construct the design matrix for assessment of vifs.")

    # if numerical attributes in the data frame are not scaled then scale them - don't scale the bias term
    numerical_attr_list.remove(bias_attr)
    if not (df[numerical_attr_list].mean() <= 1e-10).all():
        print("Scale the attributes - but not the bias term")
        design_matrix[numerical_attr_list] = StandardScaler().fit_transform(design_matrix[numerical_attr_list])

    return design_matrix, bias_attr


def print_vifs(df, numerical_attr_list):
    print("\nInvestigate multi co-linearity: calculate variance inflation factors (VIF):")
    design_matrix, bias_attr = prep_data_for_vif_calc(df, numerical_attr_list)

    vif_df = pd.DataFrame({
        "attribute": design_matrix.columns.tolist(),
        "vif": [variance_inflation_factor(design_matrix.values, i) for i in range(design_matrix.shape[1])]
    })
    vif_df = vif_df.sort_values('vif', ascending=False)
    display(vif_df)
