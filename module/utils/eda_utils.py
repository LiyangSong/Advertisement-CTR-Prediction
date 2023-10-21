from IPython.core.display_functions import display


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


def split_numerical_nominal_attr(df, target):
    print("\nSplit nominal and numerical attr:")

    cap_x_df = df.drop(target, axis=1)
    numerical_attr_list = cap_x_df.select_dtypes(include=['number']).columns.tolist()
    nominal_attr_list = cap_x_df.select_dtypes(exclude=['number']).columns.tolist()

    print(f"numerical_attr_list: \n{numerical_attr_list}")
    print(f"nominal_attr_list: \n{nominal_attr_list}")

    return numerical_attr_list, nominal_attr_list
