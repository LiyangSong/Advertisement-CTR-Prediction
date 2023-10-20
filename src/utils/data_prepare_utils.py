from general_utils import save_to_csv
from sklearn.model_selection import train_test_split


def sample_data(df, fraction, random_state=42):
    print(f"Sample {fraction} fraction from DataFrame: \n")
    sample_df = df.groupby('label').apply(lambda x: x.sample(frac=fraction, random_state=random_state))
    print(f"sample_df.shape: {sample_df.shape}")
    return sample_df


def split_train_test_df(df, target, prefix='', stratify=None, test_size=0.2, random_state=42):
    print('Split DataFrame into train and test set:')

    cap_x_df, y_df = df.drop(target, axis=1), df[[target]]
    train_cap_x_df, test_cap_x_df, train_y_df, test_y_df = train_test_split(
        cap_x_df, y_df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify)

    print("train set:")
    print(train_cap_x_df.shape, train_y_df.shape)
    print("test set:")
    print(test_cap_x_df.shape, test_y_df.shape)

    save_to_csv(train_cap_x_df, train_y_df, prefix + 'train_df.csv')
    save_to_csv(test_cap_x_df, test_y_df, prefix + 'test_df.csv')

    del test_cap_x_df, test_y_df
    return train_cap_x_df, train_y_df
