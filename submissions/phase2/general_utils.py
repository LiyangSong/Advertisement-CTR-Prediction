import zipfile
import requests
import tqdm
import pandas as pd
from IPython.core.display_functions import display


def download_file(url, save_path):
    print(f"\nDownload file from {url}:")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as f:
        for chunk in tqdm.tqdm(response.iter_content(chunk_size=8192),
                               total=total_size // 8192,
                               unit='KB',
                               desc=save_path):
            f.write(chunk)
    print(f"Succeed to download and save file: {save_path}")


def unzip_file(source_path, target_folder):
    print(f"\nUnzip file from {source_path} to {target_folder}:")
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    print(f"Succeed to unzip file: {source_path}")


def read_csv(csv_path, sep=","):
    print(f"\nRead CSV file {csv_path} into DataFrame:")
    df = pd.read_csv(csv_path, sep=sep)
    print("df.head(): ")
    display(df.head())
    print(f"df.shape: {df.shape}")
    return df


def save_to_csv(*args):
    if len(args) != 2 | len(args) != 3:
        raise ValueError("Unsupported number of arguments")

    print("\nSave DataFrame into csv file:")

    if len(args) == 2:
        df = args[0]
        csv_filename = args[1]
        df.to_csv(csv_filename, index=False)
    if len(args) == 3:
        cap_x_df = args[0]
        y_df = args[1]
        csv_filename = args[2]
        pd.concat([cap_x_df, y_df], axis=1).to_csv(csv_filename, index=False)

    print(f"File saved: {csv_filename}")


