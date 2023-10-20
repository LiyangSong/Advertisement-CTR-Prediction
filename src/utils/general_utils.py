import zipfile
import requests
import tqdm
import pandas as pd


def download_file(url, save_path):
    print(f"Download file from {url}: \n")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192),
                          total=total_size // 8192,
                          unit='KB',
                          desc=save_path):
            f.write(chunk)
    print(f"Succeed to download and save file: {save_path}")


def unzip_file(source_path, target_folder):
    print(f"Unzip file from {source_path} to {target_folder}: \n")
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    print(f"Succeed to unzip file: {source_path}")


def read_csv(csv_path, sep=","):
    print("Read CSV file {csv_path} into DataFrame: \n")
    df = pd.read_csv(csv_path, sep=sep)
    print(f"df.head: \n{df.head}")
    print(f"df.shape: {df.shape}")
    return df


def save_to_csv(cap_x_df, y_df, csv_filename):
    print("Save DataFrame into csv file: \n")
    pd.concat([cap_x_df, y_df], axis=1).to_csv(csv_filename, index=False)
    print(f"File saved: {csv_filename}")
