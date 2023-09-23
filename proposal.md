# Proposal

### I. Frame the Problem and Look at the Big Picture

##### 1. Define the objective in business terms.

TODO by Qian

##### 2. How will your solution be used?

TODO by Qian

##### 3. How should performance be measured?

TODO by Qian

##### 4. Is the performance measure aligned with the business objective?

TODO by Qian

##### 5. What would be the minimum performance needed to reach the business objective?

TODO by Qian

##### 6. List the assumptions you have made so far.

- Data is representative: the provided dataset is representative of the overall user base and ad base, and the sampling method of ads is random and no sampling bias. This will make sure the trained model can generalize well to unseen real-world data.

- Data is relevant: features in the dataset have predictive power for CTR, and not too many irrelevant ones are included. Irrelevant features can introduce noise to the model and impact the prediction power.

- Data is good-quality: not too many outliers or missing values. Poor quality data can make system difficult to detect underlying patterns, and make trained model perform unpredictably in production.

- External factors is constant: external factors that may generate impact on CTR remain constant over time. The model's performance may degrade overtime if external factors are considered constant during training, but actually change over time. 

##### 7. Verify assumptions if possible.

- Data is representative: can be verified by comparing dataset with broader business data. However, it is out of our reach in this project.

- Data is relevant: can be verified by correlation measurement or feature selection methods, which will be done in EDA phase.

- Data is good-quality: box plots can be used to identify outliers; missing values can be calculated as percentages for each feature, which will be done in EDA phase.

- External factors is constant: As the dataset contain only 7 consecutive days, it is hard to verify external factors' change without additional data. It can also be validated through monitoring the model's performance after deploying in production.

### II. Get the Data

##### 1. Find and document where you got the data.

- Data Set: https://www.kaggle.com/datasets/louischen7/2020-digix-advertisement-ctr-prediction
- The dataset contains advertising behavior data collected from seven consecutive days. Detailed data field descriptions: `data_fields.json`

##### 2. Get the data.

- The raw data file is stored in CSV format and has been compressed and uploaded to out team's AWS S3 bucket with public access.

- Use `download_data.ipynb` to download and unzip the raw data file.

##### 3. Convert the data to a format you can easily manipulate (without changing the data itself).

- The raw data is stored in CSV format with columns spit by `|`. It can be easily read by `pandas.read_csv(file_path, sep="|")` and transferred to a pandas dataframe.

##### 4. Sample a test set, put it aside, and never look at it to avoid data leakage through the data scientist.

- The train and test set split has completed using `train_test_split.ipynb` with a test size fraction of 0.2.
- Users can reproduce the split process by rerun the `train_test_split.ipynb`, but we will recommend downloading the split train and test data sets using `download_data.ipynb` for better performance.


