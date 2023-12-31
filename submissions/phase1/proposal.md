# Proposal

### I. Frame the Problem and Look at the Big Picture

##### 1. Define the objective in business terms.

- The goal is to increase the efficiency of targeted advertising by accurately predicting the likelihood of a user clicking on an advertisement, enabling improved return on investment (ROI) and lower cost on advertising. In order to reach this goal, we will try to find the best model for the `CTR (Click-Through Rate)` prediction. 

- CTR is a ratio showing how often people who see ads or free product listing end up clicking it. CTR is the number of clicks that an ad receives divided by the number of times the ad is shown: `clicks ÷ impressions = CTR`. For example, if one ad had 5 clicks and 100 impressions, then the CTR would be 5%.

##### 2. How will your solution be used?

In this project, we want to use 2020 DIGIX Advertisement CTR Prediction dataset to build a model that improves advertising CTR prediction. The model can be integrated into the advertising serving system and will be used to predict the likelihood of a click for each available ad. And the system can then choose to display ad with the highest predicted CTR to the user, thereby potentially maximizing the actual clicks and revenue.

##### 3. How should performance be measured?

Because our model predicts whether a customer will click for an advertisement or product or not, this is a classification model (target value - 0 or 1). For classification problems, we would consider using ones from the major performance metrics:

- **Accuracy**: (Number of correct predictions)/(Total number of predictions)

- **Confusion Matrix**: A table with two dimensions (Actual and Predicted) and four terms (True Positives, True Negatives, False Positives, and False Negatives)

- **Precision**: (True Positives)/(True Positives + False Positives)

- **Recall**: (True Positives)/(True Positives + False Negatives)

- **F1-Score**: The harmonic mean of both Precision and Recall

- **AUC(Area Under the Curve)-ROC**: The two-dimensional area under the entire ROC(curve plotted between True Positive Rate and False Positive Rate)

##### 4. Is the performance measure aligned with the business objective?

- **Accuracy**: Accuracy indicates how well the model predicts all the labels correctly.A high accuracy rate could be a sign for good models when the dataset is balanced.

- **Confusion Matrix**: Confusion matrix identifies classes being predicted correctly/incorrectly and types of errors being made.

- **Precision**: Precision is important for checking the correctness of the model. A precision score closer to 1 indicates that the model produces less false positive errors.

- **Recall**: Recall is important for reducing the number of false negatives. A recall score closer to 1 indicates that the model is minimizing the false negative errors.

- **F1-Score**: F1-Score optimizes precision and recall. When the value of F1 is close to 1, the model is performing well in terms of both precision and recall.

- **AUC(Area Under the Curve)-ROC**: The AUC-ROC curve is used to visualize the performance of classification models.The higher the AUC, the better the performance of the model.

##### 5. What would be the minimum performance needed to reach the business objective?

At the current step, the minimum performance needed to reach the business objective is unclear. It depends on what the company is seeking for and what specific goal the company sets as a priority. In general, a Click-Through Rate higher than the current one is the minimum performance requirement.

##### 6. List the assumptions you have made so far.

- **Data is representative**: The provided dataset is representative of the overall user base and ad base, and the sampling method of ads is random and no sampling bias. This will make sure the trained model can generalize well to unseen real-world data.

- **Data is relevant**: Features in the dataset have predictive power for CTR, and not too many irrelevant ones are included. Irrelevant features can introduce noise to the model and impact the prediction power.

- **Data is good-quality**: Not too many outliers or missing values. Poor quality data can make system difficult to detect underlying patterns, and make trained model perform unpredictably in production.

- **External factors is constant**: External factors that may generate impacts on CTR remain constant over time. The model's performance may degrade overtime if external factors are considered constant during training, but actually change over time. 

##### 7. Verify assumptions if possible.

- **Data is representative**: Can be verified by comparing dataset with broader business data. However, it is out of our reach in this project.

- **Data is relevant**: Can be verified by correlation measurement or feature selection methods, which will be done in EDA phase.

- **Data is good-quality**: Box plots can be used to identify outliers; missing values can be calculated as percentages for each feature, which will be done in EDA phase.

- **External factors is constant**: As the dataset contain only 7 consecutive days, it is hard to verify external factors' change without additional data. It can also be validated through monitoring the model's performance after deploying in production.

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

- The dataset is heavily imbalanced with over 95% zero values in `'label'` column, this will introduce a risk that the minority class might not be adequately represented in either the training set or the test set (or both). This can lead to models that are poorly generalized or validated. To counteract this, **stratified sampling** is applied.

- Users can reproduce the split process by rerun the `train_test_split.ipynb`, but we will recommend downloading the split train and test data sets using `download_data.ipynb` for better performance.


