## EDA Plan (10/22)
1. Create a copy of the data for exploration. (Liyang)
2. Describe attributes basic info. (Liyang)
3. Duplicate obs. (Liyang) - observed
4. Missing targets. (Liyang) - not observed
5. Missingness. (Liyang) - not observed
6. Identify target, numerical, categorical. (Liyang) - all categorical
7. EDA for categorical.
   - Distribution (violin plots). (Liyang)
   - Cardinality and value counts. (Qian)
8. Target encoding for categorical. (Liyang)
9. EDA for numerical. 
   - Distribution (Histogram, KDE plots, etc.). (Liyang)
   - Outliers detection (Box plots). (Liyang)
   - Correlations and associations between attributes. (Qian)
   - VIF (Liyang)
   - Usefulness for task (correlation with target, variability). (Qian)
10. Identify the promising transformations you may want to apply. (Liyang, Qian)
11. Document what you have learned. (Liyang, Qian)

## Data Preparation Plan (10/27)
1. Create a copy (Liyang)
2. Identify required transformations (Qian)
   - Drop Outliers - not need 
   - Missingness (fill or drop) - not need 
   - Discretize continuous features - not need
   - Decompose features - not need
   - Transformations of features - not need
   - Aggregate into new features - not need
3. Create a pipeline of transformer (Liyang)
   - Drop duplicate obs 
   - Target Encoding
   - Drop attributes (useless/high correlation)
   - Feature scaling
4. Document what you have learned. (Liyang, Qian)