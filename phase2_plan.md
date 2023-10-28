## EDA Plan (10/22)
1. Create a copy of the data for exploration. (Liyang) - done
2. Describe attributes basic info. (Liyang) - done
3. Duplicate obs. (Liyang) - observed
4. Missing targets. (Liyang) - not observed
5. Missingness. (Liyang) - not observed
6. Identify target, numerical, categorical. (Liyang) - all attrs are categorical
7. EDA for categorical.
   - Distribution (violin plots). (Liyang) - done
   - Cardinality and value counts. (Qian) - done
8. Target encoding for categorical. (Liyang) - done
9. EDA for numerical. 
   - Distribution (Histogram, KDE plots, etc.). (Liyang) - done
   - Outliers detection (Box plots). (Liyang) - done
   - Correlations and associations between attributes. (Qian) - done
   - VIF (Liyang) - done
   - Usefulness for task (correlation with target, variability). (Qian) - done
10. Identify the promising transformations you may want to apply. (Liyang, Qian) - done
11. Document what you have learned. (Liyang, Qian)

## Data Preparation Plan (10/27)
1. Create a copy (Liyang) - done
2. Identify required transformations (Qian)
   - Drop Outliers - not need 
   - Missingness (fill or drop) - not need 
   - Discretize continuous features - not need
   - Decompose features - not need
   - Transformations of features - not need
   - Aggregate into new features - not need
3. Create a pipeline of transformer (Liyang)
   - Drop duplicate obs  - done
   - Target Encoding - done
   - Drop attributes (useless/high correlation) - done
   - Feature scaling - done
4. EDA on transformed data (Liyang) - done
5. Document what you have learned. (Liyang, Qian)