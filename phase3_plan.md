## Shortlist Promising Models (11.30)

#### Qian: SGD, decision tree, Liyang: random forest, ada boost, gradient boost

1. Train many quick and dirty models from different categories (SGD, decision tree, random forest, ada boost, gradient boost) using standard parameters. 
2. Measure and compare their performance. For each model, use N-fold cross validation and compute the mean and standard deviation of the performance measure on the N folds.
3. Analyze the most significant attributes for each algorithm.
4. Analyze the types of errors the models make: What data would a human have used to avoid these errors?
5. Perform a quick round of feature selection and engineering.
6. Perform one or two more quick iterations of the five previous steps.
7. Shortlist the top three to five most promising models, preferring models that make different types of errors.

## Fine-tune the System (12.7)

1. Fine-tune the hyperparameters using cross validation.
    a. Treat your data transformation choices as hyperparameters. (target encoding, scaling)
    b. Use random search over grid search. For long training runs you may want to use a Bayesian optimization approach.
2. Try model ensemble methods. Combining your best models will often produce better results then running them individually.
3. Once you are confident about your final model, measure its performance on the test set to estimate the generalization error.

## Present Your Solution (12.9)

1. Document what you have done.
2. Create a presentation: Highlight the big picture first.
3. Explain why your solution achieves the business objective.
4. Present interesting points you noticed along the way: Describe what worked and what did not work.
b. List the assumptions and your systems limitations.
5. Ensure your key findings are communicated through visualizations and easy to remember statements.
