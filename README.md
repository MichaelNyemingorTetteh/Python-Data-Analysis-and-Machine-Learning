**Feature Selection With Python**
A teaching notebook that demonstrates three ways to choose which predictors to keep in a linear regression model, using the mlxtend library and the Credit data set. It is written as a lecture and lab walkthrough, so each method is introduced with a short explanation before the code.
What the notebook covers
The notebook works through three feature selection methods, in order:
1.	Best Subset Selection using the Exhaustive Feature Selector (EFS). It fits a model for every possible combination of predictors and keeps the best-scoring one.
2.	Forward Stepwise Selection using the Sequential Feature Selector (SFS) with forward=True. It starts from no predictors and adds the most useful one at each step.
3.	Backward Stepwise Selection using SFS with forward=False. It starts from all predictors and removes the least useful one at each step.
Every method is scored by 5-fold cross-validation on the R-squared metric, so the comparison between them is fair.

**Data set**
File: Credit.csv (400 rows)
Target: Balance

Predictors: the remaining columns. The categorical columns Gender, Student, Married, and Ethnicity are converted to numeric dummy variables with pd.get_dummies(..., drop_first=True), which leaves 11 predictors for the search.

Credit.csv must sit in the working directory set at the top of the notebook.

**Requirements**
•	Python 3.9 or newer
•	pandas
•	numpy
•	scikit-learn
•	mlxtend
•	matplotlib (for the selection-path plots)

The first code cell runs %pip install mlxtend, so mlxtend will be installed on first run if it is missing. The other libraries are assumed to be present, as they are in a standard Anaconda install.
How to run
1.	Place Credit.csv in the folder you want to work from.
2.	Open the notebook and edit the first cell so the path points to that folder: 
3.	%cd "your/path/to/the/folder"
4.	Run the cells in order, from top to bottom. Order matters: the fitting cells must run before the cells that print or plot their results.
5.	
**Notebook structure**
Section	What it does
Setup	Sets the working directory and imports the libraries.
Load and prepare data	Reads Credit.csv, creates dummy variables, and splits into X and the target.
**Best Subset (EFS)**	Fits the exhaustive selector, then prints the best score, the chosen indices, and the chosen feature names.

**Forward Stepwise** Fits the forward selector, prints the step-by-step subsets and the final choice, and plots the selection path.

**Backward Stepwise**	Fits the backward selector, prints the final choice, shows the metric table, and plots the selection path.

**Key result**
On the Credit data all three methods agree. The best six-feature set is Income, Limit, Rating, Cards, Age, and Student, with a cross-validated R-squared of about 0.9515. Best Subset reaches this by searching every combination; forward and backward selection reach the same set far more cheaply.

**Notes and things to watch**
A harmless warning on import. When mlxtend loads, scikit-learn's bundled documentation parser may print a UserWarning about a "potentially wrong underline length." It comes from a formatting quirk in a third-party docstring, not from your code, and it does not affect any result. To silence just that message, add this before the mlxtend import: 

import warningswarnings.filterwarnings("ignore", message="potentially wrong underline length")

Feature names need a DataFrame. k_feature_names_ and best_feature_names_ only return column names if the selector was fitted on a pandas DataFrame. If you fit on a NumPy array, use the index attributes (k_feature_idx_, best_idx_) instead.

An unused object remains. The backward setup cell still creates an unfitted backward_stepwise object that nothing uses after the fixes above. It is left in place so no cell is removed, but you can delete that line for a tidier notebook, since everything now points at sfs_backward.


An unused import. accuracy_score is imported in the setup cell. It is a classification metric and is not used anywhere in this regression notebook, so it can be removed without effect.

