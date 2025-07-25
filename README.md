# LST Modeling Procedure

This repository presents codes used for training and evaluating three separate seasonal models for LST forecasting.
The models were trained independently, but they all followed the same processing pipeline to ensure methodological consistency.
The modeling process was based on three main scripts:

  • processing-checkpoints.ipynb,
  • model-training-season.ipynb,
  • driver.ipynb.

The first script, processing-checkpoint-season.ipynb, extracts the necessary data. It takes as input an LST image obtained by averaging LST from a selection of Landsat 8 images and a set of predictor layers including:
 
  • fraction layers obtained from the spectral unmixing procedure applied over Sentinel 2 Images,
  • a set of Urban Canopy Parameters UCPs.

All predictor layers are resampled to a common spatial resolution of 10 meters. This is done using a nearest neighbour approach, which ensures that categorical variables such as LCZ classes or land cover types are preserved correctly during resampling. After resampling, all layers are spatially aligned to match a reference layer, specifically, the fraction layer. This alignment guarantees that every pixel from each layer corresponds exactly to the same geographical location. Once the layers are prepared, a set of masks is generated to filter out invalid or unwanted areas. These masks are applied based on three main conditions:

• pixels must lie within the boundaries of the AOI;
• pixels must not contain NoData values, either in the predictors or in the LST layer;
• pixels must be located in areas with stable land use, meaning no land consumption changes occurred between 2015 and 2023.

These individual masks are merged into a single composite mask, which is then used to exclude unsuitable pixels from further processing.
After the preprocessing steps are completed, the sampling phase begins.

Values from these layers are randomly sampled at coordinate points and maintained throughout the layers across the study area.
Sampling is performed by first selecting a total number of samples N. Then, the total number of samples is divided by the number of fractions. Samples are then collected over each image.

This entire procedure is repeated automatically for every fraction map corresponding to the selected season. The repetition is managed by a script called **driver.ipynb**, which uses the Python library papermill. Papermill allows running a Jupyter notebook multiple times by inserting new parameters into it at each run. This made the notebook behave like a function, where each run processed a different input. In addition, papermill saved the executed version of each notebook, including all plots and print outputs, which was useful for checking the steps and for future analysis.

Sampling points are selected and distributed across the different LCZ classes. The number of points assigned to each class is proportional to the number of pixels that the class occupies within the study area. This ensures that classes with larger spatial coverage contributed more samples to the dataset, while smaller classes were still represented. The sampling is performed in two steps. First, points are extracted from the LCZ layer, but only within the valid areas defined by the previously created mask. Then, the same coordinates were used to extract corresponding values from all other predictor rasters and the LST target layer. The result of this process is a dataframe where each row represents a sample point and each column contains the value of a specific variable (e.g., predictors, LST, LCZ class, etc.).

The third script, **model-training-season.ipynb**, which consists of the modeling part of the pipeline, first collects the sample dataframes generated for each fraction map and merges them into a single, unified dataframe. This combined dataframe includes all predictor variables along with the corresponding LST values that are used to train and evaluate the model. Next, the dataset is split into training and testing subsets. 70% of the samples are randomly assigned to the training set, while the remaining 30% are used for testing. This split helps evaluate the performance of the model on unseen data and ensures that it does not overfit.

A set of hyperparameters for the RF model was defined in the form of a grid. This grid
contains different values for key parameters that influence the model’s performance:

• n_estimators: [10, 20, 50, 100, 200] – the number of trees in the forest;
• max_depth: [5, 10, 20, 30, 50] – the maximum depth of each tree;
• max_features: [1.0, ’sqrt’] – the number of features to consider at each split;
• min_samples_split: [2, 5] – the minimum number of samples needed to split a node;
• min_samples_leaf : [1, 2] – the minimum number of samples required at a leaf node.

These combinations were tested using GridSearchCV, which performs a cross-validation on the training data. The goal was to find the best combination of parameters that leads to the lowest prediction error. The metric used for scoring in this case was the negative mean squared error, which is a common choice for regression problems.
Once the best model is selected, the average Root Mean Squared Error (RMSE) across the validation folds is calculated and plotted. This helps understand the general performance of the model and compare different configurations. Once the best combination of hyperparameters was found through the grid search, these values are used to train the final RF model. The model is built using the ensemble method available in the Python scikit-learn library, which combines the results of many decision trees to make more accurate and stable predictions.
The trained model is used to make predictions on both the training and testing datasets. This allows for a clear comparison between how well the model fits the data it learned from and how well it generalizes to unseen data.
To evaluate the model’s performance, several error metrics were calculated:

• MAE (Mean Absolute Error): average of the absolute differences between predicted and actual LST values;
• RMSE: square root of the average of the squared differences, giving more weight to larger errors;
• R² Score: it indicates how much of the variance in LST is explained by the model, where a value close to 1 means a better fit;
• Mean Bias Error (MBE): it shows if the model tends to overestimate or underestimate the LST values on average.

In addition to the performance metrics, the model’s feature importance is also visualized. This shows which predictors influenced the LST prediction the most. A scatterplot is created to display predicted versus actual values for both training and test sets, which helps visually assess how close the predictions are to the real values. The final result of the model is a complete LST map of the study area, where each pixel contains a predicted temperature value based on the input predictors.

A preliminary test of LST simulation can also be performed through the **model-training-season.ipynb** code. This involvs arbitrarily modifying input covariates by applying percentage reductions to pixel values or masking them based on vector layers to mimic changes in urban texture resulting from planned or past urban interventions. The pre-trained LST model is then run using these "altered" covariates to simulate the potential impact of such interventions on land surface temperature.
In the provided example, we increased the tall vegetation fraction and tree canopy height in Parco Lambro, an urban park within the Metropolitan city of Milan, by proportionally reducing the fractions of grass and bare soil. The chosen model for this example was the Summer one.
