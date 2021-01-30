# Car-Manufacturing-Test
(Portfolio Project #2)

---
<div>
	<img src="https://github.com/ovokpus/Car-Manufacturing-Test/blob/master/images/mercedes.jpg">
</div>


## Goal of this project: Reduce the time a Mercedes-Benz spends on the test bench.



### Defining the Problem

Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers.

Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.

To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.

Our objective in this project is to reduce the time that cars spend on the test bench using model prediction.



### Dataset Information

The Train Dataset provided contains 378 columns and 4209 rows which can help us determine the time spent by each Mercedes Benz Unit for testing. The Target Value as identified is labeled as 'y', number of seconds it takes for each unit to pass testing.

The Test Dataset contains 377 columns and 4209 rows.

## Exploratory Data Analysis

This involves looking at the Data Summaries and Visualizations in order to:

1. Examine the Data
2. Discover patterns and relationships between the features
3. Identify the types of data
4. Clean up the Data, if necessary


Below is an overview of the Train and Test Dataframes:


TRAIN DATAFRAME
![alt text](https://github.com/ovokpus/Car-Manufacturing-Test/blob/master/images/train_data_head.jpg)

TEST DATAFRAME
![alt text](https://github.com/ovokpus/Car-Manufacturing-Test/blob/master/images/test_data_head.jpg)

#### Visualizing target value

![alt text](https://github.com/ovokpus/Car-Manufacturing-Test/blob/master/images/target_value_distribution.jpg)
min: 72.11 
max: 265.32 
mean: 100.66931812782134 
std: 12.6778749695168
Count of values above 200: 1 - The outlier

![alt text](https://github.com/ovokpus/Car-Manufacturing-Test/blob/master/images/target_values_change.jpg)

![alt text](https://github.com/ovokpus/Car-Manufacturing-Test/blob/master/images/target_values_sorted.jpg)

Here, from the change in target value over the dataset AND the scatterplot of sorted values:

We see that the maximum target value (265.32) is an outlier. This may have an effect on the final model, so we eventually got rid of that single observation.


#### Visualizing features

We also created visuals to examine the cardinality of the Categorical Values. Improper coding of high Cardinality features could lead to poor model performance and memory issues when trying to fit the model.

Our solution for this will be to encode these with the Label Encoder.

Also, we shall check for null and unique values. In addition we will also check the dataset to see if any column has a variance of zero. A zero-variance column would indicate only zeros in that column and that will not be useful for our model.

## Data Cleaning

Here we identified and removed columns with zero std (all 0 values, which are of no use for the model).

We also identified and removed the outlier target value.


## Feature Engineering

At this stage we proceeded to encode the categorical columns using Label Encoder. This converts the high cardinality features into binary labels to enable us use them for prediction.

Then we proceeded further to scale the data using a Normalizer.

After Scaling we performed Principal Component Analysis as a Dimensionality Reduction technique. This reduced the number of features from 364 to 36 in both the train and test datasets



## Model Training and Evaluation

Here we fitted the Scaled and Reduced Dataset with an XGBoost Algorithm, aiming for a 99% explained variance. Model stopped running at RMSE of 3.18895 and r2-score of 0.9341

Here is a snapshot of the predicted values for y - time spent on the bench.

![alt text](https://github.com/ovokpus/Car-Manufacturing-Test/blob/master/images/model_snapshot.jpg)


## Conclusion/Next Steps

Model saved to file.
