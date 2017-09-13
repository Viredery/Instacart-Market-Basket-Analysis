# Instacart Market Basket Analysis

The triditional solution to this recommendation competition.

## Overview

In this competition, the anonymized data on customer orders over time is used to predict which previously purchased products will be in a user’s next order. 

## Goals for this project

1. Code refactoring
2. Modularize the code for future reuse.

## Pipeline


1. Preprocess

2. Feature Engineering:

   five entity profiles and three interaction profiles
   
   * order profile, product profile, user profile, 
     aisle profile, department profile
   
   * user-product profile, user-aisle profile, user-department profile

3. Training and Testing Datasets Xonstruction

4. Model

   a lightgbm model to predict the probability of products being reordered by a user.

5. Postprocess

   maximize f1-score.
   
**TODO:**

1. Stacking

2. Model

   a lightgbm model to predict the probability of the order without purchased products.


## Paper

* F1-score Maximizing: 

  Ye Nan, Kian Ming Chai, Wee Sun Lee, et al. Optimizing F-measure: A Tale of Two Approaches[J]. 2012.

* Data Mining and Recommendation Systems:

  Liu G, Nguyen T T, Zhao G, et al. Repeat Buyer Prediction for E-Commerce[C]// ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016:155-164.

## reference

1. Kaggle user Plantsgo's solution:

   https://github.com/plantsgo/Rental-Listing-Inquiries

2. Kaggle user Faron's discussion:

   https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/37221

## Score

Private Score 0.4035470, Public Score 0.4032665, which is not my final score for this competition. 

As I said, the aim of this project is refactoring and modularizing :)