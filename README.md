# Predictive Modeling of Drought Levels Using Meteorological and Soil Data

## Project Report
**Authors:** Sai Saketh Vemula, Satvik Kalyan Gundu

## Abstract
This project aims to develop predictive models for classifying drought levels in US counties based on time-series weather data and soil characteristics. The dataset comprises six distinct drought levels, from "None" to five specific levels of drought severity. Each entry in the dataset represents the drought level at a specific point in time for a US county, accompanied by the last 90 days of 18 meteorological indicators such as wind speed, humidity, temperature range, and precipitation, among others. By utilizing a combination of machine learning techniques, the project seeks to evaluate the effectiveness of different models in predicting drought levels and provide insights into the contributing factors of drought conditions.

## Keywords
Drought Prediction, Time-Series Analysis, Meteorological Indicators, Soil Data, Machine Learning, Predictive Modeling, Data Mining

## Introduction
Drought is a natural disaster of immense global significance, occurring in various geographical regions and impacting societies, agriculture, and ecosystems [2]. Recent decades have witnessed severe drought events, including the 2010–2011 East Africa drought, 2011 Texas drought, 2012 U.S. Central Great Plains drought, and 2012–2015 California drought, resulting in substantial losses and far-reaching consequences for crop production and water supply. The extensive regional and global impacts of drought underline the urgent need for improved capabilities in drought prediction and mitigation.

However, drought remains one of the least understood natural hazards, primarily due to its multifaceted nature, with various causal mechanisms operating at different spatial and temporal scales. Drought can originate from precipitation deficits, but in some cases, it may be triggered by anomalies in other meteorological variables like temperature. Furthermore, human activities such as land use changes and reservoir operation can modify hydrological processes and influence drought development, blurring the line between natural and anthropogenic factors in drought occurrence.

Drought events are diverse in terms of timing, duration, and spatial extent, occurring at subseasonal/weekly, seasonal, multiyear, or decadal scales, and ranging from local to continental or even global scales. For instance, "flash droughts" with rapid onset may persist for just a few days or weeks, while certain regions may endure multiyear or decadal droughts of significant severity.

Predicting drought poses substantial challenges due to its intricate nature and the array of factors that influence it. Researchers and scientists have developed three main approaches for drought prediction: statistical, dynamical, and hybrid methods. Statistical methods rely on historical records and empirical relationships, while dynamical methods employ state-of-the-art climate models to forecast drought based on physical processes. Hybrid methods combine aspects of both approaches to improve prediction accuracy.

Drought prediction typically focuses on assessing drought severity, though it can also encompass other aspects like duration and frequency. In this project, we primarily center on seasonal drought prediction, aligning with current operational early warning systems to mitigate drought impacts.

The ability to predict drought levels accurately can contribute to better planning and mitigation strategies. In this project, we leverage a unique dataset containing time-series weather data and soil information to predict six different levels of drought severity across various US counties. The dataset includes the last 90 days of 18 key meteorological indicators such as wind speed, temperature, surface pressure, and precipitation for each entry. Given the temporal nature of the data and the importance of timely drought prediction, this project aims to explore various data mining techniques to build and evaluate predictive models for drought classification.


## Previous Work
 For the same data, previously a 180-day window of past data is used for predictions, which also included previous drought values, static data, and meteorological data from the year prior. It was evaluated on 6 future weeks of predictions. While the baseline model was still very simple, it performed much better using this additional input data

## Methods

To achieve a comprehensive understanding and accurate prediction, the following methods are employed:

1. **Data Preprocessing:** The primary datasets consisted of the train and test datasets, consisting a total of 21 attributes. These attributes have crucial parameters such as date, temperature variations, wind speed, precipitation levels, and a target column labeled as "score." Additionally, the supplementary soil dataset encompassed 32 attributes, providing intricate details such as elevation, water content, composition of barren and urban land, nutrient capacity, oxygen levels, and toxicity.

   Our approach involved constructing predictive models by combining both weather and soil datasets. Moreover, we also developed models exclusively using the weather data. This strategy facilitated a comprehensive exploration of two distinct perspectives: first, the potential improvements in model performance by merging soil-specific information, and second, the evaluation of model efficacy just based on weather data, enabling a comparative analysis of model performance between the integrated and singular datasets. This comprehensive methodology allowed us to measure the impact and significance of incorporating soil-related insights into our predictive models for drought assessment. The dataset will undergo cleaning, normalization, and transformation to ensure it is suitable for model training. There weren't any missing values in the datasets.

2. **Exploratory Data Analysis:**
   The analysis commenced by generating bar plots to visualize the density of continuous variables. Subsequently, the investigation identified skewed density distributions among certain attributes, prompting the examination for outliers within these distributions. Box plots were utilized to represent these continuous variables, revealing a prevalence of outliers across most attributes. Specifically, an assessment was conducted to quantify the number of outliers existing beyond the threshold of (mean-3*std_deviation, mean+3*std_deviation). Remarkably, several attributes exhibited thousands of data points surpassing this range, leading to the implementation of an outlier elimination process. By excluding data points falling outside the range of (mean-3*std_deviation, mean+3*std_deviation), the dataset was reduced from 2,756,796 to 2,474,336 data points.

   Then bar plots were constructed for categorical variables such as 'score', 'year', 'month', and 'day', to illustrate their distributions. The examination of year and month distributions revealed consistent data collection trends spanning from the years 2000 to 2016. However, an uneven distribution of score values across different drought categories became evident upon analysis.

   Utilizing correlation plots to visualize associations between variables helped us understand attribute relationships. Additionally, a multivariate analysis was performed, highlighting interesting patterns through scatter plots depicting the relationships among humidity, temperature, and the score. Despite a substantial correlation coefficient of 0.88 among the set of variables given below, notable variations were observed within their distributions, underscoring their nuanced relationships.

   ![Correlation plot depicting associations between humidity, temperatures, and the score.](eda1.png)

   ![Scatter plot showcasing relationships between dew, temperatures, and the score.](eda2.png)

3. **Modeling:** 
   Different predictive models were fine-tuned through hyperparameter optimization using `GridsearchCV`. Below are the details of the best parameters obtained for each model and a little information on what these parameters signify.

   #### Bagging Classifier
   - `'max_features'`: Represents the maximum number of features to draw from the total features when training each base estimator. A value of 1.0 implies using all features.
   - `'max_samples'`: Indicates the maximum number of samples to draw from the total available when training each base estimator. Here, 1.0 denotes using the entire dataset.
   - `'n_estimators'`: The number of base estimators in the ensemble. In both merged and non-merged data, 100 estimators were found optimal.

   #### Naïve Bayes
   - `'var_smoothing'`: A small value added to the variance of each feature in the Gaussian Naïve Bayes model to handle zero variances. A value of 1e-07 was optimal for both datasets.

   #### Decision Tree
   - `'criterion'`: The function used to measure the quality of a split. 'Gini' implies using the Gini impurity.
   - `'max_depth'`: The maximum depth of the tree. A depth of 5 restricts the growth of the tree to prevent overfitting.
   - `'min_samples_leaf'`: The minimum number of samples required to be at a leaf node. A value of 1 implies that each leaf can have the lowest number of samples.
   - `'min_samples_split'`: The minimum number of samples required to split an internal node, with 2 being the optimal value.

   #### Logistic Regression
   - `'C'`: The inverse of regularization strength. A lower value specifies stronger regularization. We found 1.0 and 10.0 optimal for non-merged and merged data, respectively.
   - `'solver'`: The algorithm used for optimization.


## Results
The effectiveness of the models was evaluated based on their accuracy in different scenarios: using both weather and soil data (merged) and using only weather data (non-merged). The following is a summary of the best performing models in each scenario along with their respective accuracies.

### Best Performing Models
For the merged dataset (weather and soil data), the **Bagging Classifier** model achieved the highest accuracy of **89.5%**

For the non-merged dataset (only weather data), the **Random Forest Classifier** model performed the best, reaching an accuracy of **81.3%**.

### Detailed Model Accuracies
#### Weather and Soil Data (Merged)

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 75%      |
| Decision Tree        | 86.5%    |
| Random Forest        | 81.2%    |
| Bagging Classifier   | 89.5%    |
| Adaboost Classifier  | 60.5%    |

#### Weather Data Only (Non-merged)

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 75%      |
| Decision Tree        | 78.3%    |
| Random Forest        | 81.3%    |
| Bagging Classifier   | 61.3%    |
| Naïve Bayes Classifier| 58.0%   |

## Discussion
Our project, centered around predicting drought classes, revealed several insights and challenges, both in terms of data understanding and computational limitations.

### Data Complexity and Understanding
One of the primary challenges was understanding certain attributes within our dataset. Many of these parameters, deeply rooted in geological contexts, lacked straightforward definitions or intuitive interpretations. This necessitated a need for specialized geological knowledge or additional research to fully understand their impact on drought prediction. It was not just a matter of statistical analysis; rather, it required a synthesis of data science with domain-specific insights to decipher the relevance and influence of these complex attributes.

### Computational Limitations
Another significant constraint was the limitation in computational resources. The extensive size of the dataset posed a challenge in terms of processing power and memory requirements. This limitation impacted our ability to perform more sophisticated modeling, intricate feature engineering, and extensive hyperparameter tuning. The sheer volume of data demanded a level of computational capability that was beyond our available resources, thereby restricting the depth and breadth of our analysis.

### Discrepancies in Attribute Analysis
In our exploratory data analysis, we observed interesting discrepancies, particularly in the relationship between attributes. The correlation matrix indicated strong interdependencies between certain attributes, suggesting a substantial degree of association. However, a contrasting picture emerged when we examined the variance graphs. These same attributes exhibited significant variability, presenting a challenge in understanding their actual contribution to the prediction of drought classes. This dichotomy between correlation and variability highlighted the complexity in interpreting the data and raised questions about the reliability of these attributes as predictors.

### Lessons from Data Mining
Despite these challenges, we were able to apply several data mining techniques effectively. We managed to execute comprehensive data cleaning, handle missing values and outliers, and conduct exploratory data analysis (EDA) to identify interesting patterns and correlations. The scaling of attributes and the application of Principal Component Analysis (PCA) for dimensionality reduction were particularly successful. Moreover, hyperparameter tuning using GridsearchCV and the development of predictive models showcased the practical application of our classroom learning to a real-world problem.

### Future Scope and Improvement
Looking forward, there is significant scope for enhancing our project. Utilizing advanced computational resources could substantially improve our model training capabilities and predictive accuracy. More powerful computing would enable us to delve into more complex models and fine-tune hyperparameters more effectively. 

Furthermore, a deeper understanding of the geological factors underlying our dataset could open new avenues for feature engineering. By uncovering and leveraging hidden patterns and relationships within the attributes, our models could achieve a higher level of predictive performance, thus contributing more effectively to drought prediction and management.

## References
1. Ma, Yueling; Montzka, Carsten; Bayat, Bagher; Kollet, Stefan (2021). _An Indirect Approach Based on Long Short-Term Memory Networks to Estimate Groundwater Table Depth Anomalies Across Europe With an Application for Drought Analysis_. [Link](https://doi.org/10.3389/frwa.2021.723548.s001)
2. Hao, Z., Singh, V. P., & Xia, Y. (2018). _Seasonal drought prediction: Advances, challenges, and future prospects_. [Link](https://doi.org/10.1002/2016RG000549)
3. Dimara Kusuma Hakim, Rahmat Gernowo, Anang Widhi Nirwansyah (2023). _Flood prediction with time series data mining: Systematic review_. [Link](https://doi.org/10.1016/j.nhres.2023.10.001)
4. Band Shahab S., Karami Hojat, Jeong Yong-Wook, Moslemzadeh Mohsen, Farzin Saeed, Chau Kwok-Wing, Bateni Sayed M., Mosavi Amir (2022). _Evaluation of Time Series Models in Simulating Different Monthly Scales of Drought Index for Improving Their Forecast Accuracy_. [Link](https://www.frontiersin.org/articles/10.3389/feart.2022.839527)
5. Brakkee, E., van Huijgevoort, M. H. J., and Bartholomeus, R. P. (2022). _Improved understanding of regional groundwater drought development through time series modelling: the 2018–2019 drought in the Netherlands_. [Link](https://doi.org/10.5194/hess-26-551-2022)
6. Jon D. Pelletier, Donald L. Turcotte (1997). _Long-range persistence in climatological and hydrological time series: analysis, modeling and application to drought hazard assessment_. [Link](https://doi.org/10.1016/S0022-1694(97)00102-9)
7. Morid, S., Smakhtin, V. and Bagherzadeh, K. (2007). _Drought forecasting using artificial neural networks and time series of drought indices_. [Link](https://doi.org/10.1002/joc.1498)
