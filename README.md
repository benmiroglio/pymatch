
`pymatch`
=====

Matching techniques for observational studies. Inspired by and adapted from Jasjeet Singh Sekhon's [Matching](https://cran.r-project.org/web/packages/Matching/Matching.pdf) package in R. I wrote an adaptation in Python that is better suited for my work at Mozilla which incorporates:

* Integration with Jupyter Notebooks (we use Jupyter + Pyspark)
* Additional Non-Parametric Tests / Plotting Functionality to assess balance.
* A more modular, user-specified matching process

This package was used to support [this research project](https://dl.acm.org/citation.cfm?id=3178876.3186162).

# Installation

Install through pip!

```bash
$ pip install pymatch
```

# 

The best way to get familiar with the package is to work through an example. The example below leaves out much of the theory behind matching and focuses on the application within `pymatch`. If interested, Sekhon gives a nice overview in his [Introduction to the Matching package in R](http://sekhon.berkeley.edu/papers/MatchingJSS.pdf).

# Example

The following example demonstrates how to the use the `pymatch` package to match [Lending Club Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data). Follow the link to download the dataset from Kaggle (you'll have to create an account, it's fast and free!). You can follow along this document or download the corresponding [Example.ipynb](https://github.com/benmiroglio/pymatch/blob/master/Example.ipynb) notebook (just be sure to change the path when loading data!).

Here we match Lending Club users that fully paid off loans (control) to those that defaulted (test). The example is contrived, however a use case for this could be that we want to analyze user sentiment with the platform. Users that default on loans may have worse sentiment because they are predisposed to a bad situation--influencing their perception of the product. Before analyzing sentiment, we can match users that paid their loans in full to users that defaulted based on the characteristics we can observe. If matching is successful, we could then make a statement about the **causal effect** defaulting has on sentiment if we are confident our samples are sufficiently balanced and our model is free from omitted variable bias.

This example, however, only goes through the matching procedure, which can be broken down into the following steps:

* [Data Preparation](#data-prep)
* [Fit Propensity Score Models](#matcher)
* [Predict Propensity Scores](#predict-scores)
* [Tune Threshold](#tune-threshold)
* [Match Data](#match-data)
* [Assess Matches](#assess-matches)

----

### Data Prep


```python
import warnings
warnings.filterwarnings('ignore')
from pymatch.Matcher import Matcher
import pandas as pd
import numpy as np

%matplotlib inline
```

Load the dataset (`loan.csv`) and select a subset of columns.



```python
path = "/Users/bmiroglio/Downloads/lending-club-loan-data/loan.csv"
fields = \
[
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "loan_status"
]

data = pd.read_csv(path)[fields]
```

Create test and control groups and reassign `loan_status` to be a binary treatment indicator. This is our response in the logistic regression model(s) used to generate propensity scores.


```python
test = data[data.loan_status == "Default"]
control = data[data.loan_status == "Fully Paid"]
test['loan_status'] = 1
control['loan_status'] = 0
```

----

### `Matcher`

Initialize the `Matcher` object. 

**Note that:**

* Upon initialization, `Matcher` prints the formula used to fit logistic regression model(s) and the number of records in the majority/minority class. 
    * The regression model(s) are used to generate propensity scores. In this case, we are using the covariates on the right side of the equation to estimate the probability of defaulting on a loan (`loan_status`= 1). 
* `Matcher` will use all covariates in the dataset unless a formula is specified by the user. Note that this step is only fitting model(s), we assign propensity scores later. 
* Any covariates passed to the (optional) `exclude` parameter will be ignored from the model fitting process. This parameter is particularly useful for unique identifiers like a `user_id`. 


```python
m = Matcher(test, control, yvar="loan_status", exclude=[])
```

    Formula:
    loan_status ~ loan_amnt+funded_amnt+funded_amnt_inv+term+int_rate+installment+grade+sub_grade
    n majority: 207723
    n minority: 1219


There is a significant **Class Imbalance** in our data--the majority group (fully-paid loans) having many more records than the minority group (defaulted loans). We account for this by setting `balance=True` when calling `Matcher.fit_scores()` below. This tells `Matcher` to sample from the majority group when fitting the logistic regression model(s) so that the groups are of equal size. When undersampling this way, it is highly recommended that `nmodels` is explicitly assigned to a integer much larger than 1. This ensures that more of the majority group is contributing to the generation of propensity scores. The value of this integer should depend on the severity of the imbalance: here we use `nmodels`=100.


```python
# for reproducibility
np.random.seed(20170925)

m.fit_scores(balance=True, nmodels=100)
```

    Fitting 100 Models on Balanced Samples...
    Average Accuracy: 70.21%


The average accuracy of our 100 models is 70.21%, suggesting that there's separability within our data and justifying the need for the matching procedure. It's worth noting that we don't pay much attention to these logistic models since we are using them as a feature extraction tool (generation of propensity scores). The accuracy is a good way to detect separability at a glance, but we shouldn't spend time tuning and tinkering with these models. If our accuracy was close to 50%, that would suggest we cannot detect much separability in our groups given the features we observe and that matching is probably not necessary (or more features should be included if possible).

### Predict Scores


```python
m.predict_scores()
```

```python
m.plot_scores()
```


![png](Example_files/Example_15_0.png)


The plot above demonstrates the separability present in our data. Test profiles have a much higher **propensity**, or estimated probability of defaulting given the features we isolated in the data.

---

### Tune Threshold

The `Matcher.match()` method matches profiles that have propensity scores within some threshold. 

i.e. for two scores `s1` and `s2`, `|s1 - s2|` <= `threshold`

By default matches are found *from* the majority group *for* the minority group. For example, if our test group contains 1,000 records and our control group contains 20,000, `Matcher` will
    iterate through the test (minority) group and find suitable matches from the control (majority) group. If a record in the minority group has no suitable matches, it is dropped from the final matched dataset. We need to ensure our threshold is small enough such that we get close matches and retain most (or all) of our data in the minority group.
    
Below we tune the threshold using `method="random"`. This matches a random profile that is within the threshold
as there could be many. This is much faster than the alternative method "min", which finds the *closest* match for every minority record.


```python
m.tune_threshold(method='random')
```


![png](Example_files/Example_19_0.png)


It looks like a threshold of 0.0001 retains 100% of our data. Let's proceed with matching using this threshold.

---

### Match Data

Below we match one record from the majority group to each record in the minority group. This is done **with** replacement, meaning a single majority record can be matched to multiple minority records. `Matcher` assigns a unique `record_id` to each record in the test and control groups so this can be addressed after matching. If subsequent modeling is planned, one might consider weighting models using a weight vector of 1/`f` for each record, `f` being a record's frequency in the matched dataset. Thankfully `Matcher` can handle all of this for you :).


```python
m.match(method="min", nmatches=1, threshold=0.0001)
```


```python
m.record_frequency()
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>freq</th>
      <th>n_records</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2264</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>




It looks like the bulk of our matched-majority-group records occur only once, 68 occur twice, ... etc. We can preemptively generate a weight vector using `Matcher.assign_weight_vector()`


```python
m.assign_weight_vector()
```

Let's take a look at our matched data thus far. Note that in addition to the weight vector, `Matcher` has also assigned a `match_id` to each record indicating our (in this cased) *paired* matches since we use `nmatches=1`. We can verify that matched records have `scores` within 0.0001 of each other. 


```python
m.matched_data.sort_values("match_id").head(6)
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_id</th>
      <th>weight</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>loan_status</th>
      <th>scores</th>
      <th>match_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.0</td>
      <td>18000.0</td>
      <td>18000.0</td>
      <td>17975.000000</td>
      <td>60 months</td>
      <td>17.27</td>
      <td>449.97</td>
      <td>D</td>
      <td>D3</td>
      <td>1</td>
      <td>0.644783</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2192</th>
      <td>191970</td>
      <td>1.0</td>
      <td>2275.0</td>
      <td>2275.0</td>
      <td>2275.000000</td>
      <td>36 months</td>
      <td>16.55</td>
      <td>80.61</td>
      <td>D</td>
      <td>D2</td>
      <td>0</td>
      <td>0.644784</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>80665</td>
      <td>1.0</td>
      <td>18400.0</td>
      <td>18400.0</td>
      <td>18250.000000</td>
      <td>36 months</td>
      <td>16.29</td>
      <td>649.53</td>
      <td>C</td>
      <td>C4</td>
      <td>0</td>
      <td>0.173057</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.0</td>
      <td>21250.0</td>
      <td>21250.0</td>
      <td>21003.604048</td>
      <td>60 months</td>
      <td>14.27</td>
      <td>497.43</td>
      <td>C</td>
      <td>C2</td>
      <td>1</td>
      <td>0.173054</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.0</td>
      <td>5600.0</td>
      <td>5600.0</td>
      <td>5600.000000</td>
      <td>60 months</td>
      <td>15.99</td>
      <td>136.16</td>
      <td>D</td>
      <td>D2</td>
      <td>1</td>
      <td>0.777273</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1828</th>
      <td>153742</td>
      <td>1.0</td>
      <td>12000.0</td>
      <td>12000.0</td>
      <td>12000.000000</td>
      <td>60 months</td>
      <td>18.24</td>
      <td>306.30</td>
      <td>D</td>
      <td>D5</td>
      <td>0</td>
      <td>0.777270</td>
      <td>2</td>
    </tr>
  </tbody>
</table>




---

### Assess Matches

We must now determine if our data is "balanced" across our covariates. Can we detect any statistical differences between the covariates of our matched test and control groups? `Matcher` is configured to treat categorical and continuous variables separately in this assessment.

___categorical___

For categorical variables, we look at plots comparing the proportional differences between test and control before and after matching. 

For example, the first plot shows:

* `prop_test` - `prop_control` for all possible `term` values, `prop_test` and `prop_control` being the proportion of test and control records with a given term value, respectively. We want these (orange) bars to be small after matching.
* Results (pvalue) of a Chi-Square Test for Independence before and after matching. After matching we want this pvalue to be > 0.05, resulting in our failure to reject the null hypothesis that the frequency of the enumerated term values are independent of our test and control groups.


```python
categorical_results = m.compare_categorical(return_table=True)
```


![png](Example_files/Example_32_0.png)



![png](Example_files/Example_32_1.png)



![png](Example_files/Example_32_2.png)



```python
categorical_results
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>before</th>
      <th>after</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>term</td>
      <td>0.0</td>
      <td>0.433155</td>
    </tr>
    <tr>
      <th>1</th>
      <td>grade</td>
      <td>0.0</td>
      <td>0.532530</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub_grade</td>
      <td>0.0</td>
      <td>0.986986</td>
    </tr>
  </tbody>
</table>




Looking at the plots and test results, we did a pretty good job balancing our categorical features! The p-values from the Chi-Square tests are all > 0.05 and we can verify by observing the small proportional differences in the plots.

___Continuous___

For continuous variables we look at Empirical Cumulative Distribution Functions (ECDF) for our test and control groups  before and after matching.

For example, the first plot pair shows:

* ECDF for test vs ECDF for control **before** matching (left), ECDF for test vs ECDF for control **after** matching (right). We want the two lines to be very close to each other (or indistinguishable) after matching.
* Some tests + metrics are included in the chart titles.
    * Tests performed:
        * Kolmogorov-Smirnov Goodness of fit Test (KS-test)
            This test statistic is calculated on 1000
            permuted samples of the data, generating
            an empirical p-value.  See `pymatch.functions.ks_boot()`
            This is an adaptation of the [`ks.boot()`](https://www.rdocumentation.org/packages/Matching/versions/4.9-2/topics/ks.boot) method in 
            the R "Matching" package
        * Chi-Square Distance:
            Similarly this distance metric is calculated on 
            1000 permuted samples. 
            See `pymatch.functions.grouped_permutation_test()`

    * Other included Stats:
        * Standardized mean and median differences.
             How many standard deviations away are the mean/median
            between our groups before and after matching
            i.e. `abs(mean(control) - mean(test))` / `std(control.union(test))`


```python
cc = m.compare_continuous(return_table=True)
```


![png](Example_files/Example_35_0.png)



![png](Example_files/Example_35_1.png)



![png](Example_files/Example_35_2.png)



![png](Example_files/Example_35_3.png)



![png](Example_files/Example_35_4.png)



```python
cc
```






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>ks_before</th>
      <th>ks_after</th>
      <th>grouped_chisqr_before</th>
      <th>grouped_chisqr_after</th>
      <th>std_median_diff_before</th>
      <th>std_median_diff_after</th>
      <th>std_mean_diff_before</th>
      <th>std_mean_diff_after</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>0.0</td>
      <td>0.530</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.207814</td>
      <td>0.067942</td>
      <td>0.229215</td>
      <td>0.013929</td>
    </tr>
    <tr>
      <th>1</th>
      <td>funded_amnt</td>
      <td>0.0</td>
      <td>0.541</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.208364</td>
      <td>0.067942</td>
      <td>0.234735</td>
      <td>0.013929</td>
    </tr>
    <tr>
      <th>2</th>
      <td>funded_amnt_inv</td>
      <td>0.0</td>
      <td>0.573</td>
      <td>0.933</td>
      <td>1.000</td>
      <td>0.242035</td>
      <td>0.067961</td>
      <td>0.244418</td>
      <td>0.013981</td>
    </tr>
    <tr>
      <th>3</th>
      <td>int_rate</td>
      <td>0.0</td>
      <td>0.109</td>
      <td>0.000</td>
      <td>0.349</td>
      <td>0.673904</td>
      <td>0.091925</td>
      <td>0.670445</td>
      <td>0.079891</td>
    </tr>
    <tr>
      <th>4</th>
      <td>installment</td>
      <td>0.0</td>
      <td>0.428</td>
      <td>0.004</td>
      <td>1.000</td>
      <td>0.169177</td>
      <td>0.042140</td>
      <td>0.157699</td>
      <td>0.014590</td>
    </tr>
  </tbody>
</table>




We want the pvalues from both the KS-test and the grouped permutation of the Chi-Square distance after matching to be > 0.05, and they all are! We can verify by looking at how close the ECDFs are between test and control.

# Conclusion

We saw a very "clean" result from the above procedure, achieving balance among all the covariates. In my work at Mozilla, we see much hairier results using the same procedure, which will likely be your experience too. In the case that certain covariates are not well balanced, one might consider tinkering with the parameters of the matching process (`nmatches`>1) or adding more covariates to the formula specified when we initialized the `Matcher` object.
In any case, in subsequent modeling, you can always control for variables that you haven't deemed "balanced".
