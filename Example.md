
# Example

The following example demonstrates how to the use the `pymatch` package to match [Lending Club Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data). Follow the link to download the dataset from Kaggle (you'll have to create an account, it's fast and free!).

Here we match Lending Club users that fully paid off loans (control) to those that defaulted (test). The example is contrived, however it shows the process at work as we find similiarities between & match two very different samples. 

The Matching process can be broken down into the following steps:

* [Data Prep](#data-prep)

----

## Data Prep


```python
import warnings
warnings.filterwarnings('ignore')
from pymatch.matcher import Matcher
import pandas as pd
import numpy as np

%matplotlib inline
```

Load the dataset and select a subset of columns.



```python
data = pd.read_csv("/Users/bmiroglio/Downloads/lending-club-loan-data/loan.csv")\
        [[
            "loan_amnt",
            "funded_amnt",
            "funded_amnt_inv",
            "term",
            "int_rate",
            "installment",
            "grade",
            "sub_grade",
            "loan_status"
        ]]
```

Create test and control groups and reassign `loan_status` to a binary treatment indicator. This is our reponse in the logistic regression model(s) used to generate propensity scores.


```python
test = data[data.loan_status == "Default"]
control = data[data.loan_status == "Fully Paid"]
test['loan_status'] = 1
control['loan_status'] = 0
```

----

### `Matcher`

Initalize `Matcher` object. Note that upon intialization, `Matcher` prints the formula used to fit logistic regression model(s) and the number of records in the majority/minority class. By default, `Matcher` will use all covariates in the dataset unless a formula is specified by the user. Any covariates passed to the `exclude` parameter will be ingored from the model fitting process. The exclude parameter is particularly useful for unique identifiers like a `user_id`.

```python
m = Matcher(test, control, yvar="loan_status")
```

    Formula:
    loan_status ~ loan_amnt+funded_amnt+funded_amnt_inv+term+int_rate+installment+grade+sub_grade
    n majority: 207723
    n minority: 1219


There is a significant imbalance in our data--the majority group (fully-paid loans) having many more records than the minority group (defaulted loans). We account for this by setting `balance=True` when calling `Matcher.fit_scores()` below. This tells `Matcher` to sample from the majority group when fitting the logistic regression model(s) so that the groups are of equal size. When undersampling this way, it is highly recommended that `nmodels` is explictly assigned to a integer greater than 1. The value of this integer should depend on the severity of the imbalance; here we use `nmodels`=100.


```python
# for reproducibility
np.random.seed(20170925)

m.fit_scores(balance=True, nmodels=100)
```

    Fitting 100 Models on Balanced Samples...
    Average Accuracy: 70.21%


The average accuracy of our 100 models is 70.21%, suggesting that there's significant separability in our data and hence justifiying the need for the matching procedure. 


```python
m.predict_scores()
```

```python
m.plot_scores()
```


![png](Example_files/Example_15_0.png)


The plot above demonstrates the separability present in our data. Test profiles have a much higher **propensity**, or estimated probability of defaulting given the features we isolated in the data.

Before proceeding, we need to tune the threshold used for matching. Records are matched if their propsenity scores are within `threshold` of each other, i.e. for two scores, `s1` and `s2`, `abs(s1 - s2)` <= `threshold`. Records with _no_ matches are dropped, therefore we want to choose the smallest threshold such that most if not all of our data are retained. 

```python
m.tune_threshold(method='random')
```

![png](Example_files/Example_17_0.png)

It looks like a threshold of `0.0001` retains 100% of our data. We will now match using this threshold. 

```python
m.match(method="min", threshold=0.0001)
```


```python
cd = m.compare_discrete(return_table=True)
```


![png](Example_files/Example_19_0.png)



![png](Example_files/Example_19_1.png)



![png](Example_files/Example_19_2.png)



```python
cd
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
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
</div>




```python
cc = m.compare_continuous(return_table=True)
```


![png](Example_files/Example_21_0.png)



![png](Example_files/Example_21_1.png)



![png](Example_files/Example_21_2.png)



![png](Example_files/Example_21_3.png)



![png](Example_files/Example_21_4.png)



```python
cc
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var</th>
      <th>ks_before</th>
      <th>ks_after</th>
      <th>perm_chisqr_before</th>
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
      <td>0.565</td>
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
      <td>0.559</td>
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
      <td>0.547</td>
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
      <td>0.110</td>
      <td>0.000</td>
      <td>0.341</td>
      <td>0.673904</td>
      <td>0.091925</td>
      <td>0.670445</td>
      <td>0.079891</td>
    </tr>
    <tr>
      <th>4</th>
      <td>installment</td>
      <td>0.0</td>
      <td>0.416</td>
      <td>0.004</td>
      <td>1.000</td>
      <td>0.169177</td>
      <td>0.042140</td>
      <td>0.157699</td>
      <td>0.014590</td>
    </tr>
  </tbody>
</table>
</div>


