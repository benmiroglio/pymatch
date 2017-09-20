from __future__ import division
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
from itertools import chain
import statsmodels.api as sm
import patsy 
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def progress(i, n, prestr=''):
    sys.stdout.write('\r{}{}%'.format(prestr, round(i / n * 100, 2)))
    
def is_continuous(colname, dmatrix):
    '''
    Check if the colname was treated as continuous in the patsy.dmatrix
    Would look like colname[<factor_value>] otherwise
    '''
    return colname in dmatrix.columns

def ks_boot(tr, co, nboots=1000):
    nx = len(tr)
    ny = len(co)
    w = np.concatenate((tr, co))
    obs = len(w)
    cutp = nx
    ks_boot_pval = None
    bbcount = 0
    ss = []
    fs_ks, _ = stats.ks_2samp(tr, co)
    for bb in range(nboots):
        sw = np.random.choice(w, obs, replace=True)
        x1tmp = sw[:cutp]
        x2tmp = sw[cutp:]
        s_ks, _ = stats.ks_2samp(x1tmp, x2tmp)
        ss.append(s_ks)
        if s_ks >= fs_ks:
            bbcount += 1
    ks_boot_pval = bbcount / nboots
    return ks_boot_pval

class Matcher:
    '''
    Matcher Class -- Match data for an observational study.
    
    Args:
        test (pd.DataFrame): Data representing the test group
        control (pd.DataFrame): Data representing the control group
        formula (str): custom formula to use for logistic regression
            i.e. "Y ~ x1 + x2 + ..."
        yvar (str): Name of dependent variable (the treatment)
        exclude (list): List of variables to ignore in regression/matching.
            Useful for unique idenifiers
    '''
    def __init__(self, test, control, yvar, formula=None, exclude=[]):
        
        self.yvar = yvar
        self.exclude = exclude + [self.yvar] + ['scores', 'match_id']
        self.formula = formula
        self.models = []
        self.swdata = None
        self.model_accurracy = []
        # create unique indices for each row
        # and combine test and control
        t, c = [i.copy().reset_index(drop=True) for i in (test, control)]
        c.index += len(t)
        self.data = t.dropna(axis=1, how='all').append(c.dropna(axis=1, how='all')).dropna()
        # should be binary 0, 1
        self.data[yvar] = self.data[yvar].astype(int)
        self.xvars = [i for i in self.data.columns if i not in self.exclude and i != yvar]
        self.matched_data = []  
        # create design matrix of all variables not in <exclude>
        
        print '{} ~ {}'.format(yvar, '+'.join(self.xvars))
        self.y, self.X = patsy.dmatrices('{} ~ {}'.format(yvar, '+'.join(self.xvars)), data=self.data,
                                             return_type='dataframe')

        self.xvars = [i for i in self.data.columns if i not in exclude]
            
        self.test= self.data[self.data[yvar] == True]
        self.control = self.data[self.data[yvar] == False]
        self.testn = len(self.test)
        self.controln = len(self.control)
        
        # do we have a class imbalance problem?
        self.minority, self.majority = \
          [i[1] for i in sorted(zip([self.testn, self.controln], [1, 0]), 
                                key=lambda x: x[0])]
                        
        print 'n majority:', len(self.data[self.data[yvar] == self.majority])
        print 'n minority:', len(self.data[self.data[yvar] == self.minority])
        
        # explodes design matrix if included
        assert "client_id" not in self.xvars, \
               "client_id shouldn't be a covariate! Please set exclude=['client_id']"    

    def fit_scores(self, balance=True, nmodels=None):
        """
        Args:
            balance (bool): Should balanced datasets be used? 
                (n_control ~ n_test)
            nmodels (int): How many models should be fit?
                Score becomes the average of the <nmodels> models if nmodels > 1
        """
        if not self.formula:
            # use all columns in the model
            self.formula = '{} ~ {}'.format(self.yvar, '+'.join(self.xvars))
        if balance:
            if nmodels is None:
                # fit mutliple models based on imbalance severity (rounded up to nearest tenth)
                minor, major = [self.data[self.data[self.yvar] == i] for i in (self.minority, self.majority)]
                nmodels = int(np.ceil((len(major) / len(minor)) / 10) * 10)
            self.nmodels = nmodels
            for i in range(nmodels):
                progress(i+1, nmodels, 
                         prestr="Fitting {} Models on Balanced Samples...".format(nmodels))
                
                # sample from majority to create balance dataset
                
                df = self.balanced_sample()
                df = pd.concat([Matcher.drop_static_cols(df[df[self.yvar] == 1], yvar=self.yvar), 
                                Matcher.drop_static_cols(df[df[self.yvar] == 0], yvar=self.yvar)])
                y_samp, X_samp = patsy.dmatrices(self.formula, data=df, return_type='dataframe')
                X_samp.drop(self.yvar, axis=1, errors='ignore', inplace=True)
                
                glm = GLM(y_samp, X_samp, family=sm.families.Binomial())
                res = glm.fit()
                self.model_accurracy.append(self._scores_to_accuracy(res, X_samp, y_samp))
                self.models.append(res)
            print "\nAverage Accuracy:", "{}%".\
                  format(round(np.mean(self.model_accurracy) * 100, 2))
        else:
            # ignore any imbalance and fit one model
            self.nmodels = 1
            print '\nFitting 1 (Unbalanced) Model...'
            glm = GLM(self.y, self.X, family=sm.families.Binomial())
            res = glm.fit()
            self.model_accurracy.append(self._scores_to_accuracy(res, self.X, self.y))
            self.models.append(res)
            print "Accuracy", round(np.mean(self.model_accurracy[0]) * 100, 2)
            
    def predict_scores(self):
        """
        Predict Propensity scores for each observation
        """
        scores = np.zeros(len(self.X))
        for i in range(self.nmodels):
            progress(i+1, self.nmodels, "Caclculating Propensity Scores...")
            m = self.models[i]
            scores += m.predict(self.X[m.params.index])
        self.data['scores'] = scores/self.nmodels
        
    def match(self, threshold=0.001, nmatches=1, tie_strategy='random', max_rand=10):
        """
        Match data
        
        Args:
            threshold (float): threshold for "exact" matching
                i.e. |score_x - score_y| >= theshold
            nmatches (int): How control profiles should be matched
                (at most) to test
            tie_stratgey (str): Strategy for when multiple control profiles
                are suitable matches for a single test profile
                "random" - choose randomly
                "min" - choose the profile with the closest score
            max_rand
        """
        if 'scores' not in self.data.columns:
            print "Propensity Scores have not been calculated. Using defaults..."
            self.fit_scores()
            self.predict_scores()
        test_scores = self.data[self.data[self.yvar]==True][['scores']]
        ctrl_scores = self.data[self.data[self.yvar]==False][['scores']]
        result, match_ids = [], []
        for i in range(len(test_scores)):
            progress(i+1, len(test_scores), 'Matching Control to Test...')
            match_id = i
            score = test_scores.iloc[i]
            if tie_strategy == 'random':
                bool_match = abs(ctrl_scores - score) <= threshold
                matches = ctrl_scores.loc[bool_match[bool_match.scores].index]
            elif tie_strategy == 'min':
                matches = abs(ctrl_scores - score).sort_values('scores').head(1)
            else:
                raise AssertionError, "Invalid tie_strategy parameter, use ('random', 'min')"
            if len(matches) == 0:
                continue
            # randomly choose nmatches indices, if len(matches) > nmatches
            select = nmatches if tie_strategy != 'random' else np.random.choice(range(1, max_rand+1), 1)
            chosen = np.random.choice(matches.index, min(select, nmatches), replace=False)
            result.extend([test_scores.index[i]] + list(chosen))
            match_ids.extend([i] * (len(chosen)+1))
        self.matched_data = self.data.loc[result]
        self.matched_data['match_id'] = match_ids  
        
    def select_from_design(self, cols):
        d = pd.DataFrame()
        for c in cols:
            d = pd.concat([d, self.X.select(lambda x: x.startswith(c), axis=1)], axis=1)
        return d

    def balanced_sample(self, data=None):
        if not data:
            data=self.data
        minor, major = data[data[self.yvar] == self.minority], data[data[self.yvar] == self.majority]
        return major.sample(len(minor)).append(minor).dropna()

    def plot_scores(self):
        assert 'scores' in self.data.columns, "Propensity scores haven't been calculated, use Matcher.predict_scores()"
        sns.distplot(self.data[self.data[self.yvar]==False].scores, label='Control')
        sns.distplot(self.data[self.data[self.yvar]==True].scores, label='Test')
        plt.legend(loc='upper right')
        
        
    def ks_by_column(self):
        def split_and_test(data, column):
            ctest = data[data[self.yvar] == True][column]
            cctrl = data[data[self.yvar] == False][column]
            return ks_boot(ctest, cctrl, nboots=500)
        
        data = []
        #assert len(self.matched_data) > 0, 'Data has not been matched, use Matcher.match()'
        for column in self.data.columns:
                if column not in self.exclude and is_continuous(column, self.X):
                    pval_before = split_and_test(self.data, column)
                    pval_after = split_and_test(self.matched_data, column)
                    
                    data.append({'var': column, 
                                 'p_before': round(pval_before, 6), 
                                 'p_after': round(pval_after, 6)})
        return pd.DataFrame(data)[['var', 'p_before', 'p_after']]
    
        
    def prop_test_by_column(self):
        ret = []
        for col in self.matched_data.columns:
            if not is_continuous(col, self.X) and col not in self.exclude:
                ret.append(self.prop_test(col))
        return pd.DataFrame(ret)[["var", "before", "after"]]


            
    def prop_test(self, col):
        if not is_continuous(col, self.X) and col not in self.exclude:
            pval_before = round(stats.chi2_contingency(self.prep_prop_test(self.data, col))[1], 6)
            pval_after = round(stats.chi2_contingency(self.prep_prop_test(self.matched_data, col))[1], 6)
            return {'var':col, 'before':pval_before, 'after':pval_after}
        else:
            print "{} is a continuous variable".format(col)
                    
    def plot_ecdfs(self):
        for col in self.matched_data.columns:
            if is_continuous(col, self.X) and col not in self.exclude:
                # organize data
                trb, cob = self.test[col], self.control[col]
                tra = self.matched_data[self.matched_data[self.yvar]==True][col]
                coa = self.matched_data[self.matched_data[self.yvar]==False][col]
                xtb, xcb = ECDF(trb), ECDF(cob)
                xta, xca = ECDF(tra),ECDF(coa)
                
                # before/after stats
                std_diff_med_before, std_diff_mean_before = std_diff(trb, cob)
                std_diff_med_after, std_diff_mean_after = std_diff(tra, coa)
                pb, truthb = grouped_permutation_test(chi2_distance, trb, cob)
                pa, trutha = grouped_permutation_test(chi2_distance, tra, coa)
                ksb = round(ks_boot(trb, cob, nboots=1000), 6)
                ksa = round(ks_boot(tra, coa, nboots=1000), 6)
                
                # plotting
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))
                ax1.plot(xcb.x, xcb.y, label='Control', color=CONTROL_COLOR)
                ax1.plot(xtb.x, xtb.y, label='Test', color=TEST_COLOR)
                ax1.plot(xcb.x, xcb.y, label='Control')
                ax1.plot(xtb.x, xtb.y, label='Test')
                    
                title_str = '''
                ECDF for {} before Matching
                KS p-value: {}
                Grouped Perm p-value: {}
                Std. Median Difference: {}
                Std. Mean Difference: {}
                '''
                ax1.set_title(title_str\
                  .format(col, ksb, pb, std_diff_med_before, std_diff_mean_before))
                ax2.plot(xca.x, xca.y, label='Control')
                ax2.plot(xta.x, xta.y, label='Test')
                ax2.set_title(title_str\
                  .format(col, ksa, pa, std_diff_med_after, std_diff_mean_after))
                ax2.legend(bbox_to_anchor=(1.4, 1.03))

                plt.xlim((0, np.percentile(xta.x, 99)))
                
    def plot_bars(self):
        def prep_plot(data, var, colname):
            t, c = data[data[self.yvar]==True], data[data[self.yvar]==False]
            countt = t[[var]].groupby(var).count() / len(t)
            countc = c[[var]].groupby(var).count() / len(c)
            ret = (countt-countc).dropna()
            ret.columns = [colname]
            return ret
        
        for col in self.matched_data.columns:
            if not is_continuous(col, self.X) and col not in self.exclude:
                dbefore = prep_plot(self.data, col, colname='before')
                dafter = prep_plot(self.matched_data, col, colname='after')

                dbefore.join(dafter).plot.bar()
                plt.title('Proportional Difference (test - control)')
                plt.ylim((-.1, .1))
                
    def prep_prop_test(self, data, var):
        print var
        counts = data.groupby([var, self.yvar]).count().reset_index()
        table = []
        for t in (0, 1):
            os_counts = counts[counts.treatment==t]\
                                     .sort_values(var)
            cdict = {}
            for row in os_counts.iterrows():
                row = row[1]
                cdict[row[var]] = row.client_id
            table.append(cdict)
        # fill empty keys as 0
        all_keys = set(chain.from_iterable(table))
        for d in table:
            d.update((k, 0) for k in all_keys if k not in d)
        ctable = [[i[k] for k in sorted(all_keys)] for i in table]
        return ctable
            
        
    def _scores_to_accuracy(self, m, X, y):
        preds = [1.0 if i >= .5 else 0.0 for i in m.predict(X)]
        return (y == preds).sum() / len(y)
    
    @staticmethod
    def drop_static_cols(df, yvar, cols=None):
        if not cols:
            cols = list(df.columns)
        # will be static for both groups
        cols.pop(cols.index(yvar))
        for col in df[cols]:
            n_unique = len(np.unique(df[col]))
            if n_unique == 1:
                df.drop(col, axis=1, inplace=True)
        return df

    
    @staticmethod
    def ks_boot(tr, co, nboots=1000):
        nx = len(tr)
        ny = len(co)
        w = np.concatenate((tr, co))
        obs = len(w)
        cutp = nx
        ks_boot_pval = None
        bbcount = 0
        ss = []
        fs_ks, _ = stats.ks_2samp(tr, co)
        for bb in range(nboots):
            sw = np.random.choice(w, obs, replace=True)
            x1tmp = sw[:cutp]
            x2tmp = sw[cutp:]
            s_ks, _ = stats.ks_2samp(x1tmp, x2tmp)
            ss.append(s_ks)
            if s_ks >= fs_ks:
                bbcount += 1
        ks_boot_pval = bbcount / nboots
        return ks_boot_pval
    