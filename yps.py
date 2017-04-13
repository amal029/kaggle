#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import sklearn.metrics as metrics
import numpy as np


def main(file):
    df = pd.read_csv(file)
    # dfsr likes both scifi and into religion -- weird!
    dfsr = df[(df['Sci-fi'] > 3) & (df['Religion'] > 3)]
    print(dfsr.groupby('Gender')[['Sci-fi', 'Religion']].agg(['count']))

    # You like sci-fi, and are not religious -- the usual
    dfsnr = df[(df['Sci-fi'] > 3) & (df['Religion'] < 3)]
    print(dfsnr.groupby('Gender')[['Sci-fi', 'Religion']].agg(['count']))

    # You do not like sci-fi, and are religious -- the usual
    dfnsr = df[(df['Sci-fi'] < 3) & (df['Religion'] > 3)]
    print(dfnsr.groupby('Gender')[['Sci-fi', 'Religion']].agg(['count']))

    # Who is more religious, males or females in the UK?
    # First all males and females in the dataset.

    males = df[df['Gender'] == 'male']
    females = df[df['Gender'] == 'female']

    # Now get the males who are more interested in religion
    mr = males[df['Religion'] > 3]['Religion']  # religious males
    fr = females[df['Religion'] > 3]['Religion']  # religious females

    print('religious males:', (mr.count()/males['Religion'].count()*100),
          'religious females:', (fr.count()/females['Religion'].count()*100))

    # Hypothesis test -- people who are religious are *less* likely to
    # study biology | chemistry | Physics | Mathematics | Medicine.

    # Let's try biology.
    bmean = df['Biology'].mean()

    religious_males = males[df['Religion'] > 3]['Biology'].dropna()
    t, p = stats.ttest_1samp(religious_males, bmean)
    print("P value males:", p)

    religious_females = females[df['Religion'] > 3]['Biology'].dropna()
    t, p = stats.ttest_1samp(religious_females, bmean)
    print("P value females:", p)

    # Hypothesis test 2: male and females are equally likely to like STEM
    mstem = males[['Biology', 'Mathematics', 'Chemistry',
                   'Medicine', 'Physics']].dropna()
    fstem = females[['Biology', 'Mathematics', 'Chemistry',
                     'Medicine', 'Physics']].dropna()
    t, p = stats.ttest_ind(mstem, fstem)
    print("Males == Females:", p)
    print(mstem.mean(), '\n', fstem.mean())


def happiness(file):
    df = pd.read_csv(file)
    # Now make it into a classification problem
    df['Happiness'] = df.apply(lambda x: int(x['Happiness in life'] > 3),
                               axis=1)

    # Messaging data
    df = df.drop(['Happiness in life'], 1)

    # Remove all categorical for now
    to_rem = [x for x in df.columns if isinstance(df[x][0], str)]
    df = df.drop(to_rem, 1)     # Dropped all the categorical variables!

    df = df.dropna()

    # Now sample 10% of the data for testing
    testing = df.sample(frac=0.15, replace=True)
    i_drop = [x for x in testing.index]
    train = df.drop(i_drop, 0)

    # Reduce the dimensionality of the data
    n_components = 7
    pca = PCA(n_components=n_components)

    df_red = pca.fit_transform(X=train.drop('Happiness', 1))
    df_test = pca.transform(testing.drop('Happiness', 1))

    # Now we can do some logistic regression to predict happiness
    # > 3 means happy, else it means unhappy.

    # Now make the logit model
    df_red = sm.add_constant(df_red)
    logit_model = sm.Logit(endog=train['Happiness'], exog=df_red)
    # The result of fitting
    logit_result = logit_model.fit()
    # print(logit_result.summary())

    # Now some prediction
    df_test = sm.add_constant(df_test)
    y_pred = logit_result.predict(df_test)
    y_pred = np.apply_along_axis(np.round, 0, y_pred)
    y_true = testing['Happiness']
    # mat = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    # print(mat)
    # print('% Unhappy correctly predicted:', (mat[0][0]/sum(mat[0])*100))
    # print('% Happy correctly predicted:', (mat[1][1]/sum(mat[1])*100))

    # report
    print(metrics.classification_report(y_true, y_pred))

    # Random forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train.drop('Happiness', 1), train['Happiness'])
    y_pred = rf.predict(testing.drop('Happiness', 1))
    y_true = testing['Happiness']
    # report
    print(metrics.classification_report(y_true, y_pred))

    # Logit with transformed X using RF
    frf = SelectFromModel(rf, prefit=True)
    df_red = frf.transform(train.drop('Happiness', 1))
    df_test = frf.transform(testing.drop('Happiness', 1))

    # Now make the logit model
    df_red = sm.add_constant(df_red)
    logit_model = sm.Logit(endog=train['Happiness'], exog=df_red)
    # The result of fitting
    logit_result = logit_model.fit()
    # print(logit_result.summary())

    # Now some prediction
    df_test = sm.add_constant(df_test)
    y_pred = logit_result.predict(df_test)
    y_pred = np.apply_along_axis(np.round, 0, y_pred)
    y_true = testing['Happiness']
    # mat = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    # print(mat)
    # print('% Unhappy correctly predicted:', (mat[0][0]/sum(mat[0])*100))
    # print('% Happy correctly predicted:', (mat[1][1]/sum(mat[1])*100))

    # report
    print(metrics.classification_report(y_true, y_pred))


if __name__ == '__main__':
    plt.style.use('ggplot')
    main('yps_responses.csv')
    # happiness('yps_responses.csv')
