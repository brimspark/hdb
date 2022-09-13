import pandas as pd
import numpy as np
import datetime as dt

from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

plt.style.use('dark_background')
plt.rcParams["figure.dpi"] = 80
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['Montserrat']


# =============================================================================
# HDB Analysis
# 2022 09
# =============================================================================

df = pd.read_csv('HDB_FullData.csv')

df['time'] = pd.to_datetime( df['time'] )

DATA_RANGE = (df['time'].min().date(), df['time'].max().date())

#%%


# =============================================================================
# Overview of projects
# =============================================================================


print('Data start:', DATA_RANGE[0])
print('Data end:', DATA_RANGE[1])
print('Number of unique Towns:', df['town'].nunique())
print('Number of unique Projects:', df['full_name'].nunique())


# Projects data
pdf = pd.DataFrame()
pdf['startTime'] = df.groupby('full_name')['time'].min()
pdf['avgPSF'] = df.groupby('full_name')['cost_psf'].mean()
pdf['numTrx'] = df.groupby('full_name')['cost_psf'].count()
pdf['startPSF'] = df.groupby('full_name')['cost_psf'].first()
pdf['endPSF'] = df.groupby('full_name')['cost_psf'].last()
pdf['pctChg'] = (pdf['endPSF']/pdf['startPSF']-1) * 100
pdf['endTime'] = df.groupby('full_name')['time'].last()
pdf['years'] = (pdf['endTime']-pdf['startTime']).dt.days / 365
pdf['linearGrowthRate'] = pdf['pctChg'] / pdf['years'] * np.where(pdf['years']>0,1,0)


print( pdf['linearGrowthRate'].describe(percentiles=[.05,.1,.9,.95]) )


#%%


# =============================================================================
# Select project
# =============================================================================

projectList = sorted(df[ df['lease_commence_date']>2000 ]['full_name'].unique())

PROJECT = 'CLEMENTI AVE 4 2014'
PROJECT = 'GHIM MOH LINK 2013'
PROJECT = 'HOLLAND DR 2012'
PROJECT = 'DAWSON RD 2016'
# PROJECT = 'DOVER CRES 2012'


vdf = df[ df['full_name']==PROJECT ].sort_values('time', ascending=False)

print(vdf.groupby(['block','size_sqft'])['cost_psf'].mean().unstack().round(2))



#%%








#%%


# =============================================================================
# Linear trend for projects
# =============================================================================

def daysAgoToDate(days_ago):
    return (dt.datetime.today()-dt.timedelta(days_ago)).date()

def plotAllByName(project_names):
    fig, ax = plt.subplots( figsize=(16,9), tight_layout=True)
    ax_min = 0
    for name in project_names:
        tdf = df[ df['full_name']==name ][['time','days_ago','cost_psf']].sort_values('days_ago')
        sns.regplot( x=tdf['days_ago'], y=tdf['cost_psf'], label=name, ci=None, scatter_kws={'s':12})
        if tdf['days_ago'].min() < ax_min: ax_min = tdf['days_ago'].min()
    ax_min -= 60
    plt.grid(linestyle=':', color='grey')
    plt.title('Transactions by project name')
    plt.ylabel('Cost PSF')
    plt.xlabel('')
    plt.xlim([ax_min,0])
    ticks = [str(daysAgoToDate(-x)) for x in plt.xticks()[0]]
    ax.set_xticklabels(ticks)
    plt.legend()
    plt.show()


plotAllByName([PROJECT])



#%%




# =============================================================================
# MLR on group
# =============================================================================



def generateAndPlotMLR(af, group_name):
    df = af[ af['full_name']==group_name].dropna()
    df = df.sort_values('time').reset_index(drop=True)
    if len(df)<30:
        print(f'{group_name} - Insufficient data | {len(df)}')
        return None

    feature_cols = ['lease_years', 'days_ago', 'storey_mid', 'size_sqft']
    X = df[feature_cols]
    Y = df[['cost_psf']]

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    df['pred'] = regr.predict(X)
    df['actual'] = Y

    # MLR using statsmodels
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print(model.summary())

    fig, ax = plt.subplots( figsize=(5,5), tight_layout=True)
    sns.regplot( x=df['pred'], y=df['actual'], scatter_kws={'s':12})
    plt.grid(linestyle=':', color='grey')
    plt.title(group_name)
    plt.show()

    i = 1
    fig, ax = plt.subplots(1,4, figsize=(14,4), tight_layout=True)
    for feature in feature_cols:
        plt.subplot(1,4,i)
        sns.regplot( x=df[feature], y=df['cost_psf'], scatter_kws={'s':12})
        plt.grid(linestyle=':', color='grey')
        plt.ylabel('')
        i += 1
    plt.show()

    fig, ax = plt.subplots( figsize=(14,6), tight_layout=True)
    _scaler = 200 * (100+df['size_sqft']-df['size_sqft'].min()) / (df['size_sqft'].max()-df['size_sqft'].min())
    plt.scatter( x=df['time'], y=df['cost_psf'], s=_scaler, alpha=0.6)
    plt.plot( df['time'], df['cost_psf'].rolling(30).mean(), c='r' )
    plt.grid(linestyle=':', color='grey')
    plt.title(group_name)
    plt.show()

    fig, ax = plt.subplots(figsize=(4,4), tight_layout=True)
    sns.histplot( df['size_sqft'], ax=ax, bins=30 )
    plt.grid(linestyle=':', color='grey')
    plt.show()

    return df



sdf = generateAndPlotMLR(df, PROJECT )


#%%

# =============================================================================
# Use MLR to get a fair price for given features
# =============================================================================

def MLR_predict(af, group_name, storey_mid, size_sqft, asking_price, print_stats=False):
    df = af[ af['full_name']==group_name].dropna()
    df = df.sort_values('time').reset_index(drop=True)

    feature_cols = ['lease_years', 'days_ago', 'storey_mid', 'size_sqft']
    X = df[feature_cols]
    Y = df[['cost_psf']]

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    if print_stats:
        print('Intercept: \n', regr.intercept_)
        print('Coefficients: \n', regr.coef_)

    df['pred'] = regr.predict(X)
    df['actual'] = Y

    lease_years = df['lease_years'].min() # current lease remaining
    res = regr.predict([[lease_years, 0, storey_mid, size_sqft]])
    inputs = {'lease_years': lease_years,
              'days_ago': 0,
              'storey_mid': storey_mid,
              'size_sqft': size_sqft}

    if print_stats:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        print(model.summary())

    cost_psf = res[0][0]
    print(f"{group_name} | Storey rank: {storey_mid} | Size (sqft): {size_sqft} ")
    print(f"Predicted Cost PSF: {cost_psf:.0f}")
    print(f"Predicted Total Cost: ${cost_psf*size_sqft:,.0f}")
    print(f"Asking price: ${asking_price:,.0f}")

    asking_psf = asking_price / size_sqft


    i = 1
    fig, ax = plt.subplots(1,4, figsize=(14,4), tight_layout=True)
    for feature in feature_cols:
        plt.subplot(1,4,i)
        sns.regplot( x=df[feature], y=df['cost_psf'], scatter_kws={'s':12})
        plt.scatter( inputs[feature], cost_psf, s=100, c='gold', marker='^', zorder=9 )
        plt.scatter( inputs[feature], asking_psf, s=100, c='r', marker='^', zorder=9 )
        plt.grid(linestyle=':', color='grey')
        plt.ylabel('')
        i += 1
    plt.show()


    return df


sdf = MLR_predict(df, PROJECT, storey_mid=42, size_sqft=1163, asking_price=1315000)
