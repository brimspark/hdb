import pandas as pd
import glob
import argparse

from sklearn import linear_model
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('dark_background')
plt.rcParams["figure.dpi"] = 80
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['Montserrat']


# =============================================================================
# HDB Analysis
# JP 2021
# =============================================================================


argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument('project', help="DAWSON RD 2016")
argParser.add_argument('storey', help="Storey of unit")
argParser.add_argument('size', help="Size of unit in square feet")
argParser.add_argument('price', help="Asking price of the unit")
args = argParser.parse_args()

PROJECT = args.project
storey = args.storey
size = args.size
price = args.price



def load_project(PROJECT):
    df = pd.concat([pd.read_csv(file) for file in glob.glob("./data/*.csv")])
    df['time'] = pd.to_datetime( df['time'] )

    pdf = df[ df['full_name']==PROJECT ].dropna().sort_values('time', ascending=False)
    pdf['size_sqft'] = pdf['size_sqft'].astype(int)
    return pdf

pdf = load_project(PROJECT)


# =============================================================================
# Plot overview of cost PSF by blocks
# =============================================================================

def plot_project_block_avg(pdf):
    vdf = pdf.groupby(['block','size_sqft'])['cost_psf'].mean().unstack().round(2)
    fig, ax = plt.subplots( figsize=(7,5), tight_layout=True)
    sns.heatmap(vdf, cmap='RdYlGn_r', square=True)
    plt.title('Avg Cost PSF by Block', pad=20)
    plt.xlabel('Size (sqft)')
    plt.ylabel('Block')
    plt.show()


plot_project_block_avg(pdf)

# =============================================================================
# Use MLR to get a fair price for given features
# =============================================================================

def MLR_predict(df, group_name, storey_mid, size_sqft, asking_price, print_stats=False):
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
    pred_total_cost = cost_psf*size_sqft
    asking_psf = asking_price / size_sqft

    print(f"{group_name} | Storey: {storey_mid} | Size: {size_sqft} sqft | Asking price: ${asking_price:,.0f}")
    print(f"Predicted Cost PSF: {cost_psf:.0f}")
    print(f"Predicted Total Cost: ${pred_total_cost:,.0f}")
    print(f"Premium: ${asking_price-pred_total_cost:,.0f} ({asking_price/pred_total_cost-1:.2%})")

    xlabel_dict = {'lease_years': 'Lease Remaining',
                   'days_ago': 'Days Ago',
                   'storey_mid': 'Storey',
                   'size_sqft': 'Size (sqft)'}
    i = 1
    fig, ax = plt.subplots(1,4, figsize=(14,4), tight_layout=True)
    for feature in feature_cols:
        ax = plt.subplot(1,4,i)
        sns.regplot( x=df[feature], y=df['cost_psf'], scatter_kws={'s':12})
        plt.scatter( inputs[feature], cost_psf, s=100, c='gold', marker='^', zorder=9 )
        plt.scatter( inputs[feature], asking_psf, s=100, c='r', marker='^', zorder=9 )
        plt.grid(linestyle=':', color='grey')
        plt.ylabel('Cost PSF')
        plt.xlabel(xlabel_dict[feature])
        if feature=='lease_years':
            ax.invert_xaxis()
        i += 1
    plt.show()

    return df


sdf = MLR_predict(pdf, PROJECT, storey, size, price)
