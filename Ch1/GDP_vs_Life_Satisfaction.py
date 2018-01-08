import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


def prepare_country_stats(oecd_bli_2017, gdp_per_capita_2015):
    oecd_bli_2017 = oecd_bli_2017[oecd_bli_2017['INEQUALITY'] == 'TOT']  # Constrain to where column INEQUALITY == TOT
    oecd_bli_2017 = oecd_bli_2017.pivot(index='Country', columns='Indicator', values='Value')

    gdp_per_capita_2015.rename(columns={"2015": "GDP Per Capita"}, inplace=True)
    gdp_per_capita_2015.set_index("Country", inplace=True)

    full_country_stats = pd.merge(left=oecd_bli_2017, right=gdp_per_capita_2015, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP Per Capita", inplace=True)

    return full_country_stats


# Uses pandas to import GDP/Life satisfaction data from 2 CSVs
oecd_bli = pd.read_csv('oecd_bli_2017.csv', thousands=',')
gdp_per_capita = pd.read_csv('gdp_per_capita.csv', thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

# Reorders data and sets specific columns to variables x and y
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats['GDP Per Capita']]
y = np.c_[country_stats['Life satisfaction']]

# Plots the GDP per capita vs life satisfaction for us
country_stats.plot(kind='scatter', x='GDP Per Capita', y='Life satisfaction')
plt.show()

# Selects the specific model (uncomment the one we want to use)
model = sklearn.linear_model.LinearRegression()
# model = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 3)

# Trains the model with the sample data
model.fit(x, y)

# Predicts life satisfaction of Cyprus based on their GDP
X_new = [[22587]]
print(model.predict(X_new))
