import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# Uses pandas to import GDP/Life satisfaction data from 2 CSVs
oecd_bli = pd.read_csv('oecd_bli_2015.csv', thousands=',')
gdp_per_capita = pd.read_csv('gdp_per_capita.csv', thousands=',', delimeter='\t', encoding='latin1', na_values='n/a')

#
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats['GDP Per Capita']]
y = np.c_[country_stats['Life Satisfaction']]

# Plots the GDP per capita vs life satisfaction for us
country_stats.plot(kind='scatter', x='GDP Per Capita', y='Life Satisfaction')
plt.show()

# Selects the specific model (uncomment the one we want to use)
model = sklearn.linear_model.LinearRegression()
#model = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 3)

# Trains the model with the sample data
model.fit(x, y)

# Predicts life satisfaction of Cyprus based on their GDP
X_new = [[22587]]
print(model.predict(X-new))
