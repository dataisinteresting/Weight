import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Creates an array of elapsed days and assigns it to days_elapsed
days_elapsed = np.array([1, 2, 3 ,4, 5, 6, 7, 8, 9, 10,11,12]).reshape(-1,1)
weight = np.array([163.2, 162.6, 163.4, 161.6, 161.8, 161.2, 160.0, 160.4, 160.0, 159.6,160.0,159.2])

# Instantiates a LinearRegression model and fits it to the above arrays
model = sklearn.linear_model.LinearRegression().fit(days_elapsed,weight)

R_squared = '%.2f'%model.score(days_elapsed, weight)
slope = '%.2f'%model.coef_
intercept = '%.2f'%model.intercept_

print(f'The R^2 value for this regression is: {R_squared}')
print(f'The slope of this regression is: {slope}')
print(f'The intercept of this regression is: {intercept}')

# Instantiates a regplot of weight over days_elapsed
weight_plot = sns.regplot(x = days_elapsed, y = weight, color = 'black')

# Sets label for x-axis of plot
weight_plot.set_xlabel('Days', fontsize = 14)

# Sets label for y-axis of plot
weight_plot.set_ylabel('Weight (lbs)', fontsize = 14)

# Places text annotations of R_squared, slope, and intercept on the regplot
plt.text(3,0, f'R^2 = {R_squared}', size='medium', color='black', weight='semibold', **{'fontname':'Arial'})
plt.text(3,0, f'Slope = {slope}', size='medium', color='black', weight='semibold', **{'fontname':'Arial'})
plt.text(3,0, f'Intercept = {intercept}', size='medium', color='black', weight='semibold', **{'fontname':'Arial'})

# Saves the plot to the current working directory
plt.savefig('weight.png', dpi = 300)

# Renders the plot
plt.show()

