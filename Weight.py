import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Creates an array of elapsed days and assigns it to days_elapsed
days_elapsed = np.array([1, 2, 3 ,4, 5, 6, 7, 8, 9, 10,11,12,13, 14,15,16,17,18,
                         19, 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
                         40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,
                         63,64,65,66,67,68,69,70,71,72,73]).reshape(-1,1)
weight = np.array([163.2, 162.6, 163.4, 161.6, 161.8, 161.2, 160.0, 160.4, 160.0, 159.6,160.0,159.2,157.8, 157.4,156.8,157.4,
                   157.2, 157.8,155.8, 156.6,156.8,155.8,157.8,156.4,155.8,155.4,154.4,153.0,153.8,152.8,153.2,
                   153.0,152.8,151.8,151.6,152.8,152.2,152.2,154.2,150.2, 151.8,151.2,150.8,153.2,
                   151.0,150.0,150.4,151.2,150.0,149.0,150.2,152.8,150.2,150.2,147.4,149.0,150.0,147.4,
                   150.8,151.6,153.4,152.2,150.4,150.2,150.0,152.6,151.2,147.2,146.8,146.8,146.6,
                   148.4,150.2])
# Day 62 BP

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

