# -*- coding: utf-8 -*-
"""
Created on Mon May 15 06:48:30 2023

@author: Dell
"""
QUESTION 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv("C:\\Users\Dell\Downloads\\Q9_a.csv")
data
# Kurtosis
data.kurt()
# Skewness
data.skew()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv("C:\\Users\Dell\Downloads\\Q9_b.csv")
data
# Kurtosis
data.kurt()
# Skewness
data.skew()


QUESTION 18
 1 )The above Boxplot is not normally distributed the median is towards the higher value
 2 ) The data is a skewed towards left. The whisker range of minimum value is greater than maximum
 3 ) The Inter Quantile Range = Q3 Upper quartile – Q1 Lower Quartile = 18 – 10 =8

QUESTION 19

ANS  First there are no outliers. Second both the box plot shares the same median that is approximately in a range between 275 to 250 and they are normally distributed with zero to no skewness neither at the minimum or maximum whisker range.



QUESTION 20

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
cars=pd.read_csv('C:\\Users\Dell\Downloads\\Cars (2).csv')
cars
sns.boxplot(cars.MPG)
# P(MPG>38)
1-stats.norm.cdf(38,cars.MPG.mean(),cars.MPG.std())

# P(MPG<40)
stats.norm.cdf(40,cars.MPG.mean(),cars.MPG.std())

# P (20<MPG<50)
stats.norm.cdf(0.50,cars.MPG.mean(),cars.MPG.std())-stats.norm.cdf(0.20,cars.MPG.mean(),cars.MPG.std()) 



QUESTION 21

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
cars=pd.read_csv('C:\\Users\Dell\Downloads\\Cars (2).csv')
cars
sns.distplot(cars.MPG, label='Cars-MPG')
plt.xlabel('MPG')
plt.ylabel('Density')
plt.legend();
cars.MPG.mean()

cars.MPG.median()



import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
df=pd.read_csv('C:\\Users\Dell\Downloads\\wc-at.csv')
df.head()
df.mean()

df.median()

df.mode()
# waist is multimodal, AT is bimodal data

sns.distplot(df['Waist'])
plt.show()

sns.distplot(df['AT'])
plt.show()

sns.boxplot(df['AT'])
plt.show()

# mean> median, right whisker is larger than left whisker, data is positively skewed.

sns.boxplot(df['Waist'])
plt.show()

## mean> median, both the whisker are of same lenght, median is slightly shifted towards left. Data is fairly symetric

QUESTION 24

from scipy import stats
from scipy.stats import norm
# Assume Null Hypothesis is: Ho = Avg life of Bulb >= 260 days
# Alternate Hypothesis is: Ha = Avg life of Bulb < 260 days
# find t-scores at x=260; t=(s_mean-P_mean)/(s_SD/sqrt(n))
t=(260-270)/(90/18**0.5)
t

# Find P(X>=260) for null hypothesis
# p_value=1-stats.t.cdf(abs(t_scores),df=n-1)... Using cdf function
p_value=1-stats.t.cdf(abs(-0.4714),df=17)
p_value

#  OR p_value=stats.t.sf(abs(t_score),df=n-1)... Using sf function
p_value=stats.t.sf(abs(-0.4714),df=17)
p_value





