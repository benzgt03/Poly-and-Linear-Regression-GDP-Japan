import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd  # import libary ต่างๆที่จะใช้

#Preparation
covid_19 = pd.read_excel(r'C:\Users\Admin\Desktop\PortGT03\Covid19.xlsx')
gdp = pd.read_csv("C:\\Users\\Admin\\Desktop\\PortGT03\\GDP .csv")
df_gdp =pd.DataFrame(gdp)
df_gdpjapan = df_gdp[df_gdp['Country Name'] == 'Japan'].drop(['Country Name','Country Code','Indicator Name','Indicator Code'],axis=1)
list_japan = df_gdpjapan.values.tolist()
list_yearjp = df_gdpjapan.columns.values.tolist()
list_japan = sum(list_japan,[])
print(df_gdp.head())
print(df_gdp.describe())
print(df_gdpjapan)
print('list japan =',list_japan)
print('list year =',list_yearjp)

list_yearjp = list(map(int, list_yearjp))

#Convert to array and Train
X = np.asarray(list_yearjp)
y = np.asarray(list_japan)
X= X.reshape(-1,1)
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=60)

#Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
print('lm score =', lm.score(X_train, y_train))
y_pred = lm.predict(X_test)
print("Intercept =", lm.intercept_)
print("Coefficient =", lm.coef_)
print('Coefficient of determination: %.2f (The best case is 1)' % r2_score(y_test, y_pred))
print('Root Mean squared error: %.2f' % (np.sqrt(mean_squared_error(y_test, y_pred))))
a = []
a = np.asarray([2021,2022])
a = a.reshape(-1,1)
print(a)
y_prediction = lm.predict(a)
print('ทำนายGDPปี 2021 , 2022 =',y_prediction)

#plot Linear
plt.scatter(X,y)
plt.ylabel('Gdp point')
plt.xlabel('Year 19/20')
plt.title('Slope from Linear Regression model')
prd = lm.predict(X_test)
plt.plot(X_test, prd, 'r')

#Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_test)
lm.fit(X_poly,y_test)
print("polyIntercept =", lm.intercept_)
print("polyCoefficient =", lm.coef_)
print('polylm score =', lm.score(X_poly, y_train))
y_pred = lm.predict(X_poly)
print('polyCoefficient of determination: %.2f (The best case is 1)' % r2_score(y_test, y_pred))
print('polyRoot Mean squared error: %.2f' % (np.sqrt(mean_squared_error(y_test, y_pred))))

#plot polynomial
fig2 = plt.figure()
prd1= lm.predict(X_poly)
axes3 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
axes3.plot(X_test, prd1, 'r')
axes3.set_title('Slope from PolyRegression Model')
axes3.scatter(X,y)







plt.show()





