#import the librairies
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("age_vs_poids_vs_taille_vs_sexe.csv", sep=",")

#variables predictives
x = df[["age","taille"]]

#variable cible
y = df["poids"]

#init the model
model = LinearRegression()

#train the model
model.fit(x,y)

#test the model
print(model.score(x,y))

#print the coefficients a,b,c of the model
print(model.coef_)
