import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Fetch dataset
auto_mpg = fetch_ucirepo(id=9)

#Data (as pandas dataframes)
X = auto_mpg.data.features
y = auto_mpg.data.targets


st.title("Cryptographic Puzzle")
st.header("Challenge AI")

st.text("Op deze pagina zul je zien hoe 3 verschillende regressie algorithmen de een voorspelling gaan maken over hoeveel mpg een auto zal hebben.")

if st.button('Print X'):
    st.write(X[:10])

if st.button('Print Y'):
    st.write(y[:10])

#X moet getransformeerd worden omdat er Nan waarden inzitten. Deze Nan waarden moeten ingevuld worden, dit wordt nu gedaan door de mediaan te nemen.
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

#De eerste functie is die van Lineare regressie.
def linear(X,y):

    #Het model wordt gemaakt
    model = LinearRegression()

    #Data wordt aan het model toegevoed. Als er eerde niet de Nan waarden waren vervangen had dit een error gegeven.
    model.fit(X, y)

    #Alle X rows worden nu gegeven om op te testen.
    predictions = model.predict(X)

    #Data gebruikt van https://www.auto-data.net/en/toyota-celica-t18-1.6-sti-105hp-3133 om een keer met een echt voorbeeld te testen
    new_data = np.array([[97.7,4,105,2645.55,11,93,2]])
    predicted_values = model.predict(new_data)


    print("Based on X: ",predictions)
    print("Based on Toyota Celcica: ",predicted_values)


def knn(X,y):
    #Maak training en test datasets van de data.
    #In deze demo wordt enkel de train data gebruikt om een zo accuraat mogelijke prediction te krijgen en om de vergelijking te kunnen maken met lineare regressie.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Knn model met het aantal neighbors
    knn = KNeighborsRegressor(n_neighbors=5)

    #Data wordt aan het model toegevoed. Als er eerde niet de Nan waarden waren vervangen had dit een error gegeven.
    knn.fit(X_train, y_train)

    #Een prediction van y (mpg) wordt gemaakt op basis van de X training data. Hier zou normaalgezien de X_test data gebruikt worden.
    y_pred = knn.predict(X_train)

    #Bekijk wat de waarde van de mean squared error is. Hierdoor krijgen we toch een idee met wat voor foutmarge we moeten rekening houden.
    mse = mean_squared_error(y_train, y_pred)

    #Ook hier weer een extra test met echte data. Het is dezelfde als bij de lineare regressie dus kan er ook hierop vergeleken worden.
    new_data = np.array([[97.7,4,105,2645.55,11,93,2]])
    predicted_values = knn.predict(new_data)
    prediction = knn.predict(X_train)

    
    print("Based on X: ", prediction)
    print("Based on Toyota Celica: ", predicted_values)
    print("Mean squared error:", mse)

def poly(X,y):

    degree = 2
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()

    # Fit the model to the polynomial features
    model.fit(X_poly, y)

    #new_data = np.array([[97.7, 4, 105, 2645.55, 11, 93, 2]])
    #new_data_poly = poly_features.transform(new_data)

    y_pred = model.predict(X_poly)

    print(y_pred)