import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


st.title('Quelles sont les émissions de CO2 d’une voiture ? :dash:')

st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
  st.write("### Introduction")
  st.write("Ce projet consiste à observer quels sont les facteurs qui expliquent les émissions de CO2 d’une voiture.\nL’idée sous-jacente est d’identifier les véhicules polluants afin que les décideurs publics puissent mettre en place des actions pour favoriser l’achat de véhicules peu polluants (bonus/malus, etc…).")
  st.write("Dans la partie de modélisation, plusieurs algorithmes sont entraînés dans un but de comparaison.\nLa finalité consiste à départager les algorithmes entre eux (choisir le meilleur algorithme à mettre en place dans l’environnement de production) en pesant la performance, le temps de calcul mais aussi l’interprétabilité.")
  st.write("*Nota* : Les données sont extraites du site data.gouv.fr sur les véhicules commercialisés en 2012, 2013 et 2014.")

if page ==pages[1]:
  st.write("### Exploration")

#Importation des datasets sur plusieurs années
df2012 = pd.read_csv('BASE CL MAJ JUIN 2012.csv', encoding='latin1', sep=';')
df2013 = pd.read_csv('gov2013.csv', encoding='latin1', sep=';')
df2014 = pd.read_csv('mars-2014-complete.csv', encoding='latin1', sep=';')

#Harmonisation des noms de colonnes (nom de colonne dataset 2013 /=/ nom de colonne dataset 2012 et 2014)
#Nom de colonne = base 2013 (plus claire pour la compréhension)
dico={"lib_mrq": "Marque", "lib_mod_doss": "Modèle dossier", "lib_mod":"Modèle UTAC", "dscom": "Désignation commerciale", "cnit":"CNIT", "tvv":"Type Variante Version (TVV)", "typ_cbr":"Carburant",
      "hybride":"Hybride", "puiss_admin_98":"Puissance administrative","puiss_max":"Puissance maximale (kW)", "typ_boite_nb_rapp":"Boîte de vitesse",
      "conso_urb":"Consommation urbaine (l/100km)", "conso_exurb":"Consommation extra-urbaine (l/100km)", "conso_mixte":"Consommation mixte (l/100km)", "co2":"CO2 (g/km)",
      "co_typ_1":"CO type I (g/km)", "hc":"HC (g/km)", "nox":"NOX (g/km)", "hcnox":"HC+NOX (g/km)", "ptcl":"Particules (g/km)",
      "masse_ordma_min":"masse vide euro min (kg)", "masse_ordma_max":"masse vide euro max (kg)","champ_v9":"Champ V9", "date_maj":"Date de mise à jour"}

dico2={"cod_cbr":"Carburant",}

df2012 = df2012.rename(dico, axis=1)
df2014 = df2014.rename(dico, axis=1)
df2014 = df2014.rename(dico2, axis=1)

#Conversion de la puissance max sur data 2012 en float (au lieu de int)
df2012['Puissance maximale (kW)'] = df2012['Puissance maximale (kW)'].astype('float')

#Conversion de la puissance max sur data 2014 en float (au lieu de object)
df2014['Puissance maximale (kW)'] = df2014['Puissance maximale (kW)'].str.replace(',', '.')
df2014['Puissance maximale (kW)'] = df2014['Puissance maximale (kW)'].astype('float')

#Conversion des Series object en float
var_columns = ['Consommation urbaine (l/100km)', 'Consommation extra-urbaine (l/100km)',
               'Consommation mixte (l/100km)', 'CO type I (g/km)', 'HC (g/km)', 'NOX (g/km)', 'HC+NOX (g/km)', 'Particules (g/km)']

#pour l'année 2012
for i in var_columns:
  df2012[i] = df2012[i].str.replace(',', '.') #Remplacement des virgules par des points
  df2012[i] = df2012[i].astype('float')       #Conversion de la colonne en décimales

#pour l'année 2014
for i in var_columns:
  df2014[i] = df2014[i].str.replace(',', '.') #Remplacement des virgules par des points
  df2014[i] = df2014[i].astype('float')       #Conversion de la colonne en décimales

#Les données des 3 dataset sont harmonisées :
#Même nom de colonne
#Même type de colonne

#concaténation des datasets pour n'en former qu'un
df = pd.concat([df2012, df2013, df2014])
df.info()

#on réinitialise l'index pour faire corresepondre l'index à 1 seule ligne (et pas à 3 lignes)
df = df.reset_index(drop=True)
database = df

st.write("###### Step 1 : Sélection des variables à inclure dans notre analyse / Suppression des variables non pertinentes.")

#Suppression des colonnes non pertinentes
useless_col = ['Marque', 'Modèle dossier', 'Modèle UTAC', 'Désignation commerciale', 'CNIT', 'Type Variante Version (TVV)', 'Boîte de vitesse', 'HC (g/km)',
               'HC+NOX (g/km)', 'Champ V9', 'Date de mise à jour', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29']
df = df.drop(useless_col, axis=1)

st.write("**Variables non retenues pour l'étude :**")
st.write(useless_col)
st.write("Ces variables concernent principalement des entrées uniques (identifiants uniques, modalités nombreuses).")

st.write("Nombre d'entrées avant suppression des doublons :", str(len(df)))

#Nombre d'entrée UNIQUES (après suppression des doublons)
#Nombre de véhicule uniques égal à 15k
#Les déséquilibres entre catégories s'ajustent.
df = df.drop_duplicates()

st.write("Nombre d'entrées après suppression des doublons :", str(len(df)))

tauxNA = {}

for i in df.columns:
  missingrate = np.round(((df[i].isnull().sum()/len(df[i]))*100), 2)
  tauxNA[i] = missingrate.astype('str') + "%"

st.write("**Gestion des NaN :**")
st.write(tauxNA)
st.write("Très peu de valeurs sont manquantes, excepté pour la variable Particules (15%). La corrélation entre ces variables n'est pas vérifiée. le coefficient de corrélation n'est que de ", np.round(df['Particules (g/km)'].corr(df['CO2 (g/km)']), 2).astype('str'), ".\nOn peut observer la relation non linéaire entre ces 2 variables au travers d'un graphique : ")

#On supprime donc cette variable de notre jeu de données
df = df.drop('Particules (g/km)', axis=1)

#Gestion des NaN
#Suppression des lignes dont les données sont manquantes
df = df.dropna(axis=0, how='any', subset=['Consommation urbaine (l/100km)', 'Consommation extra-urbaine (l/100km)', 'Consommation mixte (l/100km)', 'CO2 (g/km)'])

#Complétion des valeurs manquantes avec leur médiane
df['CO type I (g/km)'] = df['CO type I (g/km)'].fillna(df['CO type I (g/km)'].median())
df['NOX (g/km)'] = df['NOX (g/km)'].fillna(df['NOX (g/km)'].median())

st.write("**Variables retenues de notre jeu de données :**")
st.write(df.columns)
