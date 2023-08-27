#!/usr/bin/env python
# coding: utf-8

# # CY CERGY PARIS UNIVERSITÉ - DU DATA ANALYST 2022-2023

# ## Cours 04 - Rédaction d'un rapport d'analyse
# ### Par : PARING, Madanassono Pouezi
# ### Professeur : Dr CISEL, Matthieu

# ### *Choix du jeu de données: 3.3 Jeu de données artificiel didactisé (fichier "users.db.csv")*
# *Vous avez enfin l’option de travailler sur un jeu de données didactisé, que nous avons conçu spécialement pour un cours visant à développer l’autonomie dans l’analyse de données. Ce jeu de données porte sur des profils Tinder artificiels, c’est-à-dire que nous avons créé le jeu de données de bout en bout, cachant des relations entre variables (nombre de photos postées, score et popularité, etc.). L’exercice consiste à retrouver et décrire le plus grand nombre possible de patterns dans les données, ainsi que les quelques incohérences que nous avons intentionnellement introduites. Si vous faites le choix de travailler en équipe, vous devrez prouver, via votre notebook, que vous n’avez pas simplement pris les figures/résultats d’un coéquipier (ces figures doivent être présentes dans votre notebook mais pas dans le sien).*
# 
# https://ucergyfr.sharepoint.com/:x:/s/DUDataAnalyst2023/ESnZ_S3R-y5KuHTXryBXc8QBWRpv8zgke22-uSg_XE-HIw

# #### Description des colonnes
# 1. userid : Identificateur de l’utilisateur
# 2. date.crea : Date de création du compte
# 3. score : Score associé au profile (reflétant le succès sur l’application)
# 4. n.matches : Nombre total de matchs depuis la création du compte
# 5. n.updates.photo : Nombre de mises à jour de photo
# 6. n.photos : Nombre de photos sur le profil
# 7. last.connex : Date de la dernière connexion
# 8. last.up.photo : Date de la dernière mise à jour de la photo de profil
# 9. last.pr.update : Date de la dernière mise à jour du texte du profil
# 10. gender : Genre, 0 pour homme, 1 pour femme, 2 pour autres
# 11. sent.ana : Analyse du sentiment du texte du profil
# 12. length.prof : Nombre de mots dans le profil
# 13. voyage : Mot-clé voyage trouvé dans le profil, 0 pour oui, 1 pour non
# 14. laugh : Mot-clé rire trouvé dans le profil, 0 pour oui, 1 pour non
# 15. photo.keke : Photo de profil prise en maillot de bain, dans un ascenseur, ou avec des lunettes de soleil, 0 pour oui, 1 pour non
# 16. photo.beach : Une des photos de profil est prise à la plage, 0 pour oui, 1 pour non

# # 1 Préparation des données

# ## 1.1 Chargement des bibliothèques et création de dossiers

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pingouin as pg
from statsmodels.graphics.mosaicplot import mosaic


# In[ ]:


# Create folders
import os
folders = ["dataset/", "img/", "export/"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)


# ## 1.2 Importation des données

# In[ ]:


# Get data from csv file
dateCols = ["date.crea", "last.connex", "last.up.photo"]
dfTinder = pd.read_csv("dataset/users.db.csv", encoding="utf-8", index_col=0, parse_dates=dateCols)

# print DataFrame info and some records
dfTinder.info()
dfTinder.head()


# ## 1.3 Nettoyage des données

# In[ ]:


# Display descriptive summary statistics
dfTinder.describe(datetime_is_numeric=True)


# In[ ]:


# Since "sent.ana" is solely profile text related, it should have the same value wherever "length.prof" = 0
dfTinder[dfTinder["length.prof"] == 0]["sent.ana"].duplicated().sum()


# In[ ]:


# Drop all values that are considered as invalid
dfTinder["last.connex"] = dfTinder["last.connex"].where(dfTinder["last.connex"] >= dfTinder["date.crea"])
dfTinder["n.updates.photo"].where(dfTinder["n.updates.photo"] >= 0, inplace=True)
dfTinder["length.prof"].where(dfTinder["length.prof"] == dfTinder["length.prof"].astype("int"), inplace=True)
dfTinder["sent.ana"].where(dfTinder["sent.ana"].isna(), inplace=True)

# Print missing values counts and plot their matrix
print("MISSING VALUES COUNT")
print(dfTinder.isna().sum())
print("\n\nMISSING DATA MATRIX")
axMissingData =msno.matrix(dfTinder)
plt.show()
axMissingData.figure.savefig("img/missing_data", dpi=300)


# In[ ]:


# Add columns for presence duration: days count from account creation to last connexion
dfTinder["n.days.to.last.connex"] = dfTinder["last.connex"] - dfTinder["date.crea"]
dfTinder["n.days.to.last.connex"] = dfTinder["n.days.to.last.connex"].transform(lambda x: x.days)


# In[ ]:


# Update coded values for more meaningful ones
dfTinder["gender"] = dfTinder["gender"].map({0: "Homme", 1: "Femme", 2: "Autres"})
for col in ["voyage", "laugh", "photo.keke", "photo.beach"]:
    dfTinder[col] = dfTinder[col].map({0: "Non", 1: "Oui"})


# In[ ]:


# Export cleaned data to CSV: table.exportcsv.csv
dfTinder.to_csv("export/table.exportcsv.csv", encoding="utf-8")


# # 2 Description du jeu de données

# In[ ]:


# Descriptive statistics, group by gender
print("GENDER VALUE COUNT PERCENTAGE")
print(dfTinder["gender"].value_counts(normalize=True))
print("\n\nSTATISTICS BY GENDER")
valuesCols = ["score", "n.matches", "n.photos", "n.days.to.last.connex"]
descStats = dfTinder.pivot_table(values=valuesCols, columns="gender", aggfunc=["count", "mean", "median", "std", "min", "max"])
descStats.to_excel("export/table.descstats.exportexcel.xlsx", sheet_name="Desc stats")
descStats


# In[ ]:


# Box plots of variables versus gender: "score", "n.matches", "n.photos", "n.days.to.last.connex"
axisLabelsFormat = dict(labelpad=10, fontdict={"fontsize": 12, "fontfamily": "Garamond", "fontweight": "bold"})
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
fig.tight_layout(pad=2, h_pad=4, w_pad=4, rect=None)

sns.boxplot(data=dfTinder, x="gender", y="n.matches", ax=ax1)
ax1.set_ylabel("Nombre de matchs", **axisLabelsFormat)

sns.boxplot(data=dfTinder, x="gender", y="score", ax=ax2)
ax2.set_ylabel("Score", **axisLabelsFormat)

sns.boxplot(data=dfTinder, x="gender", y="n.photos", ax=ax3)
ax3.set_ylabel("Nombre de photos", **axisLabelsFormat)

sns.boxplot(data=dfTinder, x="gender", y="n.days.to.last.connex", ax=ax4)
ax4.set_ylabel("Nombre de jours de présence", **axisLabelsFormat)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel("Genre", **axisLabelsFormat)

plt.show()
fig.savefig("img/boxplots", dpi=300)


# In[ ]:


# Plot by gender the mean matches count versus profile photos count, plot their distributions as well
g = sns.JointGrid(data=dfTinder, x="n.photos", y="n.matches", hue="gender")
g.plot_joint(sns.lineplot, estimator="mean", ci=None)
g.plot_marginals(sns.histplot, multiple="dodge", shrink=1.2)
g.ax_joint.set(ylim=[0, 50], xlim=[0, 13], yticks=range(0, 55, 5), xticks=range(1, 15, 2))
sns.move_legend(g.ax_joint, "lower center", bbox_to_anchor=(.5, 1.2), ncol=3, title=None, frameon=False)
axisLabelsFormat = dict(labelpad=10, fontdict={"fontsize": 12, "fontfamily": "Garamond", "fontweight": "bold"})
g.ax_joint.set_xlabel("Nombre de photos", **axisLabelsFormat)
g.ax_joint.set_ylabel("Nombre de matchs", **axisLabelsFormat)
plt.show()
g.savefig("img/jointplot_photos_matches", dpi=300)


# In[ ]:


# Overview of presence time frame by gender
ax = sns.histplot(data=dfTinder, x="n.days.to.last.connex", hue="gender", multiple="dodge")
ax.set(xlim=[0, 100], xticks=range(10, 100, 20))
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
axisLabelsFormat = dict(labelpad=10, fontdict={"fontsize": 12, "fontfamily": "Garamond", "fontweight": "bold"})
ax.set_xlabel("Nombre de jours de présence", **axisLabelsFormat)
ax.set_ylabel("Nombre de profils Tinder", **axisLabelsFormat)
plt.show()
ax.figure.savefig("img/histplot_days_users", dpi=300)


# # 3 Tests statistiques

# ## 3.1 Tests d'indépendance de variables : Chi2 et mosaic plot

# In[ ]:


def testChi2AndPlotMosaic(data, x, y, xlabel, ylabel, title):
    print("\n" + 100*"*" + "\n", "Test Chi2:", xlabel, "et", ylabel, "\n" + 100*"*")
    expected, observed, stats = pg.chi2_independence(data, x, y)
    print("\nstats", 100*"=", stats, sep="\n")
    print("\nobserved", 100*"=", observed.unstack(), sep="\n")
    fig, ax = plt.subplots()
    fig, rects = mosaic(observed.unstack(), statistic=True, gap=0.01, labelizer=(lambda x: None), ax=ax)
    [ax.spines[side].set(linewidth=2) for side in ["left", "right", "top", "bottom"]]
    # ax.set_title(title, pad=10, fontdict={"fontsize": 15, "fontfamily": "Garamond", "fontweight": "bold"})
    axisLabelsFormat = dict(labelpad=10, fontdict={"fontsize": 13, "fontfamily": "Garamond", "fontweight": "bold"})
    ax.set_ylabel(ylabel.center(35), loc="top", **axisLabelsFormat)
    ax.set_xlabel(xlabel, **axisLabelsFormat)
    plt.show()
    fig.savefig(f"img/mosaicplot-{y}-{x}.png", dpi=300)

kwargs = {"y": "gender", "xlabel": "Genre"}
testChi2AndPlotMosaic(dfTinder, title="A", x="voyage", ylabel="Voyage dans le texte", **kwargs)
testChi2AndPlotMosaic(dfTinder, title="B", x="laugh", ylabel="Rire dans le texte", **kwargs)
testChi2AndPlotMosaic(dfTinder, title="C", x="photo.keke", ylabel="Photo kéké", **kwargs)
testChi2AndPlotMosaic(dfTinder, title="D", x="photo.beach", ylabel="Photo de plage", **kwargs)
plt.show()


# ## 3.2 Évaluation de l'effet du genre et du nombre de photos sur le score (ANCOVA)

# In[ ]:


ancovaTest = pg.ancova(data=dfTinder, dv="score", covar="n.photos", between="gender")
ancovaTest.to_clipboard(excel=True)
ancovaTest


# In[ ]:


pg.homoscedasticity(data=dfTinder, dv="score", group="n.photos")


# In[ ]:


pg.homoscedasticity(data=dfTinder, dv="score", group="gender")


# ## 3.3 Évaluation de l'effet du genre et du nombre de photos sur le nombre de matchs (ANCOVA)

# In[ ]:


ancovaTest = pg.ancova(data=dfTinder, dv="n.matches", covar="n.photos", between="gender")
ancovaTest.to_clipboard(excel=True)
ancovaTest


# In[ ]:


pg.homoscedasticity(data=dfTinder, dv="n.matches", group="n.photos")


# In[ ]:


pg.homoscedasticity(data=dfTinder, dv="n.matches", group="gender")


# ## 3.3 Lien entre score et nombre de matchs

# In[ ]:


def testPearsonCorrScoreMatches(gender=None):
    df = dfTinder if gender is None else dfTinder[dfTinder["gender"] == gender]
    corrTest = pg.corr(x=df["score"], y=df["n.matches"], method="pearson")
    corrTest["r2"] = corrTest.r['pearson']**2
    corrTest["Gender"] = "All" if gender is None else gender
    return corrTest

corrTests = [testPearsonCorrScoreMatches(gender) for gender in dfTinder["gender"].unique()]
corrTests.append(testPearsonCorrScoreMatches())
dfCorrTests = pd.concat(corrTests)
dfCorrTests.to_excel("export/table.exportexcel.xlsx", sheet_name="TestPearson Score-Matches")
dfCorrTests


# In[ ]:


g = sns.lmplot(data=dfTinder, x="score", y="n.matches", hue="gender", scatter_kws={"s": 10, "facecolor":"white"}, line_kws={"linewidth":1.5})
sns.move_legend(g, "lower center", bbox_to_anchor=(.45, 1), ncol=3, title=None, frameon=False)
axisLabelsFormat = dict(labelpad=10, fontdict={"fontsize": 12, "fontfamily": "Garamond", "fontweight": "bold"})
g.set_ylabels("Nombre de matchs", **axisLabelsFormat)
g.set_xlabels("Score", **axisLabelsFormat)
plt.show()
g.savefig("img/regplot", dpi=300)

