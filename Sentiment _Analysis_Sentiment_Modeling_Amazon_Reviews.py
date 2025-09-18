"""
Probleemstelling
Kozmos, dat via Amazon verkoopt en zich richt op huistextiel en dagelijkse kleding, wil zijn verkopen verhogen door klantbeoordelingen te analyseren en productkenmerken te verbeteren op basis van klachten. In dit kader zullen de beoordelingen worden geanalyseerd op sentiment en gelabeld, en zal een classificatiemodel worden opgebouwd op basis van de gelabelde data.

Datasetbeschrijving
De dataset bevat beoordelingen van een specifieke productcategorie, inclusief:

Review: De beoordeling zelf

Title: Titel van de beoordeling, korte samenvatting

HelpFul: Aantal personen dat de beoordeling nuttig vond

Star: Het aantal sterren dat aan het product is gegeven

Taak 1: Tekst Voorbewerking

Stap 1: Lees het bestand amazon.xlsx in.

Stap 2: Voer de volgende voorbewerkingen uit op de kolom Review:
a. Converteer alle letters naar kleine letters.
b. Verwijder interpunctie.
c. Verwijder numerieke uitdrukkingen.
d. Verwijder stopwoorden (woorden zonder betekenis).
e. Verwijder woorden die minder dan 1000 keer voorkomen.
f. Voer lemmatization uit.

Taak 2: Tekstvisualisatie

Stap 1: Barplot
a. Bereken de frequenties van woorden in de Review kolom en sla deze op als tf.
b. Hernoem de kolommen van de tf dataframe naar: words, tf.
c. Filter woorden met een tf groter dan 500 en visualiseer met een barplot.

Stap 2: WordCloud
a. Sla alle woorden uit Review op als één string, genaamd text.
b. Stel een WordCloud-sjabloon in en sla deze op.
c. Genereer de WordCloud met de opgeslagen string.
d. Voltooi de visualisatie met figure, imshow, axis, show.

Taak 3: Sentimentanalyse

Stap 1: Maak een SentimentIntensityAnalyzer object met NLTK.

Stap 2: Analyseer polariteitsscores:
a. Bereken polarity_scores() voor de eerste 10 beoordelingen.
b. Filter de eerste 10 beoordelingen op basis van de compound score.
c. Wijs voor deze 10 beoordelingen "pos" toe als compound > 0, anders "neg".
d. Voeg een nieuwe kolom toe aan de dataframe met deze pos/neg labels voor alle beoordelingen.

Opmerking: Met SentimentIntensityAnalyzer zijn de beoordelingen gelabeld en is de afhankelijke variabele voor het machine learning model gecreëerd.

Taak 4: Voorbereiding voor Machine Learning

Stap 1: Bepaal onafhankelijke en afhankelijke variabelen en splits de data in train- en testsets.

Stap 2: Zet tekst om naar numerieke representatie:
a. Maak een TfidfVectorizer object.
b. Pas fit toe op de trainingsdata.
c. Transformeer train- en testdata en sla op.

Taak 5: Modellering (Logistische Regressie)

Stap 1: Train een logistisch regressiemodel op de trainingsdata.

Stap 2: Voer voorspellingen uit:
a. Gebruik predict om de testdata te voorspellen en sla op.
b. Bekijk de resultaten met classification_report.
c. Bereken de gemiddelde accuracy met cross-validation.

Stap 3: Voer een willekeurige beoordeling door het model:
a. Selecteer een willekeurige beoordeling met sample uit de kolom Review.
b. Vectoriseer deze beoordeling met CountVectorizer.
c. Pas fit en transform toe en sla op.
d. Voer de voorspelling uit met het model.
e. Print de beoordeling en het voorspelde resultaat.

Taak 6: Modellering (Random Forest)

Stap 1: Analyseer de voorspellingen met Random Forest:
a. Train een RandomForestClassifier model.
b. Bereken de gemiddelde accuracy met cross-validation.
c. Vergelijk de resultaten met het logistisch regressiemodel.
"""

from warnings import filterwarnings
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import nltk
nltk.download('vader_lexicon')
from sklearn.model_selection import (
    train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, validation_curve
)
from sklearn.metrics import classification_report
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from numpy.lib.function_base import vectorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#****************************
#   Taak 1: Tekst Voorbewerking
#****************************
# Stap 1: Lees het bestand amazon.xlsx in.
# Stap 2: Voer de volgende voorbewerkingen uit op de kolom Review
#*****************************
df_path = os.path.join(os.getcwd(), 'HW_02_NLP\\amazon.xlsx')
df = pd.read_excel(df_path)
df.head()
col = "Review"
#a. Converteer alle letters naar kleine letters.
df[col] = df[col].str.lower()

#b. Verwijder interpunctie.
df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)

#c. Verwijder numerieke uitdrukkingen.
df[col] = df[col].str.replace('\d', '', regex=True)


#d. Verwijder stopwoorden (woorden zonder betekenis).
remove_stopwords  = stopwords.words('english')

df[col]  = df[col].apply(lambda x: " ".join(x for x in str(x).split() if x not in remove_stopwords ))


#e. Verwijder woorden die minder dan 1000 keer voorkomen.
temp_df = pd.Series(" ".join(df[col]).split()).value_counts()
drops = set(temp_df[temp_df >= 1000].index)
df[col] = df[col].apply(lambda x: " ".join(w for w in x.split() if w in drops))

#f. Voer lemmatization uit.
df[col] = df[col].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#****************************
#Taak 2: Tekstvisualisatie
#****************************

#Stap 1: Barplot
#****************************
#a. Bereken de frequenties van woorden in de Review kolom en sla deze op als tf.
#b. Hernoem de kolommen van de tf dataframe naar: words, tf.
#c. Filter woorden met een tf groter dan 500 en visualiseer met een barplot.
#****************************
tf = df[col].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf = (pd.Series(" ".join(df[col]).split()).loc[lambda s: s != ""].value_counts().reset_index())
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

#Stap 2: WordCloud
#****************************
#a. Sla alle woorden uit Review op als één string, genaamd text.
#b. Stel een WordCloud-sjabloon in en sla deze op.
#c. Genereer de WordCloud met de opgeslagen string.
#d. Voltooi de visualisatie met figure, imshow, axis, show.
#****************************

text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#****************************
#Taak 3: Sentimentanalyse
#****************************
#Stap 1: Maak een SentimentIntensityAnalyzer object met NLTK.#
#****************************
# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

#Stap 2: Analyseer polariteitsscores:
#****************************
#a. Bereken polarity_scores() voor de eerste 10 beoordelingen.
#b. Filter de eerste 10 beoordelingen op basis van de compound score.
#c. Wijs voor deze 10 beoordelingen "pos" toe als compound > 0, anders "neg".
#d. Voeg een nieuwe kolom toe aan de dataframe met deze pos/neg labels voor alle beoordelingen.
#****************************
sia = SentimentIntensityAnalyzer()
df[col] = df[col].astype(str)  # NaN veya float varsa stringe çevir
df = df[df[col].str.strip() != ""]  # boş stringleri at

df["compound"] = df[col].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Sentiment"] = df["compound"].apply(lambda x: "pos" if x > 0 else"neg")

df.head()


df["Sentiment"].value_counts()

df["Sentiment"] = LabelEncoder().fit_transform(df["Sentiment"])
df.head()

#****************************
#Taak 4: Voorbereiding voor Machine Learning
#****************************

#Stap 1: Bepaal onafhankelijke en afhankelijke variabelen en splits de data in train- en testsets.
#****************************
y = df["Sentiment"]
X = df["Review"]

# Train / Test split (stratified: class dengesizliği korunur)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Stap 2: Zet tekst om naar numerieke representatie:
#****************************
#a. Maak een TfidfVectorizer object.
#b. Pas fit toe op de trainingsdata.
#c. Transformeer train- en testdata en sla op.
#****************************

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
tf_idf_ngram_vectorizer.fit(X_train)

X_train_tfidf = tf_idf_ngram_vectorizer.transform(X_train)
X_test_tfidf  = tf_idf_ngram_vectorizer.transform(X_test)

print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)
#****************************
#Taak 5: Modellering (Logistische Regressie)
#****************************

#Stap 1: Train een logistisch regressiemodel op de trainingsdata.
#****************************
log_model = LogisticRegression().fit(X_train_tfidf, y_train)

#Stap 2: Voer voorspellingen uit:
#****************************
#a. Gebruik predict om de testdata te voorspellen en sla op.
#b. Bekijk de resultaten met classification_report.
#c. Bereken de gemiddelde accuracy met cross-validation.
#****************************
y_pred = log_model.predict(X_test_tfidf)

report = classification_report(y_test, y_pred)
print(report)

cross_val_score(log_model,
                X_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()

#Stap 3: Voer een willekeurige beoordeling door het model:
#****************************
#a. Selecteer een willekeurige beoordeling met sample uit de kolom Review.
#b. Vectoriseer deze beoordeling met CountVectorizer.
#c. Pas fit en transform toe en sla op.
#d. Voer de voorspelling uit met het model.
#e. Print de beoordeling en het voorspelde resultaat.
#****************************

random_review = pd.Series(df["Review"].sample(5).values)

vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(random_review)
X_all = vectorizer.fit_transform(X_train)

sample_indices = random_review.index
y_sample = df.loc[sample_indices, "Sentiment"]  # veya hedef kolonunuz

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_c, y_sample)


predictions = log_model.predict(X_c)


for review, pred in zip(random_review, predictions):
    print(f"Review: {review}\nPrediction: {pred}\n")

#****************************
#Taak 6: Modellering (Random Forest)
#****************************

#****************************
#Stap 1: Analyseer de voorspellingen met Random Forest:
#a. Train een RandomForestClassifier model.
#b. Bereken de gemiddelde accuracy met cross-validation.
#c. Vergelijk de resultaten met het logistisch regressiemodel.
#****************************

rf_model = RandomForestClassifier().fit(X_train_tfidf , y_train)
log_model.fit(X_all , y_train)
cross_val_score(rf_model, X_train_tfidf, y_train, cv=5, n_jobs=-1).mean()
cross_val_score(log_model, X_train_tfidf, y_train, cv=5, n_jobs=-1).mean()























