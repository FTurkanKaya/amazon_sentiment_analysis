# Amazon Sentiment Analyse

## Project Beschrijving
Dit project analyseert klantbeoordelingen van Amazon-producten, met name in de categorieën huistextiel en dagelijkse kleding. Het doel is om inzicht te krijgen in klanttevredenheid en productverbetering door middel van sentimentanalyse en machine learning modellen.

## Dataset
De dataset bevat de volgende kolommen:
- **Review**: De tekst van de beoordeling
- **Title**: Titel van de beoordeling
- **HelpFul**: Aantal gebruikers dat de beoordeling nuttig vond
- **Star**: Aantal sterren gegeven aan het product

## Taken
1. **Tekst Voorbewerking**
   - Kleine letters, verwijdering van interpunctie, cijfers en stopwoorden
   - Lemmatization en filtering van zelden voorkomende woorden

2. **Visualisatie**
   - Barplot van veelvoorkomende woorden
     <img width="640" height="480" alt="barplot" src="https://github.com/user-attachments/assets/03f40b68-adee-404d-bf77-feb9c4592b69" />

   - WordCloud van alle reviews
     <img width="640" height="480" alt="WordCloud" src="https://github.com/user-attachments/assets/b44d5fe2-87b7-4cd8-a92a-90159a3e5650" />


3. **Sentimentanalyse**
   - NLTK SentimentIntensityAnalyzer voor pos/neg labeling

4. **Machine Learning Voorbereiding**
   - Train/test split
   - TF-IDF vectorisatie (2-3 ngram)

5. **Modellering**
   - Logistic Regression
   - Random Forest
   - Vergelijking van cross-validation scores

6. **Evaluatie**
   - Classification report
   - Cross-validation accuracy


