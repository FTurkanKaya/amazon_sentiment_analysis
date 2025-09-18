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
   - WordCloud van alle reviews

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


