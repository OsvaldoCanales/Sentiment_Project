#Import data analysis tools
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

plt.style.use('ggplot')

#Import Natural Language Took Kit and packages
import nltk 
#Package that anaylzes text and sends back mood feedback
from nltk.sentiment import SentimentIntensityAnalyzer
#Package that lets you see the progress as you run the code through loops
from tqdm.notebook import tqdm

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
#Module that provides sentiment scores based on the words used
nltk.download('vader_lexicon')


#Read in data
file_path = "C:/Users/Creepy Weasel/Downloads/archive/Reviews.csv"
df = pd.read_csv(file_path)

#Limit dataframe to the first 500 rows 
df = df.head(500)
print(df.shape)

'''
#Plot the count of reviews by starts
ax = df['Score'].value_counts().sort_index()  \
    .plot(kind ='bar',
          title ='Count of Reviews by Stars',
          figsize = (10,5))
ax.set_xlabel("Review Stars")
'''

#Plot the bar graph
#plt.show()

#Print the text from the 50th row or review
example = df['Text'].values[50]
print(example)

#Print the first 10 tokens that divide the string 
tokens = nltk.word_tokenize(example)
print(tokens[:10])

#Print the first 10 tokens and their parts of speech
tagged = nltk.pos_tag(tokens)
print(tagged[:10])

#Take tags and group them into chunks of tesxt
entitites = nltk.chunk.ne_chunk(tagged)
entitites.pprint()

#Use NLTK's SentimentIntensityAnalyzer to get the neg/neu/pos scores of the text
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores('I am so happy!'))
print(sia.polarity_scores('This is the worst thing ever!'))
print(sia.polarity_scores(example))

#Run the polarity score on the entire dataset 
res = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

pprint(res)

#Store into Pandas dataframe
vaders = pd.DataFrame(res).T
#Reset index and repece index with Id
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
#Merge scores onto df dataframe 
vaders = vaders.merge(df, how = 'left')
print(vaders.dtypes)

#Plot Vader results
#Note to self: We use seaborn to integrate colosely with pandas data structures
ax = sns.barplot(data = vaders, x = 'Score', y = 'compound')
ax.set_title('Compound Score by Amazon Star review')
plt.show()

#Plot three subplots side by side that showcase the scores on the three sentimental scores
fig, ax1 = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data = vaders, x = 'Score', y = 'pos', ax=ax1[0])
sns.barplot(data = vaders, x = 'Score', y = 'neu', ax=ax1[1])
sns.barplot(data = vaders, x = 'Score', y = 'neg', ax=ax1[2])
ax1[0].set_title('Positive')
ax1[1].set_title('Neutral')
ax1[2].set_title('Negative')

plt.show()


    