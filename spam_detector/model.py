import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


#reading dataset
df = pd.read_csv("spam.csv")[["v1", "v2"]]
tags = df["v1"]
text = df["v2"]

#Cleaning

#defining stopwords
stop = stopwords.words("english")
stop.append("")

def cleaner(sentences):
    clean_text = []
    for sentence in sentences:
        doc = sentence.lower()   #lower case
        doc = re.findall(r'[a-zA-Z]+', sentence)  #removing numbers and special characters
        doc = [w for w in doc if not w in stop]   #removing stopwords
        doc = " ".join(doc)
        stemmer = SnowballStemmer('english')
        clean = stemmer.stem(doc)
        clean_text.append(clean)
    return clean_text

clean_text = cleaner(text)

from sklearn.feature_extraction.text import CountVectorizer as CV
cv = CV(ngram_range=(0,2), encoding = 'latin', max_features=20000)
X = cv.fit_transform(clean_text)
features = cv.get_feature_names()
dtm = pd.DataFrame(X.toarray(), columns = features)
print dtm.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dtm, tags, test_size = 0.1, random_state = 40)

from sklearn.ensemble import RandomForestClassifier
print "Running Model"
clfrf = RandomForestClassifier(200, n_jobs = -1, bootstrap = True)
clfrf.fit(X_train, y_train)
prediction = clfrf.predict(X_test)
print clfrf.score(X_test, y_test)