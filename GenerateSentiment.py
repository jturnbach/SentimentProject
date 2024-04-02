from textblob import TextBlob
from io import StringIO
import pandas as pd
import json
import re
def getSentimentPolarity(x):
    testimonial=TextBlob(x['text'])
    s=testimonial.sentiment.polarity
    return s
def getSentimentSubjectivity(x):
    testimonial=TextBlob(x['text'])
    s=testimonial.sentiment.subjectivity
    return s
def cleanText(x):
    import re
    str1=x['text']
    str2=re.split('RT @\w+: ',str1)
    if(len(str2))==2:
        return str2[1]
    else:
        return str2[0]
		
inname=str("Tesla/Tesla-2020-3-" + range(10,30))
infile=open(inname+".json", encoding='utf-8')
data = []
for lines in infile:
    try:
        df0=json.loads(lines)
        data.append(df0)
    except:
        continue
infile.close()
df = pd.DataFrame(data)
print(df)

filename=inname
#df=pd.read_json(filename+".json", lines=True, encoding='utf-8')
df=df[['created_at','text']]
df['Polarity']=0
df['Subjectivity']=0
df['text']=df.apply(cleanText, axis=1)
df['Polarity']=df.apply(getSentimentPolarity, axis=1)
df['Subjectivity']=df.apply(getSentimentSubjectivity, axis=1)
df['Aggregate Score']=df['Polarity']*0.5+df['Subjectivity']*0.5
df.to_csv(filename+".csv")
print(df["Polarity"].mean())
print(df["Subjectivity"].mean())
print(df["Aggregate Score"].mean())


df