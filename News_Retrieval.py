import pandas as pd
from newsapi import NewsApiClient

# Initialisation
newsapi = NewsApiClient(api_key='dcb7e5eba8914cdda61d0c092d776c92')

# Retrieve all the news written in English between specified date
all_articles = newsapi.get_everything(q='coronavirus',
                                      from_param='2020-02-19',
                                      to='2020-03-18',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)

# Convert retrieved information into a csv file
def CreateDF(JsonArray,columns):
    dfData = pd.DataFrame()

    for item in JsonArray:
        itemStruct = {}

        for cunColumn in columns:
            itemStruct[cunColumn] = item[cunColumn]

        dfData = dfData.append(itemStruct,ignore_index=True)

    return dfData


columns = ['author', 'publishedAt', 'title', 'description', 'content', 'url']
df = CreateDF(all_articles['articles'], columns)
df.to_csv('news.csv')
