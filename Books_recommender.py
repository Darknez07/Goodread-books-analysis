# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


# %%
rsv = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)


# %%
rsv.head()


# %%
fn = rsv.authors.unique()
thresh = 0
writers = dict()
for f in fn:
    fc = rsv[rsv.authors == f].bookID.count() 
    if fc >= thresh: 
        writers[f] = fc


# %%
import matplotlib.pyplot as plt


# %%
writers = dict(sorted(writers.items(), key=lambda item: item[1], reverse=True)[:10])

# %% [markdown]
# ## Simple Counting of writers
# Thus obtained graph is shown as follows
# showing that P.G. Woodhouse and Stephen King are equally dominating

# %%
plt.figure(figsize=(40, 40))
plt.barh(range(len(writers)), list(writers.values()), align='center')
plt.yticks(range(len(writers)), list(writers.keys()), fontsize=30)
plt.show();


# %%
writers

# %% [markdown]
# # Question mark 
# ON the Authors columns, Clearly multiple authors are seprated by a "/" which is "incorrect" for direct
# data analysis. So after figuring out a "function" which does the following:->
# * It looks for '/' in the authors
# * Seprates them if normally acessed
# * But if the book is already seen, maximum number of writers are considered
# * Hence we find that actual count was incorrect
# * Also includes single writers
# 

# %%
title_arr = []
writers_arr = []
single_writer = []
for f in fn:
    if '/' in f:
        try:
            book = rsv[rsv.authors == f].title.values[0]
            if book in title_arr:
                if len(writers_arr[title_arr.index(book)]) < len(f.split('/')):
                    writers_arr[title_arr.index(book)] = f.split('/')
            else:
                title_arr.append(book)
                writers_arr.append(f.split('/'))
        except IndexError:
            title_arr.append(rsv[rsv.authors == f].title.values[0])
            writers_arr.append(f.split('/'))


# %%
len(title_arr)


# %%
len(title_arr)


# %%
len(set(title_arr))


# %%
unique_writes = set()
count = 0
max_len = 0
for x,y  in zip(title_arr, writers_arr):
    if len(y) > max_len:
        max_len = len(y)
    for z in y:
        unique_writes.add(z)
        count+=1


# %%
for k in writers.keys():
    if '/' not in k:
        unique_writes.add(k)
        count+=1


# %%
len(unique_writes)


# %%
count


# %%
get_books = dict.fromkeys(unique_writes, 0)


# %%
for x,y in zip(title_arr, writers_arr):
    for z in y:
        get_books[z]+=1


# %%
for x in writers.items():
    if '/' not in x[0]:
        get_books[x[0]]+=x[1]

# %% [markdown]
# ## Looking into Titles
# We find that twilight has maximum reviews and rating of 3.59
# Although max rating among top 10 is of Harry Potter and Half blood Prince

# %%
rsv.sort_values(by=['ratings_count'],
                ascending=False).loc[:,("title","average_rating")][:10]


# %%
get_books = dict(sorted(get_books.items(), key=lambda item: item[1], reverse=True))

# %% [markdown]
# ## The correct visualization
# The visualization is perfect as everything is taken into account, hence Seeing Stephen King as a dominant

# %%
plt.figure(figsize=(40, 40));
plt.barh(range(10), list(get_books.values())[:10], align='center')
plt.yticks(range(10), list(get_books.keys())[:10], fontsize=30)
plt.xticks(range(100), range(100), fontsize=25, rotation=90)
plt.show();

# %% [markdown]
# 

# %%
import datetime


# %%
dates = []
for x in rsv.publication_date.values:
    try:
        dates.append(pd.to_datetime(datetime.datetime.strptime(x,"%m/%d/%Y")))
    except:
        dates.append(pd.to_datetime(datetime.datetime.now()))


# %%
rsv["New dates"] = dates

# %% [markdown]
# ## Incorrect dates
# As you can see only 2 dates are incorrect in the dateset which can effect the anaylsis

# %%
rsv.sort_values(by='New dates',ascending=False).loc[:,("title","New dates")]


# %%
rsv["Rating_Interval"] = pd.DataFrame(pd.cut(rsv.average_rating, 5, [0.0,1.0,2.0,3.0,4.0,5.0]))


# %%
ratings = dict()
for i in range(0,5,1):
    ratings[str(i)+" to "+str(i+1)] = (rsv["Rating_Interval"] == pd.Interval(left = float(i), right = float(i+1))).sum()


# %%
del ratings['0 to 1']


# %%
ratings


# %%
plt.figure(figsize=(40, 40));
plt.barh(range(len(ratings)), list(ratings.values()), align='center')
plt.yticks(range(len(ratings)), list(ratings.keys()), fontsize=30)
# plt.yticks(range(50), range(50), fontsize=30)
plt.show();

# %% [markdown]
# ## Ratings
# From the above graph we can clearly see that the dominating rating is
# 3 to 4, moreover the average lies between 3 to 4.
# This is skewed data, as well an indication that the 
# dataset given has more 3 to 5 star books within the sample

# %%
lang = dict()


# %%
rsv.language_code.unique()


# %%
lang["eng"] = 0


# %%
skip = ["eng","en-US","en-CA","en-GB"]


# %%
unq_lang = rsv.language_code.unique()


# %%
for x in unq_lang:
    if x not in skip:
        lang[x] = rsv[rsv.language_code == x].bookID.count()
    if x in skip:
        lang["eng"]+=rsv[rsv.language_code == x].bookID.count()


# %%
lang


# %%
plt.figure(figsize=(40, 40))
plt.barh(range(len(lang)), width=sorted(list(lang.values()))[::-1], align='center')
plt.yticks(range(len(lang)), list(lang.keys()), fontsize=30)
plt.show();

# %% [markdown]
# ## Observing Above graph
# English is most dominating language,
# but in the next bar graph Down below if
# English is removed we see graph looks normal
# and not totally dominant by a single language

# %%
del lang["eng"]


# %%
plt.figure(figsize=(40, 40))
plt.barh(range(len(lang)), width = sorted(list(lang.values()))[::-1], align='center')
plt.yticks(range(len(lang)), list(lang.keys()), fontsize=30)
plt.show();


# %%
rsv = rsv.set_index('bookID')


# %%
rsv

# %% [markdown]
# # Wrong dates
# BookId
# 45531
# 31373

# %%
rsv = rsv.drop([45531, 31373])


# %%
l = rsv.publisher.value_counts()
l = l[l >= 20]


# %%
plt.figure(figsize=(20, 20))
plt.barh(range(10), width=l[:10])
plt.yticks(range(10), l.index.values[:10], fontsize= 25)
plt.show()


# %%
def calculate(word):
    return ord(word)

# %% [markdown]
# # Encoding
# This function is for encoding Authors, This function was made after a lot of thought such that to have less than 5 % clashes

# %%
def encode(strs):
    if '/' in strs:
        final_sum = 0
        intr = strs.split('/')
        for s in intr:
            noramlize = len(s)
            summation = sum([calculate(x) for x in s])
        final_sum+=(summation/noramlize)
        return final_sum
    else:
        return sum([calculate(x) for x in strs])/len(strs)


# %%
rsv.insert(2,'Encoded authors', rsv.authors.apply(encode))


# %%
encode(rsv.authors.values[0])


# %%
import math

# %% [markdown]
# # Encoding titles
# This function was made to stand out with log and bias to length of the title,****Because**** as follows:->
# * A title depends on it's length(talking syntaxically)
# * And moreover it depends upon arrangement of words such  as("a after p" or "p after a") are different things.

# %%
def encode_title(tt):
    total = 0
    for w in tt.split(' '):
        if len(w) == 0:
            continue
        total+= sum([calculate(x)*math.log2(lg + len(w)) for lg,x in enumerate(w)])/ len(w)
    return total/len(tt)


# %%
l = []
for x in rsv.title.values:
    l.append(encode(x))


# %%
len(l), len(set(l))


# %%
rsv.insert(1,'Encoded_titles',rsv.title.apply(encode_title)/rsv.average_rating.values)


# %%
rsv.head()


# %%
rsv[rsv.isna() == False]

# %% [markdown]
# # Constructing x
# For KMeans

# %%
rsv.columns


# %%
x = rsv[['Encoded_titles', 'Encoded authors', 'average_rating', '  num_pages','ratings_count','text_reviews_count', 'New dates']]


# %%
x


# %%
x['New dates'] = x['New dates'].astype(np.int64)


# %%
x['New dates'].astype(np.float64)


# %%
from sklearn.cluster import KMeans


# %%
x = x.reset_index()


# %%
x = x.drop('bookID',axis=1)

# %% [markdown]
# # Date conversion

# %%
x['New dates'] = x['New dates'] //  10**12

# %% [markdown]
# # The dtypes
# All of them are numeric

# %%
x.dtypes


# %%
x = x[x['New dates'] > 0]

# %% [markdown]
# # Adjusting
# * Dates were < 0 in timestamp format
# * The Encoded Titles had an infinity Thanks to [This guy](https://www.kaggle.com/carlosdg) for helping me.

# %%
x = x[x['Encoded_titles'] != np.inf]

# %% [markdown]
# # Doing elbow method for n
# Looking for wcss which is optimal and hence obtain correct amount of categoires.

# %%
wcss = [] 
for i in range(1,11):
    clusters = KMeans(n_clusters = i, random_state = 42) 
    clusters.fit(x.values)
    wcss.append(clusters.inertia_)


# %%
plt.plot(range(1,11),wcss,'b')
plt.title('This is Clustering')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.plot(4,wcss[3],'.',mew = 4, ms =8,color = 'r')
plt.annotate(xy = [4,wcss[3]],s='(%.2f , %.1f) Seems to be Optimal'%(4,wcss[3]))
plt.vlines(4,0,wcss[3],linestyle='dashed')
plt.hlines(wcss[3],0,4,linestyle='dashed')
plt.xlim(0,None)
plt.ylim(0,None)


# %%
clusters = KMeans(n_clusters=4, random_state=42)


# %%
x['cluster'] = clusters.fit_predict(x)


# %%
x

# %% [markdown]
# # Suggestion Logic
# %% [markdown]
# ## Find Mins
# * The function finds 10 minimum from a threshold value
# * and then get the indexes of them from dataframe
# * So the encoded values from a text based entry is matched
# * The logic bieng that the similiar texts are encoded equally

# %%
def find_mins(df, thresh):
    indexes = dict()
    for i in range(df.shape[0]):
        indexes[df.iloc[i,1]] = i
    one = []
    two = []
    for x,y in indexes.items():
        if x <= thresh:
            one.append(x)
        else:
            two.append(x)
    one = sorted(one, reverse=True)[:10]
    two = sorted(two)[:10]
    get_one = [df.iloc[indexes[val]] for val in one] 
    get_two = [df.iloc[indexes[val]] for val in two]
    see = pd.DataFrame((get_one + get_two))
    return see

# %% [markdown]
# ## Find correct values
# * This function tries to find the colums which have similiarty with Encoded titles
# * Also if n is greater than the total space of the selected items it returns all selected
# * Sorting with **number of Pages** so that we get most fat book

# %%
def find_inarr(df,n):
    lol = []
    for i in range(df.shape[0]):
        lol.append(rsv[rsv['Encoded_titles'] == df.iloc[i, 0]])
    lol = pd.concat(lol)
    if n > lol.shape[0]:
        n = lol.shape[0]
    return lol.sort_values(["  num_pages"], ascending=False).iloc[:n]

# %% [markdown]
# ## Main Suggestion Function
# **Before any of below a clustering is done already, so we do this within the same cluster**
# * This function is the actual suggestion function
# * It first finds the same author as the book passed
# * IF yes, returns the minium **diff value** that is matematically the nearest point
# * IF no, we construct a new dataframe for final according to above functions
# * When these are parse the values are sorted with **Titles** **Rating** and **Date Published**
# * And final is to returned according to number of pages

# %%
def suggest(df,n=1):
    selector = x[x['cluster'] == df.iloc[-1]]
    finals = selector[selector['Encoded authors'] == df['Encoded authors']]
    if finals.shape[0] == 1:
        finals = find_mins(selector, df['Encoded authors'])
    middle = finals - df
    suggest = middle.abs().sort_values(["Encoded_titles","average_rating","New dates"]).iloc[:-1,:]
    suggest = finals.loc[suggest.index]
#     print(suggest.iloc[:, 0])
    return find_inarr(suggest,n)

# %% [markdown]
# # Real

# %%
rsv[rsv['Encoded_titles'] == x.iloc[1311,0]]

# %% [markdown]
# # Suggested

# %%
suggest(x.iloc[1311], 2)

# %% [markdown]
# # Real

# %%
rsv[rsv['Encoded_titles'] == x.iloc[5,0]]

# %% [markdown]
# # Suggested

# %%
suggest(x.iloc[5])


# %%



