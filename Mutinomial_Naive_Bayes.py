# reading data an constructing it to structural format

import pandas as pd
data = pd.read_csv("./Comments_Data/train_comments.csv")
#data2 = pd.read_csv("./Comments_Data/test_nolabel_comments.csv")
# Preview the first 5 lines of the loaded data
#data.head()
#print(data[['comment','verification_status']][10:25])
#print(data2[['comment']][30:45])

target = data['verification_status']
text = data['comment']

target2 = target[0:120000]
text2 = text[0:120000]

target3 = target[120001:]
text3 = text[120001:]
#print(target[0],len(target))
#print(text[0],len(text))

#print(target2[0],len(target2))
#print(text2[0],len(text2))

#print(text3[120001], len(text3))

#fixing the problem of null comments

fixed_target = target2[pd.notnull(text)]
fixed_text = text2[pd.notnull(text)]


from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(lowercase=True )
count_vec.fit(fixed_text)

#print(len(count_vec.vocabulary_))

counts = count_vec.transform(fixed_text)

#===============================================================================================================
#trainig model    train with naive bayes

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(counts,fixed_target)

#===============================================================================================================
#try the classifer
# nb.predict(count_vec.transform([str(text3[i])]))

i = 120001
resault = []
wrong = 0
correct = 0
for i in range(120001,180000):
    #resault.append(nb.predict(count_vec.transform([str(text3[i])])))
    tmp = nb.predict(count_vec.transform([str(text3[i])]))
    if tmp == target3[i]:
        correct +=1
    else:
        wrong +=1

print("correct : " ,correct)
print("wrong cases : " , wrong)