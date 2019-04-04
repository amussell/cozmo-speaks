import LanguageModel as LM
import ImageFeatureGen as IFG
import CS481Dataset as CS
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter

datafile = 'data/newDataset.csv'
pickleFile = 'model/testModel.pickle'

#Load data
print('Loading dataset')
dataOrig = CS.loadFromCSV(datafile)
data = LM.prepareDataset(dataOrig)

#Split Data
train, test = train_test_split(data, test_size=0.2)


#Create model
print('Adding images to model')
langMod = LM.LanguageModel()
langMod.addImages(train)
print('Training model')
langMod.train()

#Test model
print('Testing model')
trainPos = train[train.annotation == 1]
trainX = trainPos.imgFeatures
trainTrue = trainPos.word
trainPred = [langMod.predictImageWord(img=None, imgFeatures=imgF)[0] for imgF in trainX]
trainScore = accuracy_score(trainTrue, trainPred)
print('Training Accuracy: ' + str(trainScore))

wordCount = Counter(trainPos.word)
mostCommon = wordCount.most_common(1)[0]
trainBaseline = mostCommon[1]/len(trainPos)
print("Training Common baseline: " + str(trainBaseline))
print("Training Most Common Class: " + mostCommon[0] + " " + str(mostCommon[1]) + "/" + str(len(trainPos)))

testPos = test[test.annotation == 1]
testX = testPos.imgFeatures
testTrue = testPos.word
testPred = [langMod.predictImageWord(img=None, imgFeatures=imgF)[0] for imgF in testX]
testScore = accuracy_score(testTrue, testPred)
print('Testing Accuracy: ' + str(testScore))

wordCountTest = Counter(testPos.word)
mostCommonTest = wordCountTest.most_common(1)[0]
testBaseline = mostCommonTest[1]/len(testPos)
print("Testing Common baseline: " + str(testBaseline))
print("Testing Most Common Class: " + mostCommonTest[0] + " " + str(mostCommonTest[1]) + "/" + str(len(testPos)))

print('\n\n')
#Test Individual Classifiers
for word in set(test.word):
    model = langMod.wac.wac[word]
    testDataWord = test[test.word == word]
    yTrue = testDataWord.annotation
    yPred = model.predict(testDataWord.imgFeatures.values.tolist())
    score = accuracy_score(yTrue, yPred)
    print('------------------------------------------')
    print(word + " Accuracy: " + str(score))

    #Compute baseline
    count = Counter(yTrue)
    mostCommon = count.most_common(1)[0]
    mostCommonCount = mostCommon[1]
    baseline = mostCommonCount/len(yTrue)
    print(word + "- baseline: " + str(baseline))
    print("Most Common: " + str(mostCommon[0]) + " " + str(mostCommonCount) + "/" + str(len(yTrue)))
    print('------------------------------------------')




with open(pickleFile, 'wb') as handle:
    pickle.dump(langMod, handle)

# Test WAC Nodes


