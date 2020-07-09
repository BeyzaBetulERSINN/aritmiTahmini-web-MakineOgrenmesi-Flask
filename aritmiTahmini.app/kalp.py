import numpy as np
import pandas as pd
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
import pickle


#Veri setinin yüklenmesi
dataset = pd.read_csv('cleveland.csv')
dataset.info()
#Pandalar dataframe.info()işlevi, veri çerçevesinin özlü bir özetini almak için kullanılır. Verilerin keşif analizi yaparken gerçekten kullanışlı geliyor.
print("veri setinin boyutu:")
print(dataset.shape)  #veri stinin boyutu
print("veri setinin istatistiksel özeti:")
print(dataset.describe())   #veri setinin istatistiksel özeti

print("dataset.hist:")
print(dataset.hist())
plt.show()
print()
print()

rcParams['figure.figsize'] = 8,6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()

dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

#The data is not ready for our Machine Learning application.
#Machine learning için
b = dataset['target']
A = dataset.drop(['target'], axis = 1)

#veri setinin eğitim ve test verileri olarak ayrılması:
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.33, random_state = 0)

#modellerin listesinin oluşturulması
modeller=[
    ('LR', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
    ('NB', GaussianNB())
]
# Modeller için 'cross validation' sonuçlarının  yazdırılması
# K-kat çapraz doğrulama(CV) verileri kıvrımlara bölerek ve her katın bir noktada bir test seti olarak kullanılmasını sağlayarak bu soruna bir çözüm sağlar.
sonuclar=[]
isimler=[]
print("(cv_result.mean)(cv_results.std)")
for isim, model in modeller:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results=cross_val_score(model, A_train,b_train, cv=kfold)#score yerine predict yazılabilir.?
    #    cv_results=model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring="accuary")
    # ValueError: 'accuary' is not a valid scoring value. Use sorted(sklearn.metrics.SCORERS.keys()) to get valid options.
    sonuclar.append(cv_results)
    isimler.append(isim)
    print("%s: %f    (%f)" % (isim, cv_results.mean(), cv_results.std()))

print()
print()

#6.Adım : Uygun algoritmanın seçilmesi ve tahmin yapılması
#confusion matrixi(karşılık matrisi):Karışıklık matrisi, bir sınıflandırıcı tarafından doğru tahmin edilen ve yanlış tahmin edilen değerleri görüntüler.
#Karışıklık matrisinden TP ve TN'NİN toplamı, sınıflandırıcı tarafından doğru sınıflandırılmış girişlerin sayısıdır


print('LogisticRegression:')
lOGr = LogisticRegression()
lOGr.fit(A_train, b_train)   #Modeli Eğitme
prediction_LR = lOGr.predict(A_test)   #Test Seti ile Hedef sınıfları tahmin etme
accuary_LRs=accuracy_score(b_test, prediction_LR)
print('accuracy değeri :',accuary_LRs ) #accuary:doğruluk
print("confusion matrix:")
print(confusion_matrix(b_test, prediction_LR))   #hata matrisi
sns.heatmap(confusion_matrix(b_test, prediction_LR), annot=True, lw=2, cbar=False)
plt.ylabel("True values")
plt.xlabel("Predicted Values")
plt.title("Confusion Matrix Visualization For Logistic Regression")
plt.show()
#f1_scoreLR=f1_score(b_test,prediction_LR)
#print("f1_score:", f1_scoreLR)
print("classification report:")
print(classification_report(b_test, prediction_LR))   #Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
probs=lOGr.predict_proba(A_test)
preds=probs[:,1]
fpr, tpr, treshold = metrics.roc_curve(b_test, prediction_LR)
roc_auc=metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic For Logistic Regression')
plt.plot(fpr, tpr, 'b', label='Auc=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print()
print()



print('Naive Bayes:')
nbIAS = GaussianNB()
nbIAS.fit(A_train, b_train)   #Modeli Eğitme
prediction_NB = nbIAS.predict(A_test)   #Test Seti ile Hedef sınıfları tahmin etme
accuary_NBs=accuracy_score(b_test, prediction_NB)
print('accuracy değeri :',accuary_NBs ) #accuary:doğruluk
print("confusion matrix:")
print(confusion_matrix(b_test, prediction_NB))   #hata matrisi
sns.heatmap(confusion_matrix(b_test, prediction_NB), annot=True, lw=2, cbar=False)
plt.ylabel("True values")
plt.xlabel("Predicted Values")
plt.title("Confusion Matrix Visualization For Naive Bayes")
plt.show()
#f1_scoreNB=f1_score(b_test,prediction_NB)
#print("f1_score:", f1_scoreNB)
print("classification report:")
print(classification_report(b_test, prediction_NB))   #Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
probs=nbIAS.predict_proba(A_test)
preds=probs[:,1]
fpr, tpr, treshold = metrics.roc_curve(b_test, prediction_NB)
roc_auc=metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic For Naive Bayes')
plt.plot(fpr, tpr, 'b', label='Auc=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print()
print()


print('KNN:')
knnF = KNeighborsClassifier()
knnF.fit(A_train, b_train)   #Modeli Eğitme
prediction_KNN = knnF.predict(A_test)   #Test Seti ile Hedef sınıfları tahmin etme
accuary_KNNs=accuracy_score(b_test, prediction_KNN)
print('accuracy değeri :',accuary_KNNs ) #accuary:doğruluk
print("confusion matrix:")
print(confusion_matrix(b_test, prediction_KNN))   #hata matrisi
sns.heatmap(confusion_matrix(b_test, prediction_KNN), annot=True, lw=2, cbar=False)
plt.ylabel("True values")
plt.xlabel("Predicted Values")
plt.title("Confusion Matrix Visualization KNN")
plt.show()
#f1_scoreKNN=f1_score(b_test,prediction_KNN)
#print("f1_score:", f1_scoreKNN)
print("classification report:")
print(classification_report(b_test, prediction_KNN))   #Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
probs=knnF.predict_proba(A_test)
preds=probs[:,1]
fpr, tpr, treshold = metrics.roc_curve(b_test, prediction_KNN)
roc_auc=metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic For KNN')
plt.plot(fpr, tpr, 'b', label='Auc=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print()
print()



print('Decision Tree Classifier:')
dtcF = DecisionTreeClassifier()
dtcF.fit(A_train, b_train)   #Modeli Eğitme
prediction_DTC = dtcF.predict(A_test)   #Test Seti ile Hedef sınıfları tahmin etme
accuary_DTCs= accuracy_score(b_test, prediction_DTC)
print('accuracy değeri :', accuary_DTCs) #accuary:doğruluk
print("confusion matrix:")
print(confusion_matrix(b_test, prediction_DTC))   #hata matrisi
sns.heatmap(confusion_matrix(b_test, prediction_DTC), annot=True, lw=2, cbar=False)
plt.ylabel("True values")
plt.xlabel("Predicted Values")
plt.title("Confusion Matrix Visualization For Decision Tree")
plt.show()
#f1_scoreDTC=f1_score(b_test,prediction_DTC)
#print("f1_score:", f1_scoreDTC)
print("classification report:")
print(classification_report(b_test, prediction_DTC))   #Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
probs=dtcF.predict_proba(A_test)
preds=probs[:,1]
fpr, tpr, treshold = metrics.roc_curve(b_test, prediction_DTC)
roc_auc=metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic For Decision Tree Classifier')
plt.plot(fpr, tpr, 'b', label='Auc=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print()
print()



# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=13, random_state=1)

#veri setinin eğitim ve test verileri olarak ayrılması:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

#print('LogisticRegression:')
lr = LogisticRegression()
lr.fit(X_train, y_train)   #Modeli Eğitme

#Xnew = [[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]
# make a prediction
#ynew = lr.predict(Xnew)
#print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))  #diğer fonksiyonların da doğruluk değeri aynı oldu
#print()
#print()  #bu kısmı flask a aktarabildim


#print('Naive Bayes:')
nb = GaussianNB()
nb.fit(X_train, y_train)   #Modeli Eğitme

#print('KNN:')
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)   #Modeli Eğitme

#print('Decision Tree Classifier:')
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)   #Modeli Eğitme


















