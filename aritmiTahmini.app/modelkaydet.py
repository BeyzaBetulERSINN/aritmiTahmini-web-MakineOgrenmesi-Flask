import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
# from matplotlib.cm import rainbow
# %matplotlib inline
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
import pickle

from sklearn.datasets import make_blobs

# Veri setinin yüklenmesi
dataset = pd.read_csv('cleveland.csv')
dataset.info()
# Pandalar dataframe.info()işlevi, veri çerçevesinin özlü bir özetini almak için kullanılır. Verilerin keşif analizi yaparken gerçekten kullanışlı geliyor.
print(dataset.shape)  # veri stinin boyutu
print(dataset.describe())  # veri setinin istatistiksel özeti

rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()
plt.show()

print(dataset.hist())

rcParams['figure.figsize'] = 8, 6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color=['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()

dataset = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Now, I will use the StandardScaler from sklearn to scale my dataset.
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
print(dataset)  # veri setinin içeriği

# The data is not ready for our Machine Learning application.
# Machine learning için
y = dataset['target']
X = dataset.drop(['target'], axis=1)

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=13, random_state=1)#her doğruluk değeri 1 çıkıyor.


# veri setinin eğitim ve test verileri olarak ayrılması:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# sıra, farklı modelleri uygulayıp, crossvalidation sonuçlarını karşlaştırarak en uygun modeli seçmekte

# modellerin listesinin oluşturulması
models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('RFC', RandomForestClassifier()),
    ('SVM', SVC())
]
# Modeller için 'cross validation' sonuçlarının  yazdırılması
# K-kat çapraz doğrulama(CV) verileri kıvrımlara bölerek ve her katın bir noktada bir test seti olarak kullanılmasını sağlayarak bu soruna bir çözüm sağlar.
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)  # score yerine predict yazılabilir.?
    #    cv_results=model_selection.cross_val_score(model, X_train,Y_train, cv=kfold, scoring="accuary")
    # ValueError: 'accuary' is not a valid scoring value. Use sorted(sklearn.metrics.SCORERS.keys()) to get valid options.
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

print()
print()

# 6.Adım : Uygun algoritmanın seçilmesi ve tahmin yapılması
# confusion matrixi(karşılık matrisi):Karışıklık matrisi, bir sınıflandırıcı tarafından doğru tahmin edilen ve yanlış tahmin edilen değerleri görüntüler.
# Karışıklık matrisinden TP ve TN'NİN toplamı, sınıflandırıcı tarafından doğru sınıflandırılmış girişlerin sayısıdır
print('SVC:')
svc = SVC()
svc.fit(X_train, y_train)  # Modeli Eğitme
predictions_SVC = svc.predict(X_test)  # Test Seti ile Hedef sınıfları tahmin etme
# print('accuracy degeri :', accuracy_score(y_test, predictions_SVC)) #accuary:doğruluk
cm_test = confusion_matrix(y_test, predictions_SVC)
print('cm_test:')
print(cm_test)  # hata matrisi
print('Accuracy of SVC for test set = {}'.format((cm_test[0][0] + cm_test[1][1]) / (len(X_test))))
print(classification_report(y_test,
                            predictions_SVC))  # Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
print()
print()

print('RandomForestClassifier:')
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)  # Modeli Eğitme
predictions_RFC = rfc.predict(X_test)  # Test Seti ile Hedef sınıfları tahmin etme
accuary_RFC = accuracy_score(y_test, predictions_RFC)
print('accuracy degeri :', accuary_RFC)  # accuary:doğruluk
print(confusion_matrix(y_test, predictions_RFC))  # hata matrisi
print(classification_report(y_test,
                            predictions_RFC))  # Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
print()
print()

print('LogisticRegression:')
lr = LogisticRegression()
lr.fit(X_train, y_train)  # Modeli Eğitme
# predictions_LR = lr.predict(X_test)   #Test Seti ile Hedef sınıfları tahmin etme
# accuary_LR=accuracy_score(y_test, predictions_LR)
# print('accuracy degeri :',accuary_LR ) #accuary:doğruluk
# print(confusion_matrix(y_test, predictions_LR))   #hata matrisi
# print(classification_report(y_test, predictions_LR))   #Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
# print()
# print()
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(lr, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
print()
print()  #bunda haat vermedi fakat ikinci çalıştırıldığında dosyadan kaynaklı olarak hata veriyor ve flaskta çağırıldığında hata veriyor.
#şu an ne zaman çalıştırırsam çalıştırim çalışıyor.

print('Naive Bayes:')
nb = GaussianNB()
nb.fit(X_train, y_train)  # Modeli Eğitme
predictions_NB = nb.predict(X_test)  # Test Seti ile Hedef sınıfları tahmin etme
accuary_NB = accuracy_score(y_test, predictions_NB)
print('accuracy degeri :', accuary_NB)  # accuary:doğruluk
print(confusion_matrix(y_test, predictions_NB))  # hata matrisi
print(
    classification_report(y_test, predictions_NB))  # Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
print()
print()

print('KNN:')
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # Modeli Eğitme
predictions_KNN = knn.predict(X_test)  # Test Seti ile Hedef sınıfları tahmin etme
accuary_KNN = accuracy_score(y_test, predictions_KNN)
print('accuracy degeri :', accuary_KNN)  # accuary:doğruluk
print(confusion_matrix(y_test, predictions_KNN))  # hata matrisi
print(classification_report(y_test,
                            predictions_KNN))  # Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
print()
print()

print('Decision Tree Classifier:')
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)  # Modeli Eğitme
predictions_DTC = dtc.predict(X_test)  # Test Seti ile Hedef sınıfları tahmin etme
accuary_DTC = accuracy_score(y_test, predictions_DTC)
print('accuracy degeri :', accuary_DTC)  # accuary:doğruluk
print(confusion_matrix(y_test, predictions_DTC))  # hata matrisi
print(classification_report(y_test,
                            predictions_DTC))  # Ana sınıflandırma metriklerini gösteren bir metin raporu oluşturun
print()
print()







