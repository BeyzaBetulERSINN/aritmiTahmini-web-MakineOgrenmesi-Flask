from flask import Flask,render_template,flash,redirect,url_for,session,logging,request
from flask_mysqldb import MySQL
from wtforms import Form,StringField,TextAreaField,PasswordField,validators
from passlib.hash import sha256_crypt
from functools import wraps
import kalp
from numpy import *
import pickle   #kaydedilmiş modeli kullanmak için
# Kullanıcı Giriş Decorator'ı
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "logged_in" in session:
            return f(*args, **kwargs)
        else:
            flash("Bu sayfayı görüntülemek için lütfen giriş yapın.","danger")
            return redirect(url_for("login"))

    return decorated_function
# Kullanıcı Kayıt Formu
class RegisterForm(Form):
    name = StringField("İsim Soyisim",validators=[validators.Length(min = 4,max = 25)])
    username = StringField("Kullanıcı Adı",validators=[validators.Length(min = 5,max = 35)])
    email = StringField("Email Adresi",validators=[validators.Email(message = "Lütfen Geçerli Bir Email Adresi Girin...")])
    password = PasswordField("Parola:",validators=[
        validators.DataRequired(message = "Lütfen bir parola belirleyin"),
        validators.EqualTo(fieldname = "confirm",message="Parolanız Uyuşmuyor...")
    ])
    confirm = PasswordField("Parola Doğrula")
class LoginForm(Form):
    username = StringField("Kullanıcı Adı")
    password = PasswordField("Parola")
app = Flask(__name__)
app.secret_key= "ybblog"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "ybblog"
app.config["MYSQL_CURSORCLASS"] = "DictCursor"

mysql = MySQL(app)

@app.route('/')
def entry_page() -> 'html':
    return render_template('degerler.html', page_title='Kalp Aritmi Tespit Sayfası')

@app.route('/degeral', methods=['POST'])
def sum() -> 'html':


    a = int(request.form['age'])
    b = int(request.form['trestbps'])
    c = int(request.form['chol'])
    d = int(request.form['thalach'])
    e = float(request.form['oldpeak'])
    f = int(request.form['ca'])

   # g = float(request.form.get('sex'))
    g = int(request.form['sex'])
    if(g==0):
        cins='Kadın'
    else:
        cins='Erkek'

    h = int(request.form['cp'])
    if(h==1):
        agriTipi='typical angina'
    elif(h==2):
        agriTipi='atypical angina'
    elif (h == 3):
        agriTipi = 'non - anginal pain'
    elif (h == 4):
        agriTipi = 'asymptotic'

    k = int(request.form['fbs'])
    if (k == 0):
        sekhas = 'şeker hastası değil'
    elif (k == 1):
        sekhas = 'şeker hastası'

    m = int(request.form['restecg'])
    if(m==0):
        elektrocard='normal'
    elif(m==1):
        elektrocard='having ST-T wave abnormality'
    elif (m == 2):
        elektrocard= 'left ventricular hyperthrophy'

    n = int(request.form['exang'])
    if (n == 0):
        gagrisi = 'göğüs ağrısı yok'
    elif (n == 1):
        gagrisi = 'göğüs ağrısı var'

    o = int(request.form['slope'])
    if (o == 1):
        stSegmenti = 'upsloping'
    elif (o == 2):
        stSegmenti = 'flat'
    elif (o == 3):
        stSegmenti = 'downsloping'

    p = int(request.form['thal'])
    if (p == 1):
        hasar = 'normal'
    elif (p == 2):
        hasar = 'fixed defect'
    elif (p == 3):
        hasar = 'reversable defect'



    r = request.form['isim']

    degerler = [a, b, c, d, e, f, g, h, k, m, n, o, p]
    print(degerler)

    #alınan değerleri dosyaya yazıyoruz
    sayim = 1
    dosya = open('degerler.csv', 'w')  # her çalıştığında içeriği siliniyor
    # dosya = open(degerler.csv, "a")
    # with open("degerler.csv", "a") as dosya:
    #   dosya.write("\nSelin Özden\t: 0212 222 22 22")
    for i in degerler:
        deger = i
        degerstr = str(deger)
        print(degerstr)
        dosya.write(degerstr)
        if sayim < 13:
            dosya.write(",")
            sayim = sayim + 1
    dosya.close()


    liste2 = [degerler]  # iki boyutlu liste oluşturduk
    print(liste2)

    # prediction = model.predict(liste2)
    # print('LR TAHMİNİ')
    # print(prediction)      #kaydedilmiş finalized_model.sav ı yukarda modele atamıştık ama hata verdi
    # modelkaydet i çalıştırdıktan sonra helloflaskı çalıştırdık(modeli eğitmiş olduk) ve çalıştı. Aşağıdaki kodlarla aynı işlemi gördü.

    ynew = kalp.lr.predict(liste2)  # kalp.py den lr modelini çağırdık
    print("X=%s, Predicted=%s" % (liste2[0], ynew[0]))  # diğer fonksiyonların da doğruluk değeri aynı oldu
    if ynew[0] == 1:
        sonucLR = 'aritmi tespit edildi'
    elif ynew[0] == 0:
        sonucLR = 'aritmi tespit edilmedi'
    print(sonucLR)
    dogrulukLR = kalp.accuary_LRs

    ynewNB = kalp.nb.predict(liste2)  # kalp.py den lr modelini çağırdık
    print("X=%s, Predicted=%s" % (liste2[0], ynew[0]))  # diğer fonksiyonların da doğruluk değeri aynı oldu
    if ynewNB[0] == 1:
        sonucNB = 'aritmi tespit edildi'
    elif ynewNB[0] == 0:
        sonucNB = 'aritmi tespit edilmedi'
    print(sonucNB)
    dogrulukNB = kalp.accuary_NBs

    ynewKNN = kalp.knn.predict(liste2)  # kalp.py den lr modelini çağırdık
    print("X=%s, Predicted=%s" % (liste2[0], ynew[0]))  # diğer fonksiyonların da doğruluk değeri aynı oldu
    if ynewKNN[0] == 1:
        sonucKNN = 'aritmi tespit edildi'
    elif ynewKNN[0] == 0:
        sonucKNN = 'aritmi tespit edilmedi'
    print(sonucKNN)
    dogrulukKNN = kalp.accuary_KNNs

    ynewDTC = kalp.dtc.predict(liste2)  # kalp.py den lr modelini çağırdık
    print("X=%s, Predicted=%s" % (liste2[0], ynew[0]))  # diğer fonksiyonların da doğruluk değeri aynı oldu
    if ynewDTC[0] == 1:
        sonucDTC = 'aritmi tespit edildi'
    elif ynewDTC[0] == 0:
        sonucDTC = 'aritmi tespit edilmedi'
    print(sonucDTC)
    dogrulukDTC = kalp.accuary_DTCs




    return render_template('result.html', page_title='Calculation result',

                           yas=a, dinkanbas=b, kolesterol=c, maxkalpr=d, st=e, anadamarsay=f,
                           cinsiyet=cins, agritipi=agriTipi, sekerhastaligi=sekhas, ecg=elektrocard,
                           gogusagrisi=gagrisi, stsegmenti=stSegmenti, hasarorani=hasar,
                           ad=r, durumLR=sonucLR, dogruluk1=dogrulukLR, durumNB=sonucNB, dogruluk2=dogrulukNB,
                           durumKNN=sonucKNN, dogruluk3=dogrulukKNN, durumDTC=sonucDTC, dogruluk4=dogrulukDTC,

                           )

                           
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/anasayfa")
def anasayfa():
    return render_template("anasayfa.html")
    
# Makale Sayfası
@app.route("/articles")
def articles():
    cursor = mysql.connection.cursor()

    sorgu = "Select * From articles"

    result = cursor.execute(sorgu)

    if result > 0:
        articles = cursor.fetchall()
        return render_template("articles.html",articles = articles)
    else:
        return render_template("articles.html")

@app.route("/dashboard")
@login_required

def dashboard():
    cursor = mysql.connection.cursor()

    sorgu = "Select * From articles where author = %s"

    result = cursor.execute(sorgu,(session["username"],))

    if result > 0:
        articles = cursor.fetchall()
        return render_template("dashboard.html",articles = articles)
    else:
        return render_template("dashboard.html")
#Kayıt Olma
@app.route("/register",methods = ["GET","POST"])
def register():
    form = RegisterForm(request.form)

    if request.method == "POST" and form.validate():
        name = form.name.data
        username = form.username.data
        email = form.email.data
        password = sha256_crypt.encrypt(form.password.data)

        cursor = mysql.connection.cursor()

        sorgu = "Insert into users(name,email,username,password) VALUES(%s,%s,%s,%s)"

        cursor.execute(sorgu,(name,email,username,password)) #demet olarak veriyor
        mysql.connection.commit() #veritabanından herhangi bir değişiklik yaptığımda söylemem gerekiyor(silme ,güncelleme)


        cursor.close() #mysql bağlantısını kapatıyorum
        
        flash("Başarıyla Kayıt Oldunuz...","success")
        return redirect(url_for("login"))  #bunu nex yap olmazsa
    else:
        return render_template("register.html",form = form)
# Login İşlemi
@app.route("/login",methods =["GET","POST"])
def login():
    form = LoginForm(request.form)
    if request.method == "POST":  #formdan gelen bilgileri almak için
       username = form.username.data
       password_entered = form.password.data #buraya kadar verileri aldım

       cursor = mysql.connection.cursor() #burada verileri veritabanı ile sorgulamak için

       sorgu = "Select * From users where username = %s"

       result = cursor.execute(sorgu,(username,))

       if result > 0:
           data = cursor.fetchone()
           real_password = data["password"]
           if sha256_crypt.verify(password_entered,real_password):
               flash("Başarıyla Giriş Yaptınız...","success")

               session["logged_in"] = True
               session["username"] = username

               return redirect(url_for("anasayfa"))  #BURASIIIIIIIIIIIIIIIIII degeral yap (index di)
           else:
               flash("Parolanızı Yanlış Girdiniz...","danger")
               return redirect(url_for("login")) 

       else:
           flash("Böyle bir kullanıcı bulunmuyor...","danger")
           return redirect(url_for("login"))

    
    return render_template("login.html",form = form)

# Detay Sayfası

@app.route("/article/<string:id>")  #makalenin id'sini alıyorum
def article(id):
    cursor = mysql.connection.cursor()
    
    sorgu = "Select * from articles where id = %s"

    result = cursor.execute(sorgu,(id,))

    if result > 0:
        article = cursor.fetchone()
        return render_template("article.html",article = article)
    else:
        return render_template("article.html")
# Logout İşlemi
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("anasayfa"))
# Makale Ekleme
@app.route("/addarticle",methods = ["GET","POST"])
def addarticle():
    form = ArticleForm(request.form)
    if request.method == "POST" and form.validate():
        title = form.title.data
        content = form.content.data

        cursor = mysql.connection.cursor()

        sorgu = "Insert into articles(title,author,content) VALUES(%s,%s,%s)"

        cursor.execute(sorgu,(title,session["username"],content))

        mysql.connection.commit()

        cursor.close()

        flash("Makale Başarıyla Eklendi","success")

        return redirect(url_for("dashboard"))

    return render_template("addarticle.html",form = form)

#Makale Silme
@app.route("/delete/<string:id>")
@login_required
def delete(id):
    cursor = mysql.connection.cursor()

    sorgu = "Select * from articles where author = %s and id = %s"

    result = cursor.execute(sorgu,(session["username"],id))

    if result > 0:
        sorgu2 = "Delete from articles where id = %s"

        cursor.execute(sorgu2,(id,))

        mysql.connection.commit()

        return redirect(url_for("dashboard"))
    else:
        flash("Böyle bir makale yok veya bu işleme yetkiniz yok","danger")
        return redirect(url_for("index"))
#Makale Güncelleme
@app.route("/edit/<string:id>",methods = ["GET","POST"])
@login_required
def update(id):
   if request.method == "GET":
       cursor = mysql.connection.cursor()

       sorgu = "Select * from articles where id = %s and author = %s"
       result = cursor.execute(sorgu,(id,session["username"]))

       if result == 0:
           flash("Böyle bir makale yok veya bu işleme yetkiniz yok","danger")
           return redirect(url_for("index"))
       else:
           article = cursor.fetchone()
           form = ArticleForm()

           form.title.data = article["title"]
           form.content.data = article["content"]
           return render_template("update.html",form = form)

   else:
       # POST REQUEST
       form = ArticleForm(request.form)

       newTitle = form.title.data
       newContent = form.content.data

       sorgu2 = "Update articles Set title = %s,content = %s where id = %s "

       cursor = mysql.connection.cursor()

       cursor.execute(sorgu2,(newTitle,newContent,id))

       mysql.connection.commit()

       flash("Makale başarıyla güncellendi","success")

       return redirect(url_for("dashboard"))

       
# Makale Form
class ArticleForm(Form):
    title = StringField("Makale Başlığı",validators=[validators.Length(min = 5,max = 100)]) 
    content = TextAreaField("Makale İçeriği",validators=[validators.Length(min = 10)])

# Arama URL
@app.route("/search",methods = ["GET","POST"])
def search():
   if request.method == "GET":
       return redirect(url_for("index"))
   else:
       keyword = request.form.get("keyword")

       cursor = mysql.connection.cursor()

       sorgu = "Select * from articles where title like '%" + keyword +"%'"

       result = cursor.execute(sorgu)

       if result == 0:
           flash("Aranan kelimeye uygun makale bulunamadı...","warning")
           return redirect(url_for("articles"))
       else:
           articles = cursor.fetchall()

           return render_template("articles.html",articles = articles)
if __name__ == "__main__":
    app.run(debug=True)
