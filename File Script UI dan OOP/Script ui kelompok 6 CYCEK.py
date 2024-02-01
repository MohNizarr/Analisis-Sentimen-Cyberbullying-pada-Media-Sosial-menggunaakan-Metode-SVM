# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from google_trans_new import google_translator
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

# %%
data = pd.read_csv('data_2_inggris_class.csv')

# %%
df = data

# %%
df.head()

# %%
data = df.fillna(' ')

# %%
y= data.Score.values
x= data.Text_English.values

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1, test_size=0.2, shuffle= True)
print(x_train.shape)
print(x_test.shape)

# %%
vectorizer = CountVectorizer(binary= True, stop_words ='english')
vectorizer.fit(list(x_train)+list(x_test))

x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)
print(x_train_vec.shape)
print(x_test_vec.shape)

# %%
rbf= svm.SVC(kernel='rbf', probability= True, C=100, gamma= 0.01 )
prob= rbf.fit(x_train_vec, y_train).predict_proba(x_test_vec)

y_pred_svm_rbf= rbf.predict(x_test_vec)

# %%
translator = google_translator()  

# %%
root = Tk()
root.title("Analisis Sentimen Cyberbullying")
frame_header = ttk.Frame(root)
frame_header.pack()
logo = PhotoImage(file='logo.png', master=frame_header).subsample(2, 2)
logolabel = ttk.Label(frame_header, text='logo', image=logo)
logolabel.grid(row=0, column=0, rowspan=2)

headerlabel = ttk.Label(frame_header, text='CYCEK (Cyberbullying Check)', font=('Arial', 18, 'bold'))
headerlabel.grid(row=0, column=1)
messagelabel = ttk.Label(frame_header, wraplength=300, text='Sistem cyberbullying check ini berfungsi sebagai sarana pengecekan apakah kalimat yang kalian tuliskan termasuk dalam kalimat bullying yang negatif, positif, atau netral.')
messagelabel.grid(row=1, column=1)

frame_content = ttk.Frame(root)
frame_content.pack()
# def submit():
# username = entry_name.get()
# print(username)
myvar = StringVar()
var = StringVar()
# cmnt= StringVar()
commentlabel = ttk.Label(frame_content, text='Input Your Comment: ', font=('Arial', 10))
commentlabel.grid(row=0, column=0, sticky='sw')
textcomment = Text(frame_content, width=55, height=10)
textcomment.grid(row=1, column=0, padx=5, columnspan=2)

textcomment.config(wrap ='word')
# def clear():
# textcomment.delete(1.0,'end')
def clear():
    global textcomment
    messagebox.showinfo(title='Clear', message='Do you want to clear?')
    textcomment.delete(1.0, END)


def submit():
    global textcomment
    text_masuk = str(textcomment.get(1.0, END))  #untuk menyimpan text yang ada di entry box
    translate_text = translator.translate(text_masuk, lang_tgt='en')
    y_pred_svm_rbf= rbf.predict(vectorizer.transform([translate_text]))
    if y_pred_svm_rbf == 'neutral':
        messagebox.showinfo(title='Results', message='Hasil analisis dari sentimen komentar kamu adalah cyberbullying netral')
    elif y_pred_svm_rbf == 'positive':
        messagebox.showinfo(title='Results', message='Hasil analisis dari sentimen komentar kamu adalah cyberbullying positif')
    else :
        messagebox.showinfo(title='Results', message='Hasil analisis dari sentimen komentar kamu adalah cyberbullying negatif')
    textcomment.delete(1.0, END)

submitbutton = ttk.Button(frame_content, text='Detects', command=submit).grid(row=2, column=0, padx=5, pady=5, sticky='e')
clearbutton = ttk.Button(frame_content, text='Clear', command=clear).grid(row=2, column=1, padx=5, pady=5, sticky='w')

root.geometry('500x340')
mainloop()


# %%



