from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from lemmatizer import *
import pandas as pd

def predict_theme(question):
    df_questions = pd.read_csv('csv_files/questions.csv')
    df_themes = pd.read_csv('csv_files/theme.csv')

    texts = df_questions['questions'].tolist()
    classes = df_themes['theme'].tolist()

    texts_train, texts_test, classes_train, classes_test = train_test_split(texts, classes, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)


    clf = LinearSVC()
    clf.fit(X_train, classes_train)
    classes_pred = clf.predict(X_test)
    # print(classification_report(classes_test, classes_pred))

    question_vector = vectorizer.transform([lemmatize(question)])
    predicted_class = clf.predict(question_vector)

    return predicted_class