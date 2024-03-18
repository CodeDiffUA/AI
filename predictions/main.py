from chapter_prediction import predict_chapter
from theme_prediction import predict_theme

def predict_both(question):
    chapter = predict_chapter(question)
    theme = predict_theme(question)
    return chapter, theme


if __name__ == '__main__':
    print(predict_both("правильно наголошеним є слово"))