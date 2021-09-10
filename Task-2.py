import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def features(word_list):
    return dict([(word, True) for word in word_list])

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return(sentiment_dict['pos'])

if __name__=='__main__':
   positive_fileids=movie_reviews.fileids('pos')
   negative_fileids=movie_reviews.fileids('neg')
   
features_positive=[(features(movie_reviews.words(fileids=[f])),'p') for f in positive_fileids]
features_negative=[(features(movie_reviews.words(fileids=[f])),'n') for f in negative_fileids]

threshold_factor = 0.8
threshold_positive=int(threshold_factor*len(features_positive))
threshold_negative=int(threshold_factor*len(features_negative))

features_train=features_positive[:threshold_positive]+features_negative[:threshold_negative]
features_val=features_positive[threshold_positive:]+features_negative[threshold_negative:]

classifier=NaiveBayesClassifier.train(features_train)

input_reviews=[
    "The most amazing food ever! And also the staff is so nice to everyone. I highly recommend buying food from here. The best pizza ever",
    "My lunch libre was barely palatable because the meat was so salty that I could barely eat it inside the taco and not at all on its own",
    "Sleepy service, poor food quality, and when we asked why it was like this they stated that their kitchen was backed up, yet the restaurant was damn near empty",
    "I liked the food.",
    "I hate this Resturant so much. It has the worst staff ever!!",
    "It was nice resturant. I liked the food.",
    "The food was bad and the staff was not that friendly. Poor service",
    "Actually it's kinda good.",
    "Not so good restaurant"
]

print("\nSentiments: ")

count = temp = 0
sid = SentimentIntensityAnalyzer()
for review in input_reviews:
    count = count +1
    print("\nReviews:", review)
    probdist = classifier.prob_classify(features(review.split()))
    pred_sentiment = probdist.max()
    print("Sentiment: ", pred_sentiment)
    temp += sentiment_scores(review)
    print("Probability: ", round(probdist.prob(pred_sentiment), 4))

rating = round((temp/count)*10, 1)
print("\nOverall Restaurant Rating is: ", rating)


