import pandas as pd
import nltk
nltk.download('stopwords')
# stopwords contain a list of irrelevant words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def main():
    dataset = pd.read_csv(
        "C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Restaurant_Reviews.tsv",
        delimiter='\t', quoting=3)
    cleaned_data = clean_data(dataset)
    # quoting 3 ignores the double quotes in the dataset
    create_bag_words(dataset,cleaned_data)

def clean_data(dataset):
    # cleaning the texts for the bag of relevant words
    print(dataset['Review'][0])

    # cleaned list with all the reviews
    corpus = []

    # do a cleaning process of all the reviews
    for i in range(0,1000):
        # ^a-zA-Z indicates that letters a-z or A-Z will not be removed
        # ' ' indicates to take into consideration the spaces
        #review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][0])

        # do the cleaning process for each row-review
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

        # convert to lower case
        review = review.lower()

        # separate each word in a list
        review = review.split()

        # do a stemming process for each word
        ps = PorterStemmer()

        # we add a set for speeding up the process of execution
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

        # join the elements of the list
        review = ' '.join(review)

        #print(review)

        corpus.append(review)

    #print(corpus)
    return corpus

def create_bag_words(dataset,cleaned_data):
    counter = 0
    # we are going to create a bag of words or sparse matrix to put the words in columns and then the words that appear several times
    # will be counted into these columns

    output = open("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\sparse_matrix_nlp.txt",'w')
    '''

    words-->       wow  love  terrific  disgusting
                    0    1     2           3        ..... n
                    ---------------------------------------
    reviews --> 0 |  0    0     0          0 ....         0
      |         1 |  0    0     0          0 ....         0
      |         2 |  0    0     1          0 ....         0
     \/         3 |  1    0     0          0 ....         0

    '''
    # filter the number of words or reducing sparcity with the max_features, limit the number of selected words
    cv = CountVectorizer(max_features = 1500)

    # create the sparse matrix which is the matrix of independent variables (words from comments)
    X = cv.fit_transform(cleaned_data).toarray()

    # visualize every result of the array
    # row contains the array of all the values

    for row in X:
        output.write(str(counter))
        output.write("\n")
        for value in row:
            output.write(str(value))
            output.write(str("  "))
        #output.write("\n")
        counter = counter + 1

    # take the independent variable as the result of the review to determine if it was good or bad
    # 0 bad review
    # 1 good review
    y = dataset.iloc[:,1].values

    # what's next is to train our model to teach the machine to determine if a review was good or bad based on our
    # matrix of features and output variable
    print(X)
    output.close()

    '''

        X -->       wow  love  terrific  disgusting                 y = positive (1) or negative (0) review
                        0    1     2           3        ..... n
                        ------------------------------------------------------------------------------------
        reviews --> 0 |  0    0     1          0 ....         0              1            
          |         1 |  0    0     0          1 ....         0              0
          |         2 |  0    0     1          0 ....         0              1
         \/         3 |  0    0     0          1 ....         0              0

        '''

    # ------------------------ Splitting the dataset into Training set and Test set ------------------------------------ #

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # ------------------------------------------ Feature scaling ------------------------------------------------------- #
    # THIS IS NOT GOING TO BE APPLIED BECAUSE WE ONLY HAVE 1 AND 0 IN THE SPARSE MATRIX
    #sc_X = StandardScaler()

    # we scale our training set variables to avoid domination of one big variable against the others and then we apply the changes
    # with fit_transform
    #X_train = sc_X.fit_transform(X_train)

    # we scale the variables in our test set but we do not fit in our test set because we already fit in our training set
    #X_test = sc_X.transform(X_test)

    # ------------------------------------- Fitting Classifier ---------------------------------------------- #

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # ----------------------------------- Predicting the Test Set Results -------------------------------------------- #

    y_pred = classifier.predict(X_test)
    print("Results test: \n", y_test)
    print("Results prediction: \n", y_pred)

    # ----------------------------------- Making the confusion matrix ----------------------------------------------- #

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix: \n", cm)



if __name__ == "__main__":
    main()