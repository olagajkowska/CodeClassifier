import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import os
import click


class CodeClassifier:

    """CodeClassifier recognizes the language of code, based on a provided file containing a code block. It can
    recognize a couple of languages: Python, C++, HTML, Java, JavaScript, C, Fortran, Go, Haskell, Julia, Kotlin,
    MATLAB, Mathematica, PHP, Perl, R, Ruby, Rust, Scala and Swift."""

    def __init__(self):
        self.data = self.__clean_data(pd.read_csv("data.csv"))
        self.__splitted_data = self.__split_data()
        self.__token_pattern = r"""([A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]"'`])"""

        if os.path.isfile("Model.sav"):
            self.__model = pickle.load(open("Model.sav", 'rb'))
        else:
            self.__vectorizer = pickle.loads(self.__train_vectorizer())
            self.__model = pickle.loads(self.__train_random_forest())
            pickle.dump(self.__model, open("Model.sav", 'wb'))

        self.accuracy = self.__model.score(self.__splitted_data['X_test'], self.__splitted_data['Y_test'])

    @staticmethod
    def __clean_data(data):
        """Cleans the data; drops NaN values, joins multiple code samples with the same project id.
        :param data: input DataFrame
        :return: cleaned_data: cleaned DataFrame.
        """

        data.dropna(inplace=True)

        # It reduces number of rows; accuracy of the model decreases to ~70%

        # data["language"] = data.groupby(["proj_id"])["language"].transform(lambda x: '\n'.join(x))
        # cleaned_data = data.drop_duplicates(subset=["proj_id"])
        return data

    def __split_data(self):
        """
        Splits input data into test and training sets.
        :return: Dictionary with splitted dataset.
        """

        X = self.data.file_body
        Y = self.data.language
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        return {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}

    def __train_vectorizer(self):
        """
        Splits code blocks into tokens and trains TfIdVectorizer.
        :return: Pickle of a trained model.
        """

        vectorizer = TfidfVectorizer(token_pattern=self.__token_pattern)
        vectorizer.fit(self.__splitted_data['X_train'])
        return pickle.dumps(vectorizer)

    def __train_random_forest(self):
        """ Trains a random forest model using previously vectorized data.
        :return: A binary file containing the random forest model (Pickle).
        """

        clf = RandomForestClassifier(n_jobs=4)
        pipe = Pipeline([('vectorizer', self.__vectorizer), ('clf', clf)])

        # This block takes a very long time so I put best_params output below

        # param_grid_RF = {"clf__n_estimators": [200, 300, 400],
        #                  "clf__criterion": ["gini", "entropy"],
        #                  "clf__min_samples_split": [2, 3],
        #                  "clf__max_features": ["sqrt", None, "log2"]
        #                  }

        # grid_search_RF = GridSearchCV(pipe, param_grid=param_grid_RF, cv=3)
        # grid_search_RF.fit(self.__splitted_data['X_train'], self.__splitted_data['Y_train'])
        # best_params = grid_search_RF.best_params_

        best_params = {
                'clf__criterion': 'gini',
                'clf__max_features': 'sqrt',
                'clf__min_samples_split': 3,
                'clf__n_estimators': 300
                }

        pipe.set_params(**best_params)
        pipe.fit(self.__splitted_data['X_train'], self.__splitted_data['Y_train'])

        return pickle.dumps(pipe)

    def predict(self, source_file):
        """
        Predicts the language of code sample, previously saved in a source file.
        :param source_file: File containing the sample of code.
        :return: Prediction.
        """
        with open(source_file, 'r') as file:
            code_block = file.read()
        return str(self.__model.predict([
            code_block])[0])


@click.command()
@click.option("--source_file", prompt="Source file", default="sample.txt", help="Path to the source file")
def run(source_file):
    clf = CodeClassifier()
    prediction = clf.predict(source_file)
    accuracy = clf.accuracy
    click.echo("I think this sample of code is written in %s" % prediction)
    click.echo("Accuracy of this prediction is equal to %s" % accuracy)


if __name__ == '__main__':

    run()





