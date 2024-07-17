# Description: This file contains the functions to test the model and predict the output.
def model_fit_pred(x, y):
    """
    fit and predict all models, print model name, accuracy, confusion matrix, classification report and return the best model
    :param x: x values
    :param y: y values
    :return:  best model
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Alias for Libraries
    g = GaussianNB()
    b = BernoulliNB()
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()

    models = [g, b, lr, knn, dt, rf, gb]
    model_names = ['GaussianNB', 'BernoulliNB', 'LogisticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier',
                   'GradientBoostingClassifier']
    accuracy = []
    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        print(model)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
        print('Classification Report:\n', classification_report(y_test, y_pred))
        print('-' * 40)
    print('Best Model:', model_names[accuracy.index(max(accuracy))])
    return models[accuracy.index(max(accuracy))]
