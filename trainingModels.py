import sklearn
import numpy as np
from imblearn.over_sampling import SMOTE
from keras.src.layers import Embedding, LSTM
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from keras.src.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences


def trainOnClfs(X_train, y_train, X_test, y_test):

    # Be sure training samples are shuffled.
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=42)
    # # Standardize features by removing the mean and scaling to unit variance.
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    '''Oversampling'''
    smote = SMOTE(sampling_strategy='auto')  # Choose strategy based on your needs
    X_train, y_train = smote.fit_resample(X_train, y_train)


    # Support vector classification.
    clf = sklearn.svm.SVC()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Accuracy of SVM : {:.2%}'.format(score))
    #calculate F1 score
    y_pred = clf.predict(X_test)
    print('F1 score of SVM : {:.2%}'.format(f1_score(y_test, y_pred, average='weighted')))


    # Random forest
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Accuracy of Random forest : {:.2%}'.format(score))
    #calculate F1 score
    y_pred = clf.predict(X_test)
    print('F1 score of Random forest : {:.2%}'.format(f1_score(y_test, y_pred, average='weighted')))

    # Decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Accuracy of Decision tree classifier : {:.2%}'.format(score))
    #calculate F1 score
    y_pred = clf.predict(X_test)
    print('F1 score of Decision tree classifier : {:.2%}'.format(f1_score(y_test, y_pred, average='weighted')))

    # KNN classifier
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Accuracy of KNN classifier : {:.2%}'.format(score))
    #calculate F1 score
    y_pred = clf.predict(X_test)
    print('F1 score of KNN classifier : {:.2%}'.format(f1_score(y_test, y_pred, average='weighted')))



def trainOnNN(X_1, y_1):

    # '''Oversampling'''
    # smote = SMOTE(sampling_strategy='auto')  # Choose strategy based on your needs
    # X_1, y_1 = smote.fit_resample(X_1, y_1)

    # Dense
    # Convert DataFrames to NumPy arrays
    X_np = X_1.values
    y_np = y_1.values.ravel()  # ravel() converts it to a 1D array

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_np)

    # One-hot encode the labels
    y_onehot = to_categorical(y_encoded)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_np, y_onehot, test_size=0.2, random_state=42)

    '''Standardization'''
    # Be sure training samples are shuffled.
    # X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=42)
    # # # Standardize features by removing the mean and scaling to unit variance.
    # scaler = StandardScaler(copy=False)
    # scaler.fit_transform(X_train)
    # scaler.transform(X_test)

    # '''Oversampling'''
    # smote = SMOTE(sampling_strategy='auto')  # Choose strategy based on your needs
    # X_train, y_train = smote.fit_resample(X_train, y_train)


    # Create the model
    model = Sequential()

    #input size
    inpSize = X_np.shape[1]

    # Input layer
    model.add(Dense(inpSize, input_shape=(X_np.shape[1],), activation='relu'))  # X_np.shape[1] gives the number of features
    inpSize = inpSize // 2

    while inpSize >= 128:
        model.add(Dropout(0.5))
        model.add(Dense(inpSize, activation='relu'))
        inpSize = inpSize // 2

    if inpSize >= 64:
        # Hidden layers
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        inpSize = inpSize // 2

    if inpSize >= 32:
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))

    # Output layer
    model.add(Dense(y_onehot.shape[1], activation='softmax'))  # Number of classes

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=130, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)


    # print(model.summary())
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy * 100}%")
    #f1 score of the sequential model
    # Get the model's predictions on the test set
    # y_pred = model.predict(X_test, verbose=0)
    #
    # # Convert predicted probabilities to class labels
    # y_pred_classes = np.argmax(y_pred, axis=1)
    #
    # # Convert one-hot encoded y_test to class labels
    # y_true = np.argmax(y_test, axis=1)
    #
    # # Calculate F1 score
    # f1 = f1_score(y_true, y_pred_classes,
    #               average='weighted')  # You can change the 'average' parameter to None, 'micro', 'macro', 'weighted' based on your specific needs
    #
    # print(f"F1 Score: {f1}")

    return accuracy

    # # Plot training & validation accuracy values
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    #
    # # Plot training & validation loss values
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    #
    # plt.tight_layout()
    # plt.show()



def trainOnTextLSTM(X_train_vectors, X_test_vectors, y_train, y_test):
    numOfClasses = y_train.nunique()
    # #LSTM model
    # #prepare the data for LSTM model
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_encoded = to_categorical(y_train_encoded)

    y_test_encoded = label_encoder.fit_transform(y_test)
    y_test_encoded = to_categorical(y_test_encoded)

    print(y_train_encoded.shape)
    #reshape the data
    # X_train_vectors = np.array(X_train_vectors)
    # X_test_vectors = X_test_vectors.reshape(X_test_vectors.shape[0], 1, X_test_vectors.shape[1])

    # LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128))  # Assuming vocab size is 5000 and embedding size is 128
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dense(numOfClasses, activation='softmax'))  # Number of classes is y.shape[1]

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    Xtrain = X_train_vectors.toarray()
    Xtest = X_test_vectors.toarray()
    #fit the model
    model.fit(Xtrain, y_train_encoded, epochs=10, batch_size=64, verbose=1)

    #evaluate the model
    score = model.evaluate(Xtest, y_test_encoded, verbose=1)
    print("Accuracy: ", score[1])

    #print model summary
    print(model.summary())


def trainOnTextCNN(X_train, X_test, y_train, y_test):
    # encoder = OneHotEncoder(sparse=False)
    # y_train_encoded = encoder.fit_transform(y_train.values.ravel().reshape(-1, 1))
    # y_test_encoded = encoder.transform(y_test.values.ravel().reshape(-1, 1))
    #
    # # Tokenization and Padding
    # tokenizer = Tokenizer(num_words=5000)
    # tokenizer.fit_on_texts(X_train)
    # X_train_seq = tokenizer.texts_to_sequences(X_train)
    # X_test_seq = tokenizer.texts_to_sequences(X_test)
    #
    # X_train_pad = pad_sequences(X_train_seq, maxlen=100)
    # X_test_pad = pad_sequences(X_test_seq, maxlen=100)

    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=50, input_length=100))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(units=12, activation='softmax', kernel_regularizer='l2'))  # L2 regularization

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model, avoid overfitting
    # model.fit(X_train_pad, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test_encoded))
    # model.fit(X_train_pad, y_train_encoded, epochs=20, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    # loss, accuracy = model.evaluate(X_test_pad, y_test_encoded)
    # print(f"Lyrics Test Accuracy: {accuracy}")

    return model
