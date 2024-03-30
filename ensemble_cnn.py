import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.metrics import accuracy_score
import csv
import os
import time
from classify import *

def create_cnn_model(max_words, max_len, embedding_dim):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def perform_ensemble_cnn_alg(num_models=3):
    # Load preprocessed data
    preprocessed_data = pd.read_csv('./output/Preprocessed/preprocessed_data.csv')

    # Load glove results
    glove_results = pd.read_csv('./output/Glove_word_embedded/glove_results.csv')

    file_path = './output/Ensemble_cnn/ensemble_cnn_prediction.csv'

    timer()

    # Define target variable
    preprocessed_data['label'] = preprocessed_data['preprocessed_text'].apply(lambda text: 1 if 'bully' in text.lower() else 0)

    # Preprocess text data
    max_words = 10000
    max_len = 100

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(preprocessed_data['preprocessed_text'])
    sequences = tokenizer.texts_to_sequences(preprocessed_data['preprocessed_text'])
    X = pad_sequences(sequences, maxlen=max_len)

    # Define target variable
    y = preprocessed_data['label']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

    # Convert X_test to DataFrame to retrieve indices
    X_test_df = pd.DataFrame(X_test, columns=[f'col_{i}' for i in range(X_test.shape[1])])

    # Initialize and train multiple CNN models
    models = []
    for i in range(num_models):
        model = create_cnn_model(max_words, max_len, embedding_dim=100)
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
        models.append(model)

    # Make predictions on the test set using each model
    predictions = np.zeros((X_test.shape[0], num_models))

    for i, model in enumerate(models):
        predictions[:, i] = (model.predict(X_test) > 0.5).astype(int).flatten()

    # Use vectorized operations to check for bullying words
    bullying_words = glove_results['Word'].str.lower().tolist()
    similar_words = glove_results['Similar_Words'].str.split(',').explode().str.strip().tolist()

    # Combine predictions using majority voting
    ensemble_cnn_predictions = np.round(np.mean(predictions, axis=1)).astype(int)   
    
    # Ensemble_CNN_predictions to check for bullying and non-bullying words on preprocessed_text
    preprocessed_data['ensemble_cnn_predictions'] = preprocessed_data['preprocessed_text'].apply(
        lambda text: any(word in text.lower().split() for word in bullying_words) or any(similar_word in text.lower().split() for similar_word in similar_words)
    ).astype(int)

    # Save results to a new CSV file
    results_df = pd.DataFrame({
        'S.No': preprocessed_data.loc[X_test_df.index, 'sno'],
        'Preprocessed_Text': preprocessed_data.loc[X_test_df.index, 'preprocessed_text'],
        'EnsembleCNN_Prediction': preprocessed_data.loc[X_test_df.index, 'ensemble_cnn_predictions']
    })

    results_df.to_csv(file_path, index=False)

    print('\n')
    print(results_df)
    count_rows(file_path)
    num_rows = count_rows(file_path)
    print(f"\n\nThe total number of rows in {file_path} is: {num_rows-1}")

    print("\nIt predicted the twitter data as bully and non-bully using ensemble CNN algorithm \n\nThe output file is saved at:", file_path)

    

def count_rows(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = len(list(reader))
    return row_count

def timer():
    for progress in range(10, 101, 10):
        # Print the progress percentage
        print(f"Processing: {progress}%")

        # Simulate processing time using time.sleep()
        time.sleep(0.2)  # Adjust the sleep duration as needed
        

