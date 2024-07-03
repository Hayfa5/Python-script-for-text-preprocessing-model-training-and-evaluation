import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from collections import Counter

# Téléchargement des ressources nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Chargement des données
df = pd.read_csv('train.csv')

# Fonctions de prétraitement du texte
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(tokens)
    return text

# Appliquer le prétraitement du texte
df['clean_text'] = df['text'].apply(preprocess_text)


X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['category'], test_size=0.2, random_state=42)

# Vectorisation des données textuelles
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors = tfidf_vectorizer.fit_transform(X_train)
X_test_vectors = tfidf_vectorizer.transform(X_test)

# Encodage 
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Modèle Régression Logistique
lr_model = LogisticRegression(solver='liblinear', C=1, penalty='l2', multi_class='auto')
lr_model.fit(X_train_vectors, y_train)
y_pred_lr = lr_model.predict(X_test_vectors)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_report = classification_report(y_test, y_pred_lr)

# Modèle SVM
svm_model = SVC(kernel='linear', C=1, probability=True)
svm_model.fit(X_train_vectors, y_train)
y_pred_svm = svm_model.predict(X_test_vectors)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm)

# Modèle Arbre de Décision
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
dt_model.fit(X_train_vectors, y_train)
y_pred_dt = dt_model.predict(X_test_vectors)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_report = classification_report(y_test, y_pred_dt)

# Modèle Réseaux de Neurones
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_vectors.shape[1],)),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_vectors.toarray(), y_train_encoded, epochs=10, verbose=0)
nn_loss, nn_accuracy = nn_model.evaluate(X_test_vectors.toarray(), y_test_encoded)
nn_predictions = nn_model.predict(X_test_vectors.toarray())
nn_classes = nn_predictions.argmax(axis=-1)
nn_report = classification_report(y_test_encoded, nn_classes)

# Modèle RNN (Réseau de Neurones Récurrents) avec une couche LSTM
rnn_model = Sequential([
    Embedding(input_dim=X_train_vectors.shape[1], output_dim=100, input_length=X_train_vectors.shape[1]),
    LSTM(64),
    Dense(len(label_encoder.classes_), activation='softmax')
])
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_vectors.toarray(), y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test_vectors.toarray(), y_test_encoded))
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test_vectors.toarray(), y_test_encoded)
rnn_predictions = rnn_model.predict(X_test_vectors.toarray())
rnn_classes = rnn_predictions.argmax(axis=-1)
rnn_report = classification_report(y_test_encoded, rnn_classes)


# Prétraitement des données de test
# Créer une instance de LabelEncoder
label_encoder = LabelEncoder()

# Encoder les étiquettes en nombres
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train_bert = [tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True) for text in X_train]
X_test_bert = [tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True) for text in X_test]

#  modèle BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(np.unique(y_train)))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for inputs, label in zip(X_train_bert, y_train_encoded):
    inputs_tensor = torch.tensor(inputs).unsqueeze(0)
    label_tensor = torch.tensor(label).long()  # Convertir en type Long
    outputs = model(inputs_tensor, labels=label_tensor)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Évaluation
y_pred_bert = []
for inputs in X_test_bert:
    inputs_tensor = torch.tensor(inputs).unsqueeze(0)
    outputs = model(inputs_tensor)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    y_pred_bert.append(predicted_class)

bert_accuracy = accuracy_score(y_test_encoded, y_pred_bert)
bert_report = classification_report(y_test_encoded, y_pred_bert, zero_division=1)

# Affichage des résultats
print("Accuracy Score (Logistic Regression):", lr_accuracy)
print("Classification Report (Logistic Regression):\n", lr_report)
print("Accuracy Score (SVM):", svm_accuracy)
print("Classification Report (SVM):\n", svm_report)
print("Accuracy Score (Decision Tree):", dt_accuracy)
print("Classification Report (Decision Tree):\n", dt_report)
print("Accuracy Score (Neural Network):", nn_accuracy)
print("Classification Report (Neural Network):\n", nn_report)
print("Accuracy Score (RNN):", rnn_accuracy)
print("Classification Report (RNN):\n", rnn_report)
print("Accuracy Score (bert):", bert_accuracy)
print("Classification Report (bert):\n", bert_report)

# Fonction de prédiction par vote majoritaire
def predict_majority_class(input_texts):
    majority_predictions = []
    for text in input_texts:
        cleaned_text = preprocess_text(text)
        input_vector = tfidf_vectorizer.transform([cleaned_text])
        lr_prediction = lr_model.predict(input_vector)[0]
        svm_prediction = svm_model.predict(input_vector)[0]
        dt_prediction = dt_model.predict(input_vector)[0]
        nn_prediction = label_encoder.inverse_transform([np.argmax(nn_model.predict(input_vector.toarray()))])[0]

        model_predictions = [lr_prediction, svm_prediction, dt_prediction, nn_prediction]
        predictions_counter = Counter(model_predictions)
        majority_class = predictions_counter.most_common(1)[0][0]

        if majority_class == svm_prediction:
            majority_predictions.append(svm_prediction)
        else:
            majority_predictions.append(majority_class)

    return majority_predictions

# Utiliser la fonction predict_majority_class() avec X_test.values pour obtenir une liste de prédictions majoritaires
majority_predictions = predict_majority_class(X_test.values)

# Calculer les performances et les rapports pour les prédictions majoritaires
majority_accuracy = accuracy_score(y_test, majority_predictions)
majority_report = classification_report(y_test, majority_predictions)

# Afficher les résultats
print("Accuracy Score (Majority Voting):", majority_accuracy)
print("Classification Report (Majority Voting):\n", majority_report)
