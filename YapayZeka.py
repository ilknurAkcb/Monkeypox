import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# Veri setinin yüklenmesi ve özellik vektörlerinin oluşturulması
def load_and_preprocess_images(folder, model):
    images = []
    labels = []
    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Özellik vektörünü çıkarma
                features = model.predict(img_array)
                images.append(features.flatten())  # Özellik vektörünü düzleştirme
                labels.append(class_folder)
    return np.array(images), np.array(labels)

# Veri seti dizini
data_folder = 'C:/Users/LENOVO/OneDrive/Masaüstü/monkey/Fold1/Fold1/Fold1'

# VGG16 modelini yükleyip özellik vektörlerini çıkartma
base_model_vgg16 = VGG16(weights='imagenet', include_top=False)
model_vgg16 = Sequential()
model_vgg16.add(base_model_vgg16)
model_vgg16.add(Flatten())

# Eğitim veri setini yükleme ve özellik vektörlerini çıkartma
train_data_folder = os.path.join(data_folder, 'Train')
images_train, labels_train = load_and_preprocess_images(train_data_folder, model_vgg16)

# Test veri setini yükleme ve özellik vektörlerini çıkartma
test_data_folder = os.path.join(data_folder, 'Test')
images_test, labels_test = load_and_preprocess_images(test_data_folder, model_vgg16)

# Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
encoded_labels_train = label_encoder.fit_transform(labels_train)
encoded_labels_test = label_encoder.transform(labels_test)

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = images_train, images_test, encoded_labels_train, encoded_labels_test

# Özellik seçimi (SelectKBest)
kbest_selector = SelectKBest(score_func=f_classif, k=200)

# Eğitim verileri üzerinde SelectKBest'u uygulama
X_train_selected = kbest_selector.fit_transform(X_train, y_train)

# Test verileri üzerinde SelectKBest'u uygulama
X_test_selected = kbest_selector.transform(X_test)

# Veri standardizasyonu
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_selected)
X_test_std = scaler.transform(X_test_selected)

# Güçlü bir sinir ağı oluşturma
model_nn = Sequential()
model_nn.add(Dense(1024, activation='relu', input_shape=(200,)))
model_nn.add(Dropout(0.5))
model_nn.add(Dense(512, activation='relu'))
model_nn.add(Dropout(0.4))
model_nn.add(Dense(256, activation='relu'))
model_nn.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Modeli derleme
optimizer = Adam(learning_rate=0.0001)
model_nn.compile(loss='sparse_categorical_crossentropy',
                 optimizer=optimizer,
                 metrics=['accuracy'])

# Modeli daha uzun süre eğitme
history = model_nn.fit(X_train_std, y_train, epochs=300, batch_size=64, validation_split=0.2)

# Eğitim ve doğrulama loss'unu görselleştirme
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Test seti üzerinde modeli değerlendirme
accuracy_nn = model_nn.evaluate(X_test_std, y_test)[1]
print(f'Test Accuracy (Neural Network with Feature Selection): {accuracy_nn}')

# Sınıflandırma raporu
y_pred_nn = np.argmax(model_nn.predict(X_test_std), axis=-1)
print(classification_report(y_test, y_pred_nn, target_names=label_encoder.classes_))

# Confusion Matrix
cm_nn = confusion_matrix(y_test, y_pred_nn)

# Confusion Matrix'i görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Neural Network with Feature Selection)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# MLP modelini oluşturma ve eğitme
mlp_classifier = MLPClassifier(hidden_layer_sizes=(200,), max_iter=500, random_state=42)
mlp_classifier.fit(X_train_std, y_train)

# Test verileri üzerinde tahminleri alın
y_pred_mlp = mlp_classifier.predict(X_test_std)

# MLP ve CNN çıktılarını birleştirme
X_test_combined = np.concatenate((y_pred_mlp.reshape(-1, 1), y_pred_nn.reshape(-1, 1)), axis=1)

# Yeni bir MLP modeli oluşturma ve eğitme
ensemble_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
ensemble_model.fit(X_test_combined, y_test)

# Ensemble modelinin performansını değerlendirme
y_pred_ensemble = ensemble_model.predict(X_test_combined)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f'Ensemble Model Accuracy: {accuracy_ensemble}')

# Ensemble modelinin performansını değerlendirme
y_pred_ensemble = ensemble_model.predict(X_test_combined)

# Confusion Matrix (Ensemble Model)
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)

# Confusion Matrix'i görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix (Ensemble Model)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()