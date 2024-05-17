import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
np.random.seed(42)

# 하이퍼파라미터 설정
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# 데이터 불러오기
testing_set = pd.read_csv('emnist-bymerge-test.csv')
training_set = pd.read_csv('emnist-bymerge-train.csv')

# 훈련 데이터와 레이블 추출
train_y = np.array(training_set.iloc[:, 0].values)
train_x = np.array(training_set.iloc[:, 1:].values)

# 테스트 데이터와 레이블 추출
test_y = np.array(testing_set.iloc[:, 0].values)
test_x = np.array(testing_set.iloc[:, 1:].values)

# 데이터 정규화
train_images = train_x / 255.0
test_images = test_x / 255.0

# 이미지 크기 재조정 (데이터 차원 변경)
images_height = 28
images_width = 28
train_x = train_x.reshape(train_x.shape[0], images_height, images_width, 1)
test_x = test_x.reshape(test_x.shape[0], images_height, images_width, 1)

# 클래스 수 및 원-핫 인코딩
number_of_classes = 47
train_y = tf.keras.utils.to_categorical(train_y, number_of_classes)
test_y = tf.keras.utils.to_categorical(test_y, number_of_classes)


train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42)

# 모델 구성
model = tf.keras.Sequential([
    Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1), padding='same'),
    AveragePooling2D(),
    Conv2D(16, kernel_size=(5, 5), activation='tanh'),
    AveragePooling2D(),
    Flatten(),
    Dense(120, activation='tanh'),
    Dense(84, activation='tanh'),
    Dense(number_of_classes, activation='softmax')
])

# 모델 컴파일
#LEARNING_RATE = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 콜백 설정
callbacks = [
    ModelCheckpoint('Best_points.h5', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=5, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, min_lr=0.0001)
]

# 모델 학습
#BATCH_SIZE = 32
#EPOCHS = 25
start_time = time.time()
history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_x, val_y), callbacks=callbacks)

training_time = time.time() - start_time

# 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, len(history.history['accuracy']) + 1), y=history.history['accuracy'], label='Train Accuracy')
sns.lineplot(x=range(1, len(history.history['val_accuracy']) + 1), y=history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

start_time = time.time()
predictions = model.predict(test_x)
total_inference_time = time.time() - start_time
average_inference_time = total_inference_time / len(test_x)

predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(test_y, axis=1)

correct_predictions = np.sum(predicted_classes == actual_classes)
accuracy = correct_predictions / len(predicted_classes)

print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {len(predicted_classes)}")
print(f"정확도: {accuracy:.2f}")
print(f"훈련시간 : {training_time:.2f} seconds.")
print(f"평균 추론 시간: {average_inference_time:.4f} seconds.") #각 샘플에 대한 추론 시간을 개별적으로 측정함으로써, 추론에 필요한 시간을 더 정확히 파악
