
import os
import sklearn
import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

BATCH_SIZE = 32
EPOCHS = 25
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 0.001
PATIENCE = 3
MIN_LEARNING_RATE = 0.0001
DECAY_FACTOR = 0.2
DENSE_LAYER_ACTIVATION = 'tanh'
FINAL_LAYER_ACTIVATION = 'softmax'
LOSS_FUNCTION = 'categorical_crossentropy'
METRICS = ['accuracy']

np.random.seed(42)

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

# 클래스 수 = 62
number_of_classes = 62

# 레이블 원-핫 인코딩
train_y = tf.keras.utils.to_categorical(train_y, 62)
test_y = tf.keras.utils.to_categorical(test_y, 62)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=VALIDATION_SPLIT, random_state=42)


base_model = ResNet50(weights=None, include_top=False, input_tensor=Input(shape=(28, 28, 1)))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(120, activation=DENSE_LAYER_ACTIVATION)(x)
x = Dense(84, activation=DENSE_LAYER_ACTIVATION)(x)
output = Dense(number_of_classes, activation=FINAL_LAYER_ACTIVATION)(x)

# 새로운 모델 정의
model = Model(inputs=base_model.input, outputs=output)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=LOSS_FUNCTION,
              metrics=METRICS)


# 콜백 설정
callbacks = [
    ModelCheckpoint('Best_points_resnet50.h5', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=PATIENCE, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', patience=PATIENCE, factor=DECAY_FACTOR, min_lr=MIN_LEARNING_RATE)
]
start_time = time.time()  # 학습 시작 시간

# 모델 학습
history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_x, val_y), callbacks=callbacks)

training_time = time.time() - start_time

# 학습 결과 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, len(history.history['accuracy']) + 1), y=history.history['accuracy'], label='Train Accuracy')
sns.lineplot(x=range(1, len(history.history['val_accuracy']) + 1), y=history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy with ResNet-50')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

start_time = time.time()  # 추론 시작 시간
predictions = model.predict(test_x)
end_time = time.time()  # 추론 종료 시간
total_inference_time = end_time - start_time
average_inference_time = total_inference_time / len(test_x)

# 테스트 데이터에 대한 예측 결과 처리
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(test_y, axis=1)

# 정확도 계산
correct_predictions = np.sum(predicted_classes == actual_classes)
accuracy = correct_predictions / len(predicted_classes)

# 결과 출력
print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {len(predicted_classes)}")
print(f"정확도: {accuracy:.2f}")
print(f"훈련시간 : {training_time:.2f} seconds.")
print(f"평균 추론 시간: {average_inference_time:.4f} seconds.") #각 샘플에 대한 추론 시간을 개별적으로 측정함으로써, 추론에 필요한 시간을 더 정확히 파악
