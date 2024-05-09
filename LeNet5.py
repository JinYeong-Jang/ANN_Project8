
import os
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from sklearn.model_selection import train_test_split
np.random.seed(42)

testing_set = pd.read_csv('emnist-bymerge-test.csv')
training_set = pd.read_csv('emnist-bymerge-train.csv')

#print(training_set.shape)
#print(testing_set.shape)

# 훈련 데이터와 레이블 추출
train_y = np.array(training_set.iloc[:, 0].values)
train_x = np.array(training_set.iloc[:, 1:].values)

# 테스트 데이터와 레이블 추출
test_y = np.array(testing_set.iloc[:, 0].values)
test_x = np.array(testing_set.iloc[:, 1:].values)

#print(y1.shape)
#print(x1.shape)

#fig,axes = plt.subplots(10,4,figsize=(10,8))
#for i,ax in enumerate(axes.flat):
#    ax.imshow(x1[i].reshape([28,28]))

#plt.show()

# 데이터 정규화
train_images = train_x / 255.0
test_images = test_x / 255.0

# 이미지 크기 재조정 (데이터 차원 변경)
images_height = 28
images_width = 28
train_x = train_x.reshape(train_x.shape[0], images_height, images_width, 1)
test_x = test_x.reshape(test_x.shape[0], images_height, images_width, 1)

# 클래스 수 정의
number_of_classes = 62

# 레이블 원-핫 인코딩
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
MCP = ModelCheckpoint('Best_points.h5', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
ES = EarlyStopping(monitor='val_accuracy', min_delta=0, verbose=0, restore_best_weights=True, patience=3, mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)

# 모델 학습
history1 = model.fit(train_x, train_y, epochs=25, validation_data=(val_x, val_y), callbacks=[MCP, ES, RLP])

# 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x = range(1, len(history1.history['accuracy']) + 1), y = history1.history['accuracy'], label='Accuracy')
sns.lineplot(x = range(1, len(history1.history['val_accuracy']) + 1), y = history1.history['val_accuracy'], label='Val_Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

predictions = model.predict(test_x)

# 예측된 클래스 인덱스를 가져오기 (가장 높은 확률을 가진 클래스 인덱스)
predicted_classes = np.argmax(predictions, axis=1)

# 실제 테스트 레이블의 클래스 인덱스를 가져오기
actual_classes = np.argmax(test_y, axis=1)

# 예측 결과와 실제 결과 비교
correct_predictions = np.sum(predicted_classes == actual_classes)
total_predictions = len(predicted_classes)
accuracy = correct_predictions / total_predictions

print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {total_predictions}")
print(f"Accuracy: {accuracy:.2f}")