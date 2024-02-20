from tensorflow import keras
import  tensorflow_datasets as tfds

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder
#mimi=encoder.encode("ouyang")
#print(mimi)# [2102, 7381, 8032]
#print(encoder.decode(mimi))# ouyang

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_dataset.padded_batch(BATCH_SIZE)


model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 64),
    keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])
model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(1e-4),
    metrics=['acc']
)
model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    validation_steps=30
)
model.evaluate(test_dataset)

