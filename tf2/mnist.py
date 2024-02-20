from  tensorflow import keras
import matplotlib.pyplot as plt
# import keras
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)
print(y_train.shape)

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=keras.activations.relu), #need激活函数itself而不是返回值,不用加括号执行
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10,keras.activations.softmax),
])
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['acc']
)
model.fit(x_train,y_train,epochs=2)
model.evaluate(x_test,y_test,verbose=1)
print(model.summary())
