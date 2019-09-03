from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import sys

print("Loading", sys.argv[1])
data = np.load(sys.argv[1]+'_train_data.npy')
labels = np.load(sys.argv[1]+'_train_labels.npy')
val_data = np.load(sys.argv[1]+'_dev_data.npy')
val_labels = np.load(sys.argv[1]+'_dev_labels.npy')
test_data = np.load(sys.argv[1]+'_test_data.npy')
test_labels = np.load(sys.argv[1]+'_test_labels.npy')

LIMIT = 3031
data = data[:LIMIT,:]
labels = labels[:LIMIT,:]

test_accs = []
dev_accs = []
with open("exp_classification.log",'a') as log:
    for exp_i in range(5):
        input = Input(shape=(data.shape[1],))
        output = Dense(2, activation='softmax')(input)
        model = Model(inputs=input, outputs=output)

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        hist = model.fit(data, labels, batch_size=32, verbose=1, epochs=50, validation_data=(val_data, val_labels))
        dev_accs.append(hist.history['val_acc'][-1])
        test_accs.append(model.evaluate(test_data, test_labels, batch_size=32)[1])

    log.write("%s\tlen:%d\tdev_acc:%.4f\ttest_acc:%.4f\tbase:%.4f\n" % (sys.argv[1], len(labels), np.mean(dev_accs), np.mean(test_accs), max(sum(labels)/sum(sum(labels)))))
