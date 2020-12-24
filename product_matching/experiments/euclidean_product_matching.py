#!/usr/bin/env python
# coding: utf-8

import matchzoo as mz
print(mz.__version__)
print(mz.__path__)

model_name = "intersection"
BOX_DIM = 2*50
MAX_LEN_LEFT = 28
MAX_LEN_RIGHT = 100
EMB_DIM = 100
BATCH_SIZE = 10
NEG_SIZE = 4
DROPOUT = 0.4
task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=NEG_SIZE))
task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(task)

train_raw = mz.load_data_pack('amazon500Kproducts_train.v2.words.datapack')
test_raw = mz.load_data_pack('amazon500Kproducts_test.v2.words.datapack')

preprocessor = mz.preprocessors.DSSMPreprocessor()
preprocessor.fit(train_raw)
train_processed = preprocessor.transform(train_raw)
test_processed = preprocessor.transform(test_raw)

from tensorflow.python.keras import backend as K
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = mz.models.EuclideanIntersection()
    model.params['input_shapes'] = preprocessor.context['input_shapes']
    model.params['vocab_size'] = preprocessor.context['vocab_size']
    model.params['emb_dim'] = EMB_DIM
    model.params['box_dim'] = BOX_DIM
    model.params['dropout'] = DROPOUT
    model.params['task'] = task
    print(model.params)
    print("Building and Compiling the Model")
    model.build()
    model.compile()

print("Training the Model")
train_generator = mz.PairDataGenerator(train_processed, num_dup=1, num_neg=NEG_SIZE, batch_size=BATCH_SIZE, shuffle=True)
valid_x, valid_y = test_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=BATCH_SIZE)
history = model.fit_generator(train_generator, epochs=10, callbacks=[evaluate], workers=6, use_multiprocessing=True)

import pickle
pickle.dump(history.history, open("history_"+model_name.strip()+".pkl","wb"))
try:
    model.save("model_"+model_name.strip()+".model")
except:
    model._backend.save("model_"+model_name.strip()+".model")



