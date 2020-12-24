#!/usr/bin/env python
# coding: utf-8

import matchzoo as mz
print(mz.__version__)
print(mz.__path__)
import pickle
import sys
name,start,end = sys.argv
start, end = int(start),int(end)
print("Start",start,"End",end)
task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print(task)

BATCH_SIZE = 64
train_raw = mz.load_data_pack('amazon500Kproducts_train.v2.datapack')
test_raw = mz.load_data_pack('amazon500Kproducts_test.v2.datapack')

for model_class in mz.models.list_available()[start:end]: 
    model_name = str(model_class).split(".")[2]
    if model_name == "hyperboloid": pass
    print(model_class)
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
        model, preprocessor, data_generator_builder, embedding_matrix = mz.auto.prepare(
            task=task,
            model_class=model_class,
            data_pack=train_raw,
            )
    train_processed = preprocessor.transform(train_raw, verbose=0)
    test_processed = preprocessor.transform(test_raw, verbose=0)
    train_gen = data_generator_builder.build(train_processed)
    valid_x, valid_y = test_processed.unpack()
    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=BATCH_SIZE)
    history = model.fit_generator(train_gen, epochs=10, callbacks=[evaluate], workers=6, use_multiprocessing=True)
    pickle.dump(history.history, open("history_"+str(model_name)+".pkl","wb"))
    try:
        model.save("model_"+str(model_name)+".model")
    except:
        model._backend.save("model_"+str(model_name)+".model")
    print()
    del(model,preprocessor,data_generator_builder,embedding_matrix,train_processed,test_processed,train_gen,valid_x,valid_y,evaluate,history)




