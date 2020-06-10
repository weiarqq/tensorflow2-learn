import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
num_train_steps = 40000
end_learning_rate = 0
num_warmup_steps = 4000
init_lr = 5e-5


global_step = tf.placeholder(dtype=tf.int32, name='x')
learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

# Implements linear decay of the learning rate.
learning_rate = tf.compat.v1.train.polynomial_decay(
    learning_rate,
    global_step,
    num_train_steps,
    end_learning_rate=end_learning_rate,
    power=1.0,
    cycle=False)
# Implements linear warmup. I.e., if global_step < num_warmup_steps, the
# learning rate will be `global_step/num_warmup_steps * init_lr`.
if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

with tf.Session() as sess:
    x = []
    y = []
    for i in range(num_train_steps):
        x.append(i)
        lr = sess.run(learning_rate, feed_dict={global_step: i})
        y.append(lr)
        print(lr)

l1=plt.plot(x,y,label='type1')
plt.plot(x,y,)
plt.show()
