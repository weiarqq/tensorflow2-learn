import tensorflow as tf
y_true = [[0., 0., 1.]], [[0.,1.0, 0.]]
y_pred = [[0.1, 0.4, 0.9]], [[0.4, 0.6, 0.4]]
# Using 'auto'/'sum_over_batch_size' reduction type.
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.AUTO)
res = bce(y_true, y_pred)
rea = tf.reduce_mean(res)
print(res.numpy())
print(rea.numpy())
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
res = bce(y_true, y_pred)
rea = tf.reduce_mean(res)
print(res.numpy())
print(rea.numpy())
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
res = bce(y_true, y_pred)
rea = tf.reduce_mean(res)
print(1, res.numpy())
print(1, rea.numpy())

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
res = bce(y_true, y_pred)
rea = tf.reduce_mean(res)
print(1, res.numpy())
print(1, rea.numpy())

# x=tf.constant(value=4.0)
# with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
#     tape.watch(x)
#     y1=2*x
#     y2=x*x+2
#     y3=x*x+2*x
# #一阶导数
#     dy1_x = tape.gradient(target=[y1, y2, y3], sources=x)
#     dy1_dx = tape.gradient(target=y1, sources=x)
#     dy2_dx = tape.gradient(target=y2, sources=x)
#     dy3_dx = tape.gradient(target=y3, sources=x)
# print('dy1_x', dy1_x.numpy())
# print("dy1_dx:", dy1_dx.numpy())
# print("dy2_dx:", dy2_dx.numpy())
# print("dy3_dx:", dy3_dx.numpy())