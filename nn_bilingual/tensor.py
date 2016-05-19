import tensorflow as tf
sess = tf.InteractiveSession()


filename_queue = tf.train.string_input_producer(["results_tensor2.csv"])
# filename_queue = tf.train.string_input_producer(["test_file.csv"])


reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
                   [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
                   [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
                   [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
                   [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
                   [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],[0.],[0.]]

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38, col39, col40, col41, col42, col43, col44, col45, col46, col47, col48, col49, col50, col51, col52, col53, col54, col55, col56, col57, col58, col59, col60, col61, col62, col63, col64, col65, col66, col67, col67, col68, col69 = tf.decode_csv(
    value, record_defaults=record_defaults)

features = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21, col22, col23, col24, col25, col26, col27, col28, col29, col30, col31, col32, col33, col34, col35, col36, col37, col38, col39, col40, col41, col42, col43, col44, col45, col46, col47, col48, col49, col50, col51, col52, col53, col54, col55, col56])

label_result = tf.pack([col57, col58, col59, col60, col61, col62, col63, col64, col65, col66, col67, col67, col68, col69])
print features
print label_result
x = tf.placeholder(tf.float32, shape=[1,56])
y_ = tf.placeholder(tf.float32, shape=[14])


W = tf.Variable(tf.zeros([56,14]))
b = tf.Variable(tf.zeros([14]))
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
for i in range(1):
  # Retrieve a single instance:
  example, label = sess.run([features, label_result])

  train_step.run(feed_dict={x: [example], y_: label})
  coord.request_stop()
  coord.join(threads)


# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

a = 10
accuracy_avg = 0.0
for i in range(a):
  # Retrieve a single instance:
  example, label = sess.run([features, label_result])
  print(tf.nn.top_k(example, k=1, sorted=True, name=None))

#   accuracy_avg +=accuracy.eval(feed_dict={x:  [example], y_: label})
# accuracy_avg = accuracy_avg/a
# print(accuracy_avg)
