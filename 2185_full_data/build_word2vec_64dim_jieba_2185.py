# -*- coding: UTF-8 -*-
import pickle
import numpy as np
import collections
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
'''
这个文件用来根据2185个专利内容建立一个word2vec的向量表，word2vec向量维度为64维
'''
output = open('2185_data_directory\content_one_hot_jieba_2185.pkl', 'rb')
one_hot = pickle.load(output)
output.close()
output = open('2185_data_directory\content_dict_jieba_2185.pkl', 'rb')
dicts = pickle.load(output)
output.close()
reversed_dictionary = dict(zip(dicts.values(), dicts.keys()))

all_content = list()
for i in range(len(one_hot)):
    all_content.extend(one_hot[i])
data_index = 0


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        if data_index == len(data):
            # print(data_index,len(data),span,len(data[:span]))
            # buffer[:] = data[:span]
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(all_content, batch_size=8, num_skips=2, skip_window=1)

for i in range(8):  # 取第一个字，后一个是标签，再取其前一个字当标签，
    print(batch[i], reversed_dictionary[batch[i]], '->', labels[i, 0], reversed_dictionary[labels[i, 0]])
words_size = len(dicts)
batch_size = 128
embedding_size = 64  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = np.int32(words_size / 2)  # Only pick dev samples in the head of the distribution.
print("valid_window", valid_window)
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 0-words_size/2,中的数取16个。不能重复。
num_sampled = 64  # Number of negative examples to sample.

tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Ops and variables pinned to the CPU because of missing GPU implementation
with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([words_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([words_size, embedding_size],
                                                  stddev=1.0 / tf.sqrt(np.float32(embedding_size))))

    nce_biases = tf.Variable(tf.zeros([words_size]))

# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                   labels=train_labels, inputs=embed,
                   num_sampled=num_sampled, num_classes=words_size))

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# Compute the cosine similarity between mini-batch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
print("________________________", similarity.shape)

# Begin training.
num_steps = 200001
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(all_content, batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 通过打印测试可以看到  embed的值在逐渐的被调节
        #        emv = sess.run(embed,feed_dict = {train_inputs: [37,18]})
        #        print("emv-------------------",emv[0])

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)

        if step % 10000 == 0:
            sim = similarity.eval(session=sess)
            # print(valid_size)
            for i in range(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                # print("valid_word",valid_word)#16
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]  # argsort函数返回的是数组值从小到大的索引值
                # print("nearest",nearest,top_k)
                log_str = 'Nearest to %s:' % valid_word

                for k in range(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s,%s' % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()
    vvvv = np.array(final_embeddings)
    output = open('2185_data_directory\word2vec_64dim_2185.pkl', 'wb')
    pickle.dump(vvvv, output)
    output.close()
    print(vvvv.shape)


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                     ha='right', va='bottom')
    plt.savefig(filename)


try:
    # pylint: disable=g-import-not-at-top
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 80  # 输出100个词
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reversed_dictionary[i] for i in range(plot_only)]
    # print(labels)
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
