def read_files(files):
    labels = []
    features = []
    for ans, files in files.items():
        for file in files:
            wave, sr = librosa.load(file, mono=True)
            label = dense_to_one_hot(ans, 10)
            labels.append(label)
            mfcc = librosa.feature.mfcc(wave, sr)
            mfcc = np.pad(mfcc, ((0, 0), (0, 100 - len(mfcc[0]))), mode='constant', constant_values=0)
            features.append(np.array(mfcc))
    return np.array(features), np.array(labels)

def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value

class CNNConfig():
    # 网络结构
    filter_sizes = [2, 3, 4, 5]
    num_filters = 64
    hidden_dim = 256

    # 训练过程
    learning_rate = 0.001
    num_epochs =
    dropout_keep_prob = 0.5
    print_per_batch = 100       # 训练过程中,每100次batch迭代，打印训练信息
    save_tb_per_batch = 200

class ASRCNN(object):
    def __init__(self, config, width, height, num_classes):  # 20,100
        self.config = config
        # 训练过程
        # 输入的语音变成了一张特征图
        self.input_x = tf.placeholder(tf.float32, [None, width, height], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # input_x = tf.reshape(self.input_x, [-1, height, width])
        # 将图由width * height变成了height * width
        input_x = tf.transpose(self.input_x, [0, 2, 1])
        pooled_outputs = []
        # 也就是说有一系列不同size的filter
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print("conv-maxpool-%s" % filter_size)
                conv = tf.layers.conv1d(input_x, self.config.num_filters, filter_size, activation=tf.nn.relu)
                print(conv.shape)
                pooled = tf.reduce_max(conv, reduction_indices=[1])
                print(pooled.shape)
                pooled_outputs.append(pooled)
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)  # 64*4
        print(pooled_outputs.shape)
        pooled_reshape = tf.reshape(tf.concat(pooled_outputs, 1), [-1, num_filters_total])
        print(pooled_reshape.shape)

        fc = tf.layers.dense(pooled_reshape, self.config.hidden_dim, activation=tf.nn.relu, name='fc1')
        fc = tf.contrib.layers.dropout(fc, self.keep_prob)

        # 分类器
        self.logits = tf.layers.dense(fc, num_classes, name='fc2')
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")  # 预测类别
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)
        # 优化器
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # 准确率
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(argv=None):
        '''
        batch = mfcc_batch_generator()
        X, Y = next(batch)
        trainX, trainY = X, Y
        testX, testY = X, Y
        # overfit for now
        '''
        train_files, valid_files, test_files = load_files()
        train_features, train_labels = read_files(train_files)
        train_features = mean_normalize(train_features)
        print('read train files down')
        valid_features, valid_labels = read_files(valid_files)
        valid_features = mean_normalize(valid_features)
        print('read valid files down')
        test_features, test_labels = read_files(test_files)
        test_features = mean_normalize(test_features)
        print('read test files down')

        width = 20  # mfcc features
        height = 100  # (max) length of utterance
        classes = 10  # digits

        config = CNNConfig
        cnn = ASRCNN(config, width, height, classes)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join('cnn_model', 'model.ckpt')
        tensorboard_train_dir = 'tensorboard/train'
        tensorboard_valid_dir = 'tensorboard/valid'

        if not os.path.exists(tensorboard_train_dir):
            os.makedirs(tensorboard_train_dir)
        if not os.path.exists(tensorboard_valid_dir):
            os.makedirs(tensorboard_valid_dir)
        tf.summary.scalar("loss", cnn.loss)
        tf.summary.scalar("accuracy", cnn.acc)
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(tensorboard_train_dir)
        valid_writer = tf.summary.FileWriter(tensorboard_valid_dir)

        total_batch = 0
        for epoch in range(config.num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = batch_iter(train_features, train_labels)
            for x_batch, y_batch in batch_train:
                total_batch += 1
                feed_dict = feed_data(cnn, x_batch, y_batch, config.dropout_keep_prob)
                session.run(cnn.optim, feed_dict=feed_dict)
                if total_batch % config.print_per_batch == 0:
                    train_loss, train_accuracy = session.run([cnn.loss, cnn.acc], feed_dict=feed_dict)
                    valid_loss, valid_accuracy = session.run([cnn.loss, cnn.acc], feed_dict={cnn.input_x: valid_features,
                        cnn.input_y: valid_labels,
                        cnn.keep_prob: config.dropout_keep_prob})
                    print('Steps:' + str(total_batch))
                    print('train_loss:' + str(train_loss) +
                            ' train accuracy:' + str(train_accuracy) +
                            '\tvalid_loss:' + str(valid_loss) +
                            ' valid accuracy:' + str(valid_accuracy))
                if total_batch % config.save_tb_per_batch == 0:
                    train_s = session.run(merged_summary, feed_dict=feed_dict)
                    train_writer.add_summary(train_s, total_batch)
                    valid_s = session.run(merged_summary, feed_dict={cnn.input_x: valid_features, cnn.input_y: valid_labels,
                        cnn.keep_prob: config.dropout_keep_prob})
                    valid_writer.add_summary(valid_s, total_batch)

            saver.save(session, checkpoint_path, global_step=epoch)

            test_loss, test_accuracy = session.run([cnn.loss, cnn.acc],
                    feed_dict={cnn.input_x: test_features, cnn.input_y: test_labels,
                         cnn.keep_prob: config.dropout_keep_prob})
            print('test_loss:' + str(test_loss) + ' test accuracy:' + str(test_accuracy))

# 测试数据准备,读取文件并提取音频特征
def read_test_wave(path):
    files = os.listdir(path)
    feature = []
    features = []
    label = []
    for wav in files:
        # print(wav)
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])        
        wave, sr = librosa.load(path+wav, mono=True)
        label.append(ans)
        # print("真实lable: %d" % ans)
        mfcc = librosa.feature.mfcc(wave, sr)
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - len(mfcc[0]))), mode='constant', constant_values=0)
        feature.append(np.array(mfcc))   
    features = mean_normalize(np.array(feature))
    return features,label

def test(path):
    features, label = read_test_wave(path)
    print('loading ASRCNN model...')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('cnn_model/model.ckpt-999.meta')
        saver.restore(sess, tf.train.latest_checkpoint('cnn_model'))  
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        pred = graph.get_tensor_by_name("pred:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        for i in range(0, len(label)):
            feed_dict = {input_x: features[i].reshape(1,20,100), keep_prob: 1.0}
            test_output = sess.run(pred, feed_dict=feed_dict)

            print("="*15)
            print("真实lable: %d" % label[i])
            print("识别结果为:"+str(test_output[0]))
        print("Congratulation!")  

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train()
    else:
        test()
