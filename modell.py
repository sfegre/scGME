import tensorflow as tf
#import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

class Autoencoder(object):
    def __init__(self, n_input, n_hidden1,n_hidden2,n_hidden3, transfer_function=tf.nn.softplus, optimizer = tf.optimizers.Adam()):
        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.transfer = transfer_function

        network_weights = self._initialize_weights()  #权重初始化
        self.weights = network_weights

        # model

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.n_input])   #tf.compat.v1.disable_v2_behavior;tf.compat.v1.placeholder
        #为将始终输入的张量插入占位符
        self.hidden1 = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.hidden2 = self.transfer(tf.add(tf.matmul(self.hidden1, self.weights['w2']), self.weights['b2']))
        self.hidden3 = self.transfer(tf.add(tf.matmul(self.hidden2, self.weights['w3']), self.weights['b3']))
        self.reconstruction = tf.add(tf.matmul(self.hidden3, self.weights['w4']), self.weights['b4'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        #tf.reduce_sum 张量和 ；tf.pow 张量平方 ；tf.subtract 张量减法
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.cost)
        #寻找全局最优点的优化算法，引入了二次方梯度校正。
        self.loss = 0.5 * tf.square(self.reconstruction - self.x)
        #平方

        init = tf.compat.v1.global_variables_initializer()
        #有tf.Variable的环境下，必须要使用global_variables_initializer的场合
        #因为tf中建立的变量是没有初始化的，也就是在debug时还不是一个tensor量，而是一个Variable变量类型
        self.sess = tf.compat.v1.Session()
        #Session 是 Tensorflow 为了控制,和输出文件的执行的语句，运行 session.run()获得你要得知的运算结果, 或者是你所要运算的部分
        self.sess.run(init)
        #计算与fetch值有关的部分


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.compat.v1.get_variable("w1", shape=[self.n_input, self.n_hidden1],
            initializer=tf.keras.initializers.glorot_normal())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden1], dtype=tf.float32))
        all_weights['w2'] = tf.compat.v1.get_variable("w2", shape=[self.n_hidden1, self.n_hidden2],
            initializer=tf.keras.initializers.glorot_normal())
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_hidden2], dtype=tf.float32))
        all_weights['w3'] = tf.compat.v1.get_variable("w3", shape=[self.n_hidden2, self.n_hidden3],
            initializer=tf.keras.initializers.glorot_normal())
        all_weights['b3'] = tf.Variable(tf.zeros([self.n_hidden3], dtype=tf.float32))
        all_weights['w4'] = tf.Variable(tf.zeros([self.n_hidden3, self.n_input], dtype=tf.float32))
        all_weights['b4'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.loss), feed_dict={self.x: X})    #optimizer
        return cost
    #替换图中的某个tensor的值，设置graph的输入值

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden3, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'],self.weights['w2'],self.weights['w3'],self.weights['w4'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'],self.weights['b2'],self.weights['b3'],self.weights['b4'])

