import tensorflow as tf
import ker_rnn as K
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

print('model10_noOM_att.py')

# todo 参数个数

#  统计参数 4587720 (Black)  xydir_attention
#  统计参数 4587720 (White)  all_attention

#  MRR最佳          (Black) xydir_attention
#  MRR最佳          (Black) all_attention

# todo 参数配置

# #  序列长度
# self.query_length = 15
# self.code_length = 200
# #  迭代次数
# self.num_epochs = 50
# #  批次大小
# self.batch_size = 256
# #  评测的批次
# self.eval_batch = 256
#
# #  隐层大小
# self.hidden_size = 256
# #  隐藏维数
# self.output_size = 0
# #  保存率
# self.keep_prob = 0.5
# #  滤波器 [1, 2, 3, 5, 7, 9]
# self.filter_sizes = 0
# #  滤波器个数
# self.n_filters = 0
# #  循环器个数
# self.n_layers = 2
#  中间维数
# self.layer_size = 200
#  交互方式 ['matmul','cosine','linear','tensor']
# self.interact= 'tensor'
#  池化前K
# self.kmax_pool= 0
#
# #  词嵌入矩阵
# self.embeddings = np.array(embeddings).astype(np.float32)
# self.embedding_size = 300
#
# #  优化器的选择
# self.optimizer = 'adam'
# #  切割率
# self.clip_value = 5
# #  正则化
# self.l2_lambda = 0.02
#
# #  学习率
# self.learning_rate = 0.002
# #  间距值
# self.margin = 0.5
#
# self.save_path = './'
#
# self.best_path = './'


class SiameseCSNN(object):
    def __init__(self, config):
        #  序列长度
        self.query_len = config.query_length
        self.code_len = config.code_length
        #  隐藏个数
        self.hidden_size = config.hidden_size
        #  中间维数
        self.output_size = config.output_size
        #  学习率
        self.learning_rate = config.learning_rate
        #  优化器
        self.optimizer = config.optimizer
        #  正则化
        self.l2_lambda = config.l2_lambda
        #  切割率
        self.clip_value = config.clip_value
        #  词向量矩阵
        self.embeddings = config.embeddings
        #  滤波器大小
        self.filter_sizes = config.filter_sizes
        #  滤波器个数
        self.n_filters = config.n_filters
        #  循环器层数
        self.n_layers = config.n_layers
        #  中间维数
        self.layer_size = config.layer_size
        #  交互方式
        self.interact = config.interact
        #  池化前K
        self.kmax_pool = config.kmax_pool
        #  间隔阈值
        self.margin = config.margin
        # self-attention惩罚项参数
        self.p_coef =  config.p_coef

        self.embedding_size = config.embedding_size

        self.placeholder_init()

        # 正负样本距离
        self.q_cpos_cosine, self.q_cneg_cosine = self.build(self.embeddings)
        # 损失和精确度
        # self.total_loss = self.add_loss_op(self.q_cpos_cosine, self.q_cneg_cosine, self.l2_lambda, self.p_coef)
        self.total_loss = self.add_loss_op(self.q_cpos_cosine, self.q_cneg_cosine, self.l2_lambda)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)

    def placeholder_init(self):

        self.code_pos = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')
        self.query = tf.placeholder(tf.int32, [None, self.query_len], name='query_point')
        self.code_neg = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size, self.list_size = tf.shape(self.code_pos)[0], tf.shape(self.code_neg)[1]
        # #on-lstm初始状态
        # self.h0 = tf.zeros([self.batch_size, self.hidden_size*2])
        # self.h0 = tf.stack([self.h0, self.h0])

    def max_pooling(self, lstm_out):  # (?,512,200)  #(?,15,512)
        #  512, 200   15,512
        height, width = lstm_out.get_shape().as_list()[1], lstm_out.get_shape().as_list()[2]

        # (?,#512,#200,1)  (1,#1,#200,1)
        lstm_out = tf.expand_dims(lstm_out, -1)
        # (?,512,1,1)
        output = tf.nn.max_pool(lstm_out, ksize=[1, 1, width, 1], strides=[1, 1, 1, 1], padding='VALID')

        # (?,512)
        max_pool = tf.reshape(output, [-1, height])

        return max_pool

    #  查询在代码上的注意力
    def atten_all_wrapper(self, input_q, input_c,scope):

        # (?,15,512) (?,200,512)
        with tf.device('/gpu:0'):

            Wq = tf.get_variable('Wc'+scope,shape=[300,self.layer_size],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))

            Wc = tf.get_variable('Wq'+scope,shape=[300, self.layer_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

            V= tf.get_variable('V'+scope,shape=[self.layer_size,1],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

            #  (?,15,512)  * (512,300)  = (?,15,300)
            # print('input_q')
            # print(input_q)
            # print('Wq')
            # print(Wq)

            fq = tf.einsum('abc,cd->abd', input_q, Wq)


            #  (?,200,512) * (512,300)  = (?,200,300)
            fc = tf.einsum('abc,cd->abd', input_c, Wc)

            re_fq = tf.tile(tf.expand_dims(fq,2),[1,1,self.code_len,1])    #  (?,15,1,300)---(?,15,200,300)

            re_fc = tf.tile(tf.expand_dims(fc,1),[1,self.query_len,1,1])   # (?,1,200,300)---(?,15,200,300)


            # 注意力矩阵（?,15,200,300) *(300,1) =（?,15,200)
            M = tf.squeeze(tf.einsum('abcd,df->abcf',tf.tanh(tf.add(re_fq,re_fc)),V),[3])

            #  查询在代码上注意力 (?,15,200)
            delta_q = tf.nn.softmax(M,dim=1)

            #  代码在查询上注意力 (?,15,200)
            delta_c = tf.nn.softmax(M,dim=-1)

            # 维度 (?,512,200)      (?,15,512) * (?,15,200)
            output_q = tf.einsum('abc,abd->acd',input_q, delta_q)
            # 维度 (?,15,512)       (?,15,200) * (?,200,512)
            output_c = tf.einsum('abc,acd->abd',delta_c,input_c)

            # 取平均
            output_q = self.max_pooling(output_q)

            output_c = self.max_pooling(tf.transpose(output_c,[0,2,1]))


        return output_q, output_c


    def onlstmm(self,  x, dropout, scope, hidden_size):
        """
               LSTM编码层
               """
        with tf.device('/gpu:0'):
            input_x = tf.transpose(x, [1, 0, 2])
            input_x = tf.unstack(input_x)

            with tf.variable_scope("cell" + scope):
                lstm_cell = K.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
                # self.wtfuck = lstm_cell.wt
                # self.fifuck = lstm_cell.fi
                # self.tifuck = lstm_cell.ti

                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout)

                multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.n_layers, state_is_tuple=True)

            with tf.variable_scope("decode" + scope):
                # lrt = K.BasicLSTMCelllrt(hidden_size, forget_bias=1.0, state_is_tuple=True)
                # _, _,  self.wt, self.fi, self.ti = K.static_rnn(lrt, input_x, dtype=tf.float32)

                output, _ = tf.contrib.rnn.static_rnn(multi_cell, input_x, dtype=tf.float32)

            output = tf.stack(output)
            output = tf.transpose(output, [1, 0, 2])

        return output

    def build(self, embeddings):
        self.Embedding = tf.Variable(tf.to_float(embeddings), trainable=False, name='Embedding')

        # 维度(?, 200, 300)
        c_pos_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_pos), self.dropout_keep_prob)

        # 维度(?, 15, 300)
        q_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.query), self.dropout_keep_prob)

        # 维度(?, 200, 300)
        c_neg_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_neg), self.dropout_keep_prob)


        # --------------------------------直接循环分开-----------------------------
        # todo 2种不同的循环方式

        # todo 1: 单向LSTM编码

        # with tf.variable_scope('onlstm') as scope0:
        #
        #     # LSTM (?, 200, 512)
        #     self.sc = 0
        #     lstm_c_pos = self.onlstmm(c_pos_embed, self.dropout_keep_prob, "code", self.hidden_size*2)
        #     # lstm_c_pos = self.call_onlstm(c_pos_embed)
        #
        #     # LSTM (?, 15, 512)
        #     self.sc = 1
        #     lstm_q = self.onlstmm(q_embed, self.dropout_keep_prob, "query", self.hidden_size*2)
        #     # lstm_q = self.call_onlstm(q_embed)
        #
        #     scope0.reuse_variables()
        #
        #     # LSTM (?, 200, 512)
        #     self.sc = 0
        #     lstm_c_neg = self.onlstmm(c_neg_embed, self.dropout_keep_prob, "code", self.hidden_size*2)
        #     # lstm_c_neg = self.call_onlstm(c_neg_embed)
        #
        # print('lstm_c_pos')
        # print(lstm_c_pos)
        # print('lstm_q')
        # print(lstm_q)
        # print('lstm_c_neg')
        # print(lstm_c_neg)

        #查询在代码上的注意力
        with tf.variable_scope('attention') as scope1:

            # 维度 (?,512,200)  (?,512,15)
            q_pos_atted, c_pos_atted = self.atten_all_wrapper(q_embed, c_pos_embed,'att')


            scope1.reuse_variables()  # 复制参数
            # 维度 (?,512,200)  (?,512,15)
            q_neg_atted, c_neg_atted = self.atten_all_wrapper(q_embed, c_neg_embed,'att')


        # todo 2：COS式

        q_pos_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(q_pos_atted, dim=1),
                                                           tf.nn.l2_normalize(c_pos_atted, dim=1)), axis=1)

        q_neg_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(q_neg_atted, dim=1),
                                                           tf.nn.l2_normalize(c_neg_atted, dim=1)), axis=1)

        return q_pos_cosine, q_neg_cosine


    def margin_loss(self, pos_sim, neg_sim):
        original_loss = self.margin - pos_sim + neg_sim
        l = tf.maximum(tf.zeros_like(original_loss), original_loss)
        loss = tf.reduce_sum(l)
        return loss, l

    def add_loss_op(self, p_sim, n_sim, l2_lambda=0.0001):
        """
        损失节点
        """
        self.loss, l = self.margin_loss(p_sim, n_sim)
        print('loss')
        # print(loss)

        tv = tf.trainable_variables()
        self.l2_loss = l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
        print('l2loss')
        # print(l2_loss)

        pairwise_loss = self.loss + self.l2_loss
        # pairwise_loss = loss + l2_loss
        tf.summary.scalar('pairwise_loss', pairwise_loss)
        self.summary_op = tf.summary.merge_all()
        return pairwise_loss

    def add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads_vars = opt.compute_gradients(loss)  # len(grads_vars) 12

            capped_grads_vars = [[tf.clip_by_value(g, -0.5, 0.5), v] for g, v in grads_vars]  # 梯度进行截断（更新）
            train_op = opt.apply_gradients(capped_grads_vars, self.global_step)  #
            # train_op = opt.minimize(loss, self.global_step)
            return train_op

