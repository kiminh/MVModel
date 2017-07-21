import tensorflow as tf
import numpy as np
import os

from seq_rnn_model import SequenceRNNModel
import data_utils
import model_data
import csv
# data path parameter
tf.flags.DEFINE_string('data_path', '', 'file dir for saving features and labels')
tf.flags.DEFINE_string("save_seq_basicmvmodel_path", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/basic/seq_mvmodel.ckpt", "file path to save model")
tf.flags.DEFINE_string('seq_basicmvmodel_path', '/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/basic/seq_mvmodel.ckpt-100', 'trained mvmodel path')
tf.flags.DEFINE_string("save_seq_embeddingmvmodel_path", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/embedding/seq_mvmodel.ckpt", "file path to save model")
tf.flags.DEFINE_string('seq_embeddingmvmodel_path', '/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/embedding/seq_mvmodel.ckpt-70', 'trained mvmodel path')
tf.flags.DEFINE_string('test_acc_file', 'seq_acc.csv', 'test acc file')

# model parameter
tf.flags.DEFINE_boolean("use_embedding", True, "whether use embedding")
tf.flags.DEFINE_boolean("use_attention", True, "whether use attention")

tf.flags.DEFINE_integer("training_epoches", 100, "total train epoches")
tf.flags.DEFINE_integer("save_epoches", 5, "epoches can save")
tf.flags.DEFINE_integer("n_views", 12, "number of views for each model")
tf.flags.DEFINE_integer("n_input_fc", 4096, "size of input feature")
tf.flags.DEFINE_integer("decoder_embedding_size", 256, "decoder embedding size")
tf.flags.DEFINE_integer("n_classes", 10, "total number of classes to be classified")
tf.flags.DEFINE_integer("n_hidden", 128, "hidden of rnn cell")
tf.flags.DEFINE_float("keep_prob", 1.0, "kepp prob of rnn cell")
tf.flags.DEFINE_boolean("use_lstm", False, "use lstm or gru cell")

# attention parameter
tf.flags.DEFINE_integer("num_heads", 1, "Number of attention heads that read from attention_states")

# training parameter
tf.flags.DEFINE_boolean('train', True, 'train mode')
tf.flags.DEFINE_integer("batch_size", 32, "training batch size")
tf.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.flags.DEFINE_integer("n_max_keep_model", 20, "max number to save model")

FLAGS = tf.flags.FLAGS

def main(unused_argv):
    if FLAGS.train:
        train()
    else:
        test()

def train():
    data =  model_data.read_data(FLAGS.data_path)
    seq_rnn_model = SequenceRNNModel(FLAGS.n_input_fc, FLAGS.n_views, FLAGS.n_hidden, FLAGS.decoder_embedding_size, FLAGS.n_classes+1, FLAGS.n_hidden,
                                     learning_rate=FLAGS.learning_rate,
                                     keep_prob=FLAGS.keep_prob,
                                     batch_size=FLAGS.batch_size,
                                     is_training=True,
                                     use_lstm=FLAGS.use_lstm,
                                     use_attention=FLAGS.use_attention,
                                     use_embedding=FLAGS.use_embedding,
                                     num_heads=FLAGS.num_heads)
                                     #init_decoder_embedding=model_data.read_class_yes_embedding(FLAGS.data_path))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    if not os.path.exists(get_modelpath()):
        os.makedirs(get_modelpath())
    with tf.Session(config=config) as sess:
        seq_rnn_model.build_model()
        saver = tf.train.Saver(max_to_keep=FLAGS.n_max_keep_model)
        init = tf.global_variables_initializer()
        sess.run(init)

        epoch = 1
        while epoch <= FLAGS.training_epoches:
            batch = 1
            while batch * FLAGS.batch_size <= data.train.size():
                batch_encoder_inputs, batch_decoder_inputs = data.train.next_batch(FLAGS.batch_size)
                # target_labels = get_target_labels(batch_decoder_inputs)
                batch_encoder_inputs = batch_encoder_inputs.reshape((FLAGS.batch_size, FLAGS.n_views, FLAGS.n_input_fc))
                batch_encoder_inputs, batch_decoder_inputs, batch_target_weights = seq_rnn_model.get_batch(batch_encoder_inputs, batch_decoder_inputs, batch_size=FLAGS.batch_size)
                _, loss, _, _ = seq_rnn_model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_target_weights,forward_only=False)
                # predict_labels = seq_rnn_model.predict(outputs)
                # acc = accuracy(predict_labels, target_labels)
                print("epoch %d batch %d: loss=%f" %(epoch, batch, loss))
                batch += 1
            # if epoch % display_epoch == 0:
            #     print("epoch %d:display" %(epoch))
            if epoch % FLAGS.save_epoches == 0:
                saver.save(sess, get_modelpath(), global_step=epoch)
            #     # do test using test dataset
            #     test_encoder_inputs, test_decoder_inputs = data.test.next_batch(data.test.size())
            #     target_labels = get_target_labels(test_decoder_inputs)
            #     test_encoder_inputs = test_encoder_inputs.reshape((-1, n_steps, n_input))
            #     test_encoder_inputs, test_decoder_inputs, test_target_weights = seq_rnn_model.get_batch(test_encoder_inputs, test_decoder_inputs, batch_size=data.test.size())
            #     _, _, outputs = seq_rnn_model.step(sess, test_encoder_inputs, test_decoder_inputs, test_target_weights, forward_only=True) # don't do optimize
            #     predict_labels = seq_rnn_model.predict(outputs)
            #     acc = accuracy(predict_labels, target_labels)
            #     print("epoch %d:save, acc=%f" %(epoch, acc))
            epoch += 1

def test():
    data = model_data.read_data(FLAGS.data_path)
    seq_rnn_model = SequenceRNNModel(FLAGS.n_input_fc, FLAGS.n_views, FLAGS.n_hidden, FLAGS.decoder_embedding_size, FLAGS.n_classes+1, FLAGS.n_hidden,
                                     batch_size=data.test.size(),
                                     is_training=False,
                                     use_lstm=FLAGS.use_lstm,
                                     use_attention=FLAGS.use_attention,
                                     use_embedding=FLAGS.use_embedding,
                                     num_heads=FLAGS.num_heads)
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=config) as sess:
        seq_rnn_model.build_model()
        saver = tf.train.Saver()
        saver.restore(sess, get_modelpath())

        # proj_w_var = [v for v in tf.trainable_variables() if 'proj_w' in v.name][0]
        # proj_w_value = sess.run(proj_w_var)
        # print("shape:", proj_w_value.shape)
        # np.save("proj_w", proj_w_value)
        # return

        # train_encoder_inputs, train_decoder_inputs = data.train.next_batch(data.train.size(), shuffle=False)
        test_encoder_inputs, test_decoder_inputs = data.test.next_batch(data.test.size(), shuffle=False)
        target_labels = get_target_labels(test_decoder_inputs)
        test_encoder_inputs = test_encoder_inputs.reshape((-1, FLAGS.n_views, FLAGS.n_input_fc))
        #train_encoder_inputs = train_encoder_inputs.reshape((-1, n_steps, n_input))
        test_encoder_inputs, test_decoder_inputs, test_target_weights = seq_rnn_model.get_batch(test_encoder_inputs,
                                                                                                test_decoder_inputs,
                                                                                                batch_size=data.test.size())
        # train_encoder_inputs, train_decoder_inputs, train_target_weights = seq_rnn_model.get_batch(train_encoder_inputs,
                                                                                                # train_decoder_inputs,
                                                                                                # batch_size=data.train.size())
        # _, _, outputs_train, hidden_train = seq_rnn_model.step(sess, train_encoder_inputs, train_decoder_inputs, train_target_weights,
        #                                     forward_only=True)
        #np.save("hidden_feature_train_embedding", hidden_train)
        _, _, outputs, attns_weights = seq_rnn_model.step(sess, test_encoder_inputs, test_decoder_inputs, test_target_weights,
                                          forward_only=True)  # don't do optimize
        attns_weights = np.array([attn_weight[0] for attn_weight in attns_weights])
        attns_weights = np.transpose(attns_weights, (1, 0, 2))
        predict_labels = seq_rnn_model.predict(outputs, all_min_no=False)
        acc = accuracy(predict_labels, target_labels)

        with open(FLAGS.test_acc_file, 'a') as f:
            w = csv.writer(f)
            w.writerow([get_modelpath(), acc])
        print("model:%s, acc=%f" % (get_modelpath(), acc))


def get_target_labels(seq_labels):
    target_labels = []
    for i in xrange(np.shape(seq_labels)[0]): #loop batch_size
        for j in xrange(np.shape(seq_labels)[1]): #loop label
            if seq_labels[i][j] % 2 == 1:
                target_labels.append((seq_labels[i][j]+1)/2)
                break
    return target_labels

def accuracy(predict, target):
    return np.mean(np.equal(predict, target))

def get_modelpath():
    if FLAGS.use_embedding and FLAGS.train:
        return FLAGS.save_seq_embeddingmvmodel_path
    elif FLAGS.use_embedding and not FLAGS.train:
        return FLAGS.seq_embeddingmvmodel_path
    elif not FLAGS.use_embedding and FLAGS.train:
        return FLAGS.save_seq_basicmvmodel_path
    else:
        return FLAGS.seq_basicmvmodel_path

if __name__ == '__main__':
    tf.app.run()
