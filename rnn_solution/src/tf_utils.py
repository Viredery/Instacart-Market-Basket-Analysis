import tensorflow as tf

def lstm_layer(inputs, lengths, state_size, keep_prob=1.0, scope='lstm-layer',reuse=False, return_final_state=False):
    """Long Short-Term Memory Layer

    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(state_size, reuse=reuse),
                output_keep_prob=keep_prob)
        outputs, output_state = tf.nn.dynamic_rnn(
                inputs=inputs, cell=cell_fw, sequence_length=lengths, dtype=tf.float32)
        
        if return_final_state:
            return outputs
        else:
            return outputs, output_state
