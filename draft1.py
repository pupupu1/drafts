from tensorflow.python.ops.rnn import _transpose_batch_time
import tensorflow as tf


def sampling_rnn(self, cell, initial_state, input_, seq_lengths):

    # raw_rnn expects time major inputs as TensorArrays
    max_time = ...  # this is the max time step per batch
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time, clear_after_read=False)
    inputs_ta = inputs_ta.unstack(_transpose_batch_time(input_))  # model_input is the input placeholder
    input_dim = input_.get_shape()[-1].value  # the dimensionality of the input to each time step
    output_dim = ...  # the dimensionality of the model's output at each time step

        def loop_fn(time, cell_output, cell_state, loop_state):
            """
            Loop function that allows to control input to the rnn cell and manipulate cell outputs.
            :param time: current time step
            :param cell_output: output from previous time step or None if time == 0
            :param cell_state: cell state from previous time step
            :param loop_state: custom loop state to share information between different iterations of this loop fn
            :return: tuple consisting of
              elements_finished: tensor of size [bach_size] which is True for sequences that have reached their end,
                needed because of variable sequence size
              next_input: input to next time step
              next_cell_state: cell state forwarded to next time step
              emit_output: The first return argument of raw_rnn. This is not necessarily the output of the RNN cell,
                but could e.g. be the output of a dense layer attached to the rnn layer.
              next_loop_state: loop state forwarded to the next time step
            """
            if cell_output is None:
                # time == 0, used for initialization before first call to cell
                next_cell_state = initial_state
                # the emit_output in this case tells TF how future emits look
                emit_output = tf.zeros([output_dim])
            else:
                # t > 0, called right after call to cell, i.e. cell_output is the output from time t-1.
                # here you can do whatever ou want with cell_output before assigning it to emit_output.
                # In this case, we don't do anything
                next_cell_state = cell_state
                emit_output = cell_output  

            # check which elements are finished
            elements_finished = (time >= seq_lengths)
            finished = tf.reduce_all(elements_finished)

            # assemble cell input for upcoming time step
            current_output = emit_output if cell_output is not None else None
            input_original = inputs_ta.read(time)  # tensor of shape (None, input_dim)

            if current_output is None:
                # this is the initial step, i.e. there is no output from a previous time step, what we feed here
                # can highly depend on the data. In this case we just assign the actual input in the first time step.
                next_in = input_original
            else:
                # time > 0, so just use previous output as next input
                # here you could do fancier things, whatever you want to do before passing the data into the rnn cell
                # if here you were to pass input_original than you would get the normal behaviour of dynamic_rnn
                next_in = current_output

            next_input = tf.cond(finished,
                                 lambda: tf.zeros([self.batch_size, input_dim], dtype=tf.float32),  # copy through zeros
                                 lambda: next_in)  # if not finished, feed the previous output as next input

            # set shape manually, otherwise it is not defined for the last dimensions
            next_input.set_shape([None, input_dim])

            # loop state not used in this example
            next_loop_state = None
            return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    outputs_ta, last_state, _ = tf.nn.raw_rnn(cell, loop_fn)
    outputs = _transpose_batch_time(outputs_ta.stack())
    final_state = last_state

return outputs, final_state


______________________________________________________
import tensorflow as tf
from tensorflow.keras import layers, models

class CascadedRNNLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, output_size):
        super(CascadedRNNLayer, self).__init__()
        self.hidden_size = hidden_size
        
        # Feedforward network components
        self.input_to_hidden = layers.Dense(hidden_size, activation='relu')
        self.hidden_to_output = layers.Dense(output_size)
        self.input_to_output = layers.Dense(output_size)
        
        # LSTM cell for the feedback loop
        self.lstm_cell = layers.LSTMCell(hidden_size)
    
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Initialize hidden and cell state for LSTM cell
        h_t = tf.zeros((batch_size, self.hidden_size))
        c_t = tf.zeros((batch_size, self.hidden_size))
        
        # Initialize previous output for the feedback loop
        y_t_minus_1 = tf.zeros((batch_size, self.hidden_size))
        
        outputs = []
        
        for t in range(seq_length):
            x_t = inputs[:, t, :]
            
            # Input to hidden layer
            h_t_ff = self.input_to_hidden(x_t)
            
            # Hidden to output layer
            output_ff = self.hidden_to_output(h_t_ff)
            
            # Direct input to output layer (skip connection)
            output_skip = self.input_to_output(x_t)
            
            # Combine feedforward outputs
            y_t_ff = output_ff + output_skip
            
            # LSTM cell for feedback loop
            _, (h_t, c_t) = self.lstm_cell(y_t_minus_1, states=[h_t, c_t])
            y_t_fb = self.hidden_to_output(h_t)
            
            # Final output for the current time step
            y_t = y_t_ff + y_t_fb
            
            # Store the current output for the next time step feedback
            y_t_minus_1 = y_t
            
            outputs.append(y_t)
        
        outputs = tf.stack(outputs, axis=1)
        return outputs

# Parameters
input_size = 10
hidden_size = 20
output_size = 1

# Initialize custom layer
cascaded_rnn_layer = CascadedRNNLayer(input_size, hidden_size, output_size)

# Build the Sequential model
model = models.Sequential([
    layers.Input(shape=(None, input_size)),
    cascaded_rnn_layer
])

# Example input
batch_size = 5
sequence_length = 7
input_data = tf.random.normal((batch_size, sequence_length, input_size))

# Forward pass
output = model(input_data)
print(output)
