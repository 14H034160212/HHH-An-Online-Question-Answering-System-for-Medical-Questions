This is the hiererachical BiLSTM Word Attention Model. 

The training platform is Colab.

The total number of data = 10000

The number distribution of Train: test = 8:2

Epoch: 9

First experiment result eval_accuracy: 0.8130

Second experiment result eval_accuracy: 0.8064

Third experiment result eval_accuracy: 0.8244

Average eval_accuracy: 0.8146

Range of change: (-0.0082, +0.0098)

The main innovation is to use the Siamese_LSTM + word attention + Manhattan distance 
compare to the normal Siamese_LSTM + normal attention + Manhattan distance.

The main reason that we need to model is to retrieve the top k answer from the medical question answer pair dataset
according to the medical semantic similarity model (HBAM).


The AttentionLayer is referred from the following link.

https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb


The Siamese_LSTM network and Manhattan distance are referred from the following links.

https://zhuanlan.zhihu.com/p/31638132

https://github.com/likejazz/Siamese-LSTM

https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM

Here are the key code for the AttentionLayer.py
```
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)
    
    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        # general
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
```
