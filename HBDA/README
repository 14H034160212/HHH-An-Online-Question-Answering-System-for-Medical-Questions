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
according to the medical semantic similarity model (HBDA).


The AttentionLayer is referred from the following link.

https://github.com/uhauha2929/examples/blob/master/Hierarchical%20Attention%20Networks%20.ipynb


The Siamese_LSTM network and Manhattan distance are referred from the following links.

https://zhuanlan.zhihu.com/p/31638132

https://github.com/likejazz/Siamese-LSTM

https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM
