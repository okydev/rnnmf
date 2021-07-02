import numpy as np
np.random.seed(1337)

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Reshape, Flatten, Dropout, Dense
from keras.layers import Embedding
from keras.models import Model
from keras.layers import Input, concatenate, LSTM
from keras.preprocessing import sequence


class RNN_module():
    '''
    classdocs
    '''
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        filter_lengths = [3, 4, 5]

        '''Embedding Layer'''
        model_input = Input(shape=(max_len,), dtype='int32', name='input')

        if init_W is None:
            model_embedding = Embedding(max_features, emb_dim, input_length=max_len)(model_input)
        else:
            model_embedding = Embedding(max_features, emb_dim, input_length=max_len, weights=[
                                init_W / 20])(model_input)

        ''' Use RNN LSTM'''
        model_chuanlian = Sequential()
        model_chuanlian.add(LSTM(emb_dim, return_sequences=True, activation='tanh'))
        model_chuanlian.add(LSTM(nb_filters, activation='tanh'))
        model_chuanlian = model_chuanlian(model_embedding)

        '''Convolution Layer & Max Pooling Layer'''
        # for i in filter_lengths:
        #     model_internal = Sequential()
        #     model_internal.add(Reshape((self.max_len,emb_dim,1),input_shape=(self.max_len, emb_dim)))
        #     model_internal.add(Conv2D(nb_filters, (i,emb_dim),strides=(1,1),activation="relu"))
        #     model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1)))
        #     model_internal.add(Flatten())

        #     if i==3:
        #        model_convolutional_3=model_internal(model_embedding)
        #     if i==4:
        #        model_convolutional_4=model_internal(model_embedding) 
        #     if i==5:
        #         model_convolutional_5=model_internal(model_embedding)

        '''Dropout Layer'''
        # model_chuanlian=concatenate([model_convolutional_3,model_convolutional_4,model_convolutional_5])
        model_dropout = Dense(vanila_dimension, activation='tanh')(model_chuanlian)
        model_dropout_all = Dropout(dropout_rate)(model_dropout)

        '''Projection Layer & Output Layer'''
        model_projection=Dense(projection_dimension, activation='tanh')(model_dropout_all)

        # Output Layer
        model_output=Model(inputs=model_input,outputs=model_projection)
        model_output.compile(optimizer='rmsprop', loss='mse')
        self.model=model_output

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
        self.max_len = max_len
        max_features = vocab_size

        filter_lengths = [3, 4, 5]
        print("Build model...")

        self.qual_conv_set = {}
        '''Embedding Layer'''
        qual_model_input=Input(input_shape=(max_len,), dtype='int32',name='qual_model_input')

        qual_model_embedding=Embedding(max_features, emb_dim, input_length=max_len, weights=model_embedding.get_weights())(qual_model_input)

        ''' Use RNN LSTM'''
        model_chuanlian = Sequential()
        model_chuanlian.add(LSTM(emb_dim, return_sequences=True, activation='tanh'))
        model_chuanlian.add(LSTM(nb_filters, activation='tanh'))
        model_chuanlian = model_chuanlian(qual_model_embedding)
        '''Convolution Layer & Max Pooling Layer'''
        """
        for i in filter_lengths:
            model_internal = Sequential()
            model_internal.add(Reshape((max_len, emb_dim,1), input_shape=(max_len, emb_dim)))
            self.qual_conv_set[i] = Conv2D(nb_filters,(i,emb_dim),strides=(1,1),activation="relu",weights='model_convolutional_'+str(i).layers[1].get_weights())
            model_internal.add(self.qual_conv_set[i])
            model_internal.add(MaxPooling2D(pool_size=(max_len - i + 1, 1)))
            model_internal.add(Flatten())
            
            if i==3:
               qual_model_convolutional_3=model_internal(qual_model_embedding)
            if i==4:
               qual_model_convolutional_4=model_internal(qual_model_embedding)
            if i==5:
               qual_model_convolutional_5=model_internal(qual_model_embedding)
        """
        qual_model_output=Model(inputs=qual_model_input,outputs=[model_chuanlian.get_layer(-1).output]) 

        qual_model_output.compile(optimizer='rmsprop', loss={model_chuanlian.get_layer(-1).name: 'mse'})
        self.qual_model=qual_model_output

    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit(x=X_train,y=V,verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch, sample_weight=item_weight)

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, X_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        Y = self.model.predict(
            {'input': X_train}, batch_size=len(X_train))
        return Y
