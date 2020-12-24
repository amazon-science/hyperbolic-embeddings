"""An implementation of DSSM, Deep Structured Semantic Model."""
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Lambda, Layer, Concatenate, Embedding, Reshape, ReLU, Minimum, Maximum, Add, Subtract, Bidirectional, LSTM
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers.wrappers import TimeDistributed
from keras_self_attention import SeqSelfAttention
from keras_layer_normalization import LayerNormalization
from tqdm import tqdm

from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_model import BaseModel
import tensorflow_manopt as manopt
#from tensorflow_manopt.layers import ManifoldEmbedding
from tensorflow_manopt.variable import assign_to_manifold
from matchzoo import preprocessors
import itertools

@tf.keras.utils.register_keras_serializable(name="ManifoldEmbedding")
class ManifoldEmbedding(Embedding):
    """
    Manifold Embedding that utilizes manifolds from tensorflow_manopt
    """
    def __init__(self, *args, manifold, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifold = manifold

    def build(self, input_shape):
        super().build(input_shape)
        setattr(self.embeddings, "manifold", self.manifold)
    
    def get_config(self):
        config = {"manifold": self.manifold}
        return dict(list(super().get_config().items()) + list(config.items()))

class SelfAttention(Layer):
    """
    Self Attention Layer
    """
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(SelfAttention,self).__init__()   
    def build(self, input_shape):       
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")     
        super(SelfAttention,self).build(input_shape) 
    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        return mask
    def call(self, x, mask=None):      
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a        
        if self.return_sequences:
            return output   
        return K.sum(output, axis=1)
    def get_config(self):
        config = super().get_config().copy()
        return config

class Hyperboloid(BaseModel):
    """
    Hyperboloid model.

    Examples:
        >>> model = Hyperboloid()
        >>> model.params['input_shapes'] = preprocessor.context['input_shapes']
    	>>> model.params['vocab_size'] = preprocessor.context['vocab_size']
    	>>> model.params['emb_dim'] = EMB_DIM
    	>>> model.params['dropout'] = DROPOUT
    	>>> model.params['box_dim'] = BOX_DIM
    	>>> model.params['task'] = task
	>>> model.guess_and_fill_missing_params(verbose=0) 
    """
    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params.add(Param(name='vocab_size', value=30000,
                         desc="Vocabulary Size"))
        params.add(Param(name='emb_dim', value=128,
                         desc="Embedding Dimension"))
        params.add(Param(name='box_dim', value=2*200,
                         desc="Box Dimension"))
        params.add(Param(name='dropout', value=0.2,
                         desc="Dropout"))
        params.add(Param(name='manifold', value=manopt.manifolds.Poincare(),
                         desc="Manifold"))
        return params
    
    def gen_trigrams(self):
        """
        Generates all trigrams for characters from `trigram_chars`
        """
        trigram_chars="0123456789abcdefghijklmnopqrstuvwxyz"
        t3=[''.join(x) for x in itertools.product(trigram_chars,repeat=3)] #len(words)>=3
        t2_start=['#'+''.join(x) for x in itertools.product(trigram_chars,repeat=2)] #len(words)==2
        t2_end=[''.join(x)+'#' for x in itertools.product(trigram_chars,repeat=2)] #len(words)==2
        t1=['#'+''.join(x)+'#' for x in itertools.product(trigram_chars)] #len(words)==1
        trigrams=t3+t2_start+t2_end+t1
        vocab_size=len(trigrams)
        trigram_map=dict(zip(trigrams,range(1,vocab_size+1))) # trigram to index mapping, indices starting from 1
        return trigram_map

    def crop_box(self, ind, start, end):
        # Crops (or slices) a Tensor on a given dimension from start to end
        # example : to crop tensor x[:, :, 5:10]
        # call slice(2, 5, 10) as you want to crop on the second dimension
        def func(x):
            return x[:,ind,start:end]
        return Lambda(func)

    def hyperbolic_add(self):
        # Adds two hyperbolic tensors
	def func(x):
            first = x[0]
            for item in x[1:]:
                first = self.params['manifold']._mobius_add(first,item)
            return first
        return Lambda(func)
    
    def hyperbolic_subtract(self):
	# Subtracts two hyperbolic tensors
        # Crops (or slices) a Tensor on a given dimension from start to end
        # example : to crop tensor x[:, :, 5:10]
        # call slice(2, 5, 10) as you want to crop on the second dimension
        def func(x):
            return self.params['manifold']._mobius_add(x[0],-x[1])
        return Lambda(func)

    def intersection_layer(self,x):
	# Intersection Layer with Attention on centers and Minimum on offsets
        all_pairs = []
        for ind1 in tqdm(range(x[0].shape[1])):
            for ind2 in range(ind1, x[1].shape[1]):
                box1_center = self.crop_box(ind1,0,self._params["box_dim"]//2)(x[0])
                box1_offset = self.crop_box(ind1,self._params["box_dim"]//2,None)(x[0])
                box2_center = self.crop_box(ind2,0,self._params["box_dim"]//2)(x[1])
                box2_offset = self.crop_box(ind2,self._params["box_dim"]//2,None)(x[1])
                concat_center = Concatenate()([box1_center,box2_center])
                reshape_concat_center = Reshape((2,self._params["box_dim"]//2))(concat_center)
                center = SelfAttention(return_sequences=True)(reshape_concat_center)
                center = Dense(self._params["box_dim"]//2)(center)
                #center = Add()([self.crop_box(i,None,None)(center) for i in range(center.shape[1])])
                center = self.hyperbolic_add()([self.crop_box(i,None,None)(center) for i in range(center.shape[1])])
                offset = Minimum()([box1_offset,box2_offset])
                intersection = Concatenate(axis=1)([center,offset])
                all_pairs.append(intersection)
        return Concatenate()(all_pairs)

    def merge_layer(self,x):
        # Merge layer according to the distance between center and boxes
	# d = 0.5 * d_in + d_out
        asin_embedding = x[1]
        all_dists = []
        for i in tqdm(range(x[0].shape[1])):
            q_center = self.crop_box(i,0,self._params["box_dim"]//2)(x[0])
            q_offset = self.crop_box(i,self._params["box_dim"]//2,None)(x[0])
            q_min = self.hyperbolic_subtract()([q_center,q_offset])
            q_max = Add()([q_center,q_offset])
            
            asin_q_max = ReLU()(self.hyperbolic_subtract()([asin_embedding,q_max]))
            q_min_asin = ReLU()(self.hyperbolic_subtract()([q_min,asin_embedding]))
            d_out = self.hyperbolic_add()([asin_q_max,q_min_asin])
            d_out_norm = LayerNormalization()(d_out)

            q_min_asin = Maximum()([q_min,asin_embedding])
            q_max_q_min_asin = Minimum()([q_max,q_min_asin])
            d_in = self.hyperbolic_subtract()([q_center,q_max_q_min_asin])
            d_in_norm = LayerNormalization()(d_in)
            scaled_d_in_norm = Lambda(lambda x: x*0.5)(d_in_norm)

            d_total = self.hyperbolic_add()([scaled_d_in_norm,d_out_norm])
            all_dists.append(d_total)
        return Add()(all_dists)

    def build(self):
        """
        Build model structure.

        Hyperboloid use Joint learning arthitecture.
        """
        print("Query")
        query_input = Input(
            name='text_left',
            shape=self._params['input_shapes'][0],
        )
        asin_input = Input(
            name='text_right',
            shape=self._params['input_shapes'][1]
        )
        #query_input = Input(shape=(self.hyper_params["MAX_LEN"],), dtype='float32')
        print(query_input)
        query = ManifoldEmbedding(self._params["vocab_size"], self._params["emb_dim"], trainable = True, mask_zero=True,manifold=self.params['manifold'])(query_input)
        #query = Embedding(self._params["vocab_size"], self._params["emb_dim"], trainable = True, mask_zero=True)(query_input)
        print(query)
        box = TimeDistributed(Dense(self._params["box_dim"], activation='sigmoid'))(query)
        print(box)        
        box_bn = BatchNormalization()(box)
        box_dropout = Dropout(self._params["dropout"])(box_bn)
        print(box_dropout)
        intersection = Lambda(self.intersection_layer)([box_dropout,box_dropout])
        print(intersection)
        reshape_intersection = Reshape((intersection.shape[1]//self._params["box_dim"],self._params["box_dim"]))(intersection)
        print(reshape_intersection)
        query_attention = SelfAttention(return_sequences=True)(reshape_intersection)
        print(query_attention)
        query_attention_bn = BatchNormalization()(query_attention)
        query_attention_dropout = Dropout(self._params["dropout"])(query_attention_bn)
        print(query_attention_dropout)
        query_boxes = Dense(self._params["box_dim"])(query_attention_dropout)
        print(query_boxes)

        print("\nASIN")
        #asin_input = Input(shape=(self.hyper_params["MAX_LEN"],), dtype='float32')
        print(asin_input)
        asin = ManifoldEmbedding(self._params["vocab_size"], self._params["emb_dim"], trainable = True, mask_zero=True,manifold=self.params['manifold'])(asin_input)
        #asin = Embedding(self._params["vocab_size"], self._params["emb_dim"], trainable = True, mask_zero=True)(asin_input)
        print(asin)
        lstm = Bidirectional(LSTM(self._params["emb_dim"]//2, return_sequences=True))(asin)
        print(lstm)
        attention = SelfAttention(return_sequences=True)(lstm)
        print(attention)
        attention_bn = BatchNormalization()(attention)
        print(attention_bn)
        attention_dropout = Dropout(self._params["dropout"])(attention_bn)
        print(attention_dropout)
        dense_attention = Dense(self._params["box_dim"]//2)(attention_dropout)
        print(dense_attention)
        asin_embedding = self.hyperbolic_add()([self.crop_box(i,None,None)(dense_attention) for i in range(dense_attention.shape[1])])
        print(asin_embedding)
        print("\nJoin")
        merged_layer = Lambda(self.merge_layer)([query_boxes,asin_embedding])
        print(merged_layer)
    	merged_layer_bn = BatchNormalization()(merged_layer)
        print(merged_layer_bn)
        merged_layer_dropout = Dropout(self._params["dropout"])(merged_layer_bn) 
        print(merged_layer_dropout)
	final_loss = self._make_output_layer()(merged_layer_dropout)
        print(final_loss)
        self._backend = Model(inputs=[query_input,asin_input], outputs=final_loss)

    def compile(self):
        """
        Compile model for training.

        Only `keras` native metrics are compiled together with backend.
        MatchZoo metrics are evaluated only through :meth:`evaluate`.
        Notice that `keras` count `loss` as one of the metrics while MatchZoo
        :class:`matchzoo.engine.BaseTask` does not.

            >>> model.build()
            >>> model.compile()

        """
        opt = manopt.optimizers.RiemannianAdam(learning_rate=0.2)
        self._backend.compile(optimizer=opt,
                              loss=self._params['task'].loss)
    @classmethod
    def get_default_preprocessor(cls):
        """:return: Default preprocessor."""
        return preprocessors.BasicPreprocessor()
