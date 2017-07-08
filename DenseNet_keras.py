import tensorflow as tf
from tensorflow.contrib.keras.api.keras.models import Model,Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D,ZeroPadding2D,BatchNormalization,Activation,MaxPool2D,Input,AveragePooling2D,GlobalAveragePooling2D,Dense
from tensorflow.contrib.keras.python.keras.layers import merge,concatenate
import numpy as np

weight_path="densenet121_weights_tf.h5"
im=np.random.ranf((224,224,3))


def conv_block(input_,filter):
    in_filter=filter*4

    x=BatchNormalization()(input_)
    x=Activation("relu")(x)
    x=Conv2D(in_filter,(1,1),strides=(1,1),padding="SAME")(x)

    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    # x = ZeroPadding2D((1, 1))(x)
    x=Conv2D(filter,(3,3),strides=(1,1),padding="SAME")(x)
    return x
def dense_block(input_,num_nodes,update_filter,growth_rate):

    concat_feat=input_
    for i in range(num_nodes):
        x=conv_block(concat_feat,growth_rate)
        # concat_feat=ZeroPadding2D((1,1))(concat_feat)
        concat_feat=concatenate([concat_feat,x],axis=3)

        update_filter+=growth_rate
    return concat_feat,update_filter

def transition_block(in_,filter,compression):
    x=BatchNormalization()(in_)
    x=Activation("relu")(x)
    x=Conv2D(int(filter*compression),(1,1),(1,1))(x)
    x=AveragePooling2D((2,2),strides=(2,2),padding="VALID")(x)
    return x
class DenseNet121():
    def __init__(self):
        self.growth_rate=32
        in_=Input(shape=(224,224,3))
        self.num_dense_block=4

        # Layer 1:
        x=Conv2D(64,(7,7),(2,2),padding="SAME")(in_)
        x=BatchNormalization()(x)
        x=Activation("relu")(x)
        x=MaxPool2D((3,3),(2,2),padding="VALID")(x)

        filter=64
        num_node_each_layer=[6,12,24,16]
        for i in range(self.num_dense_block):
            x,filter=dense_block(x,num_node_each_layer[i],filter,growth_rate=32)
            if i !=self.num_dense_block-1:
                x=transition_block(x,filter,1.0)
                filter=filter*1.0

        # Output from loop statement, x still in conv layer
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x=GlobalAveragePooling2D()(x)
        x=Dense(1000,activation="softmax")(x)
        model=Model(inputs=in_,outputs=x)
        model.summary()
        self.model=model
    # def train(self):
        # self.model.load_weights(weight_path)
model=DenseNet121()
# model.train()
