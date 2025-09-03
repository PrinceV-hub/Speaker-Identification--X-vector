from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, BatchNormalization, Dropout, 
    GlobalAveragePooling1D, Concatenate, Lambda
)
import tensorflow.keras.backend as K

def create_xvector_model(input_shape, num_classes):
    """Create X-Vector model architecture"""
    inputs = Input(shape=input_shape, name='input')
    
    # Frame-level layers (TDNN)
    x = Conv1D(512, kernel_size=5, dilation_rate=1, activation='relu', name='tdnn1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    
    x = Conv1D(512, kernel_size=3, dilation_rate=2, activation='relu', name='tdnn2')(x)
    x = BatchNormalization(name='bn2')(x)
    
    x = Conv1D(512, kernel_size=3, dilation_rate=3, activation='relu', name='tdnn3')(x)
    x = BatchNormalization(name='bn3')(x)
    
    x = Conv1D(512, kernel_size=1, dilation_rate=1, activation='relu', name='tdnn4')(x)
    x = BatchNormalization(name='bn4')(x)
    
    x = Conv1D(1500, kernel_size=1, dilation_rate=1, activation='relu', name='tdnn5')(x)
    x = BatchNormalization(name='bn5')(x)
    
    # Statistics pooling
    mean = GlobalAveragePooling1D(name='global_mean')(x)
    mean_expanded = Lambda(lambda x: K.expand_dims(x, axis=1))(mean)
    mean_tiled = Lambda(lambda x: K.tile(x[0], [1, K.shape(x[1])[1], 1]))([mean_expanded, x])
    variance = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([x, mean_tiled])
    std = Lambda(lambda x: K.sqrt(K.maximum(x, 1e-8)))(variance)
    
    stats = Concatenate(name='stats_pool')([mean, std])
    
    # Segment-level layers
    x = Dense(512, activation='relu', name='segment1')(stats)
    x = BatchNormalization(name='bn6')(x)
    x = Dropout(0.5, name='dropout1')(x)
    
    embeddings = Dense(512, activation='relu', name='embeddings')(x)
    x = BatchNormalization(name='bn7')(embeddings)
    x = Dropout(0.5, name='dropout2')(x)
    
    outputs = Dense(num_classes, activation='softmax', name='classification')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='XVector')

def create_embedding_model(trained_model):
    """Create embedding extraction model from trained model"""
    return Model(
        inputs=trained_model.input, 
        outputs=trained_model.get_layer('embeddings').output
    )
