from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, ReLU, BatchNormalization, Add
from tensorflow.keras.models import Model


def conv_layer(input_block, filters, kernel_size):
    """
    Exact Feature
    """

    x = Conv2D(filters, kernel_size, padding='same', data_format="channels_first")(input_block)
    
    x = BatchNormalization(axis=1)(x)
    
    x = ReLU()(x)
    return x

def residual_layer(input_block, filters, kernel_size):
    
    x = conv_layer(input_block, filters, kernel_size)
    
    x = Conv2D(filters, kernel_size, padding='same', data_format="channels_first")(x)
    x = BatchNormalization(axis=1)(x)
    
    x = Add()([input_block, x])
    
    x = ReLU()(x)
    return x

def build_architecture(res_blocks=7):
   
    main_input = Input(shape=(3, 6, 7), name='main_input')

    x = conv_layer(main_input, filters=64, kernel_size=(4, 4))
    
    for _ in range(res_blocks):
        x = residual_layer(x, filters=64, kernel_size=(4, 4))

    # --- Policy Head ---
    policy = Conv2D(filters=2, kernel_size=(1, 1), padding='same', data_format="channels_first")(x)
    policy = BatchNormalization(axis=1)(policy)
    policy = ReLU()(policy)
    policy = Flatten()(policy)
    
    policy = Dense(7, activation='softmax', name='policy_head')(policy)

    # --- 2: Value Head  ---
    value = Conv2D(filters=1, kernel_size=(1, 1), padding='same', data_format="channels_first")(x)
    value = BatchNormalization(axis=1)(value)
    value = ReLU()(value)
    value = Flatten()(value)
    value = Dense(64, activation='relu')(value)
    #  tanh to ensure out put is between -1 and 1, representing loss and win respectively
    value = Dense(1, activation='tanh', name='value_head')(value)

    return main_input, policy, value