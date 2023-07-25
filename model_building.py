#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf

def build_model(input_shape_numerical, input_shape_categorical, num_unique_categories, embedding_size):
    # Create separate input layers for numerical and categorical features
    input_numerical = tf.keras.layers.Input(shape=input_shape_numerical, name='numerical_input')
    input_marca = tf.keras.layers.Input(shape=input_shape_categorical, name='marca_input')
    input_modelo = tf.keras.layers.Input(shape=input_shape_categorical, name='modelo_input')
    input_versao = tf.keras.layers.Input(shape=input_shape_categorical, name='versao_input')

    # Neural network branch for numerical features
    numerical_branch = tf.keras.layers.Dense(128, activation='relu')(input_numerical)
    numerical_branch = tf.keras.layers.Dense(64, activation='relu')(numerical_branch)

    # Neural network branches for categorical features with embeddings
    embedding_marca = tf.keras.layers.Embedding(input_dim=num_unique_categories['marca'], output_dim=embedding_size)(input_marca)
    embedding_modelo = tf.keras.layers.Embedding(input_dim=num_unique_categories['modelo'], output_dim=embedding_size)(input_modelo)
    embedding_versao = tf.keras.layers.Embedding(input_dim=num_unique_categories['versao'], output_dim=embedding_size)(input_versao)

    flatten_marca = tf.keras.layers.Flatten()(embedding_marca)
    flatten_modelo = tf.keras.layers.Flatten()(embedding_modelo)
    flatten_versao = tf.keras.layers.Flatten()(embedding_versao)

    # Concatenate the outputs from both branches
    combined_features = tf.keras.layers.Concatenate()([numerical_branch, flatten_marca, flatten_modelo, flatten_versao])

    # Additional fully connected layers
    fc1 = tf.keras.layers.Dense(32, activation='relu')(combined_features)
    fc2 = tf.keras.layers.Dense(16, activation='relu')(fc1)

    # Output layer for predicting the price (Regression)
    output = tf.keras.layers.Dense(1, activation='linear')(fc2)

    # Create the model with multiple inputs and a single output
    model = tf.keras.Model(inputs=[input_numerical, input_marca, input_modelo, input_versao], outputs=output)

    return model

def train_model(model, X_numerical, X_marca, X_modelo, X_versao, y, validation_split=0.2, epochs=100, batch_size=32):
    # Compile the model for regression
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Prepare categorical features for training
    X_marca = X_marca.reshape(-1, 1)
    X_modelo = X_modelo.reshape(-1, 1)
    X_versao = X_versao.reshape(-1, 1)

    # Train the model
    history = model.fit(
        [X_numerical, X_marca, X_modelo, X_versao],
        y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return history

