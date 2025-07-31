! pip install -q "flwr-datasets[vision]"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import SGDRegressor
from tensorflow.keras.models import Model # Import Model for functional API
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization, Activation, Add # Import Add for skip connections
from itertools import combinations
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.partitioner import PathologicalPartitioner
import itertools
print("GPUs available:", tf.config.list_physical_devices('GPU'))
# --- Paper parameters ---
CNN_CLIENTS = 20
CNN_ACTIVE = 5
LR_CLIENTS = 20
LR_ACTIVE = 5
MNIST_INPUT_SHAPE = (28, 28, 1)
CIFAR10_INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10
MIN_CLIENT_TIME = 1
MAX_CLIENT_TIME = 2

# --- Global Model Building Functions (used by both Server and Clients) ---

# Helper function to create a basic residual block
def residual_block(x, filters, kernel_size=(3, 3), stride=1, conv_shortcut=False, name=None):
    """
    A simplified residual block for ResNet-like architecture.
    """
    if name is None:
        name = 'res_block'

    shortcut = x

    # First convolutional layer
    x = Conv2D(filters, kernel_size, strides=stride, padding='same', name=name + '_conv1')(x)
    x = BatchNormalization(name=name + '_bn1')(x)
    x = Activation('relu', name=name + '_relu1')(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', name=name + '_conv2')(x)
    x = BatchNormalization(name=name + '_bn2')(x)

    # Shortcut connection
    if stride != 1 or x.shape[-1] != shortcut.shape[-1] or conv_shortcut:
        # If dimensions change, apply 1x1 convolution for shortcut
        shortcut = Conv2D(filters, (1, 1), strides=stride, name=name + '_shortcut_conv')(shortcut)
        shortcut = BatchNormalization(name=name + '_shortcut_bn')(shortcut)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_relu_out')(x)
    return x


def CNN_model_building_static(input_shape, num_classes, learning_rate=0.01):
    """
    Builds a ResNet-like CNN model.
    Adjusts input_shape and output based on dataset.
    """
    input_tensor = Input(shape=input_shape)

    # Initial Convolutional Block
    # Using smaller kernel for initial conv if images are smaller (e.g., Fashion MNIST)
    initial_kernel_size = (3, 3) if input_shape[0] <= 32 else (7, 7)
    initial_stride = (1, 1) if input_shape[0] <= 32 else (2, 2)
    initial_pool_size = (2, 2) if input_shape[0] <= 32 else (3, 3)
    initial_pool_stride = (2, 2) if input_shape[0] <= 32 else (2, 2)


    x = Conv2D(64, initial_kernel_size, strides=initial_stride, padding='same', name='conv1')(input_tensor)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    x = MaxPool2D(initial_pool_size, strides=initial_pool_stride, padding='same', name='pool1')(x)

    # --- Residual Blocks ---
    # Block 1 (64 filters)
    x = residual_block(x, filters=64, name='res2a')
    x = residual_block(x, filters=64, name='res2b')

    # Block 2 (128 filters, with stride to downsample)
    x = residual_block(x, filters=128, stride=2, conv_shortcut=True, name='res3a')
    x = residual_block(x, filters=128, name='res3b')

    # Global Average Pooling (common in ResNets before the final dense layer)
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Final Dense Layer with Softmax
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create the model using Keras Functional API
    with tf.device('/GPU:0'):
        model = Model(inputs=input_tensor, outputs=x, name='ResNet_Like_Dynamic')
        METRICS = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # שנה כאן את ה-learning rate
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=METRICS)
    return model

def LR_model_building_static(learning_rate=0.01):
    model = SGDRegressor(learning_rate='constant', eta0=learning_rate)
    return model

class Client:
    def __init__(self, X_train, y_train, method, learning_rate):
        #t parameter latency of the client
        self.t = random.uniform(MIN_CLIENT_TIME, MAX_CLIENT_TIME)
        #parameters for cs-ucb algorithm
        self.cs_ucb_z = 0
        self.cs_ucb_y = 0
        self.cs_ucb_r = 0
        self.cs_ucb_s = 0
        #parameters for cs-ucb-q algorithm
        self.cs_ucb_q_z = 0
        self.cs_ucb_q_y = 0
        self.cs_ucb_q_y_hat = 0
        self.cs_ucb_q_b = 0
        self.cs_ucb_q_c = 0
        self.cs_ucb_q_D = 0
        self.cs_ucb_q_s = 0
        #parameters for rbcf-f algorithm
        self.rbcs_f_z = 0
        self.rbcs_f_b = np.zeros(3)
        self.rbcs_f_H = np.identity(3)
        self.rbcs_f_theta = np.zeros(3)
        self.rbcs_f_T_hat = 0
        self.rbcs_f_T = 0
        self.rbcs_f_c = np.array([1, 0, 1])
        self.rbcs_f_f = 0
        self.rbcs_f_x = 0
        #parameters for weighted random
        self.probability = 0
        #parameters for bsfl algorithm'
        self.bsfl_ucb = float('inf')
        self.bsfl_c = 0
        self.bsfl_miu = 0
        self.bsfl_g = 0
        self.bsfl_r = 0
        #general parameters for client
        self.loss = 0
        self.method = method
        self.learning_rate = learning_rate
        self.model = self.set_model()
        self.X_train = X_train
        self.y_train = y_train
        self.dataset_size = len(self.X_train)
        self.optimal = None

    def reset_model(self):
        del self.model
        self.model = self.set_model()

    def set_model(self):
        if self.method not in ['CIFAR10', 'MNIST', 'LR']:
            raise ValueError("Invalid case. Must be 'CIFAR10', 'MNIST' or 'LR'.")
        if self.method == 'LR':
            return LR_model_building_static(self.learning_rate)
        elif self.method == 'MNIST':
            return CNN_model_building_static(MNIST_INPUT_SHAPE, NUM_CLASSES, learning_rate=self.learning_rate)
        else:
            return CNN_model_building_static(CIFAR10_INPUT_SHAPE, NUM_CLASSES, learning_rate=self.learning_rate)
        
    def get_local_model_weights(self):
        """Returns the current weights of the client's local model."""
        if self.method in ['CIFAR10', 'MNIST']:
            return self.model.get_weights()
        elif self.method == 'LR':
            if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                return (self.model.coef_, self.model.intercept_)
            else:
                num_features = 10
                random_coef = np.random.randn(num_features) * 0.01 
                random_intercept = np.random.randn() * 0.01 
                return (random_coef, random_intercept)
        else:
            raise ValueError("Invalid model type for getting weights.")

    def set_weights(self, weights):
        """
        Sets the weights of the client's local models.
        Should be called by the server to send the global model.
        """
        if self.method in ['CIFAR10', 'MNIST']:
            self.model.set_weights(weights)
        else:
            # Check if model is initialized and has coef_ and intercept_ attributes
            if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
                self.model.coef_ = weights[0]
                self.model.intercept_ = weights[1]
            else:
                # If not initialized, set them directly if model was just created
                # This is a bit of a hack, sklearn models are fitted to learn parameters
                # A fresh LR model doesn't have coef_ or intercept_ until .fit() is called.
                # For federated learning, you might need to manually set these after init.
                # However, for simplicity here, we assume it's initialized via server or a prior fit.
                # The preferred way is to ensure a dummy fit happens to create these attributes
                # or initialize them manually if LRmodel_params is provided.
                if len(weights[0]) == self.X_train.shape[1]: # Basic check
                    self.model.coef_ = np.array(weights[0])
                    self.model.intercept_ = np.array(weights[1])
                else:
                    print(f"Warning: LR model params mismatch input shape for client. Skipping setting weights.")

    def local_train(self, epochs=1, batch_size=32):
        """Trains the client's CNN model."""
        if self.method in ['CIFAR10', 'MNIST']:
            r = self.model.fit(self.X_train, self.y_train,
                                epochs=epochs, batch_size=batch_size, verbose=0)
            self.loss = r.history['loss'][-1]
        else:
            """Trains the client's LR model."""
            self.model.fit(self.X_train, self.y_train)
            # Ensure there's test data for evaluation
            y_pred = self.model.predict(self.X_train)
            self.loss = mean_squared_error(self.y_train, y_pred)

class Server:
    def __init__(self, K, m, method, server_case, learning_rate):
        # general parameters
        self.K = K
        self.m = m
        self.t_max = 0
        self.t_min = 0
        self.loss = 0
        self.model = self.set_model(method, learning_rate)
        self.learning_rate = learning_rate
        self.method = method
        self.case = server_case
        self.X_train, self.y_train, self.X_test, self.y_test = self.set_dataset()
        self.dataset_size = len(self.X_train)
        self.clients = self.set_clients()

    def reset_model(self):
        del self.model
        self.model = self.set_model(self.method, self.learning_rate)
        for client in self.clients:
            client.reset_model()

    def set_model(self, method, learning_rate=0.01):
        if method not in ['CIFAR10', 'MNIST', 'LR']:
            raise ValueError("Invalid case. Must be 'CIFAR10', 'MNIST' or 'LR'.")
        if not isinstance(learning_rate, float):
            raise ValueError("learning_rate must be a float value!")
        if method == 'LR':
            return LR_model_building_static(learning_rate)
        elif method == 'MNIST':
            return CNN_model_building_static(MNIST_INPUT_SHAPE, NUM_CLASSES, learning_rate=learning_rate)
        else:
            return CNN_model_building_static(CIFAR10_INPUT_SHAPE, NUM_CLASSES, learning_rate=learning_rate) 
    
    def set_dataset(self):
        if self.method == 'LR':
            # Generate random features
            X = np.random.rand(100000, 10)# Features between 0 and 10
            true_coef = np.random.rand(10) * 10 # Coefficients between 0 and 5
            true_intercept = random.uniform(1, 10) # Intercept between 1 and 10
            y = X @ true_coef + true_intercept + np.random.randn(100000) * 10
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        elif self.method == 'CIFAR10':
            fds = FederatedDataset(dataset="cifar10", partitioners={"train": 1}, trust_remote_code=True,)
            partition = fds.load_partition(0, "train")
            X_train = partition['img']
            y_train = partition['label']
            centralized_dataset = fds.load_split("test")
            X_test = centralized_dataset['img']
            y_test = centralized_dataset['label']
        elif self.method == 'MNIST':
            fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": 1}, trust_remote_code=True,)
            partition = fds.load_partition(0, "train")
            X_train = partition['img']
            y_train = partition['label']
            centralized_dataset = fds.load_split("test")
            X_test = centralized_dataset['img']
            y_test = centralized_dataset['label']
        else:
            raise ValueError("method must be 'LR', 'CIFAR10' or 'MNIST'.")
        if self.method in ['CIFAR10', 'MNIST']:
            X_test = np.stack([np.array(img) for img in X_test])  # shape (K, 32, 32, 3) or (K, 28, 28, 3)
            X_test = X_test.astype('float32') / 255.0
            num_classes = 10  # classes of CIFAR10 and Fashion-MNIST
            y_test = to_categorical(y_test, num_classes=num_classes)
        return X_train, y_train, X_test, y_test
    
    def partition_data(self):
        clients_dataset = []
        if self.case == 'i_i_d':
            if self.method == 'LR':
                n_samples = self.X_train.shape[0]
                indices = np.random.permutation(n_samples)  # Shuffle indices randomly
    
                # Split indices into K chunks (as equal as possible)
                splits = np.array_split(indices,self.K)
                for idxs in splits:
                    X_train = self.X_train[idxs]
                    y_train = self.y_train[idxs]
                    clients_dataset.append([X_train, y_train])
            elif self.method == 'CIFAR10':
                partitioner = IidPartitioner(num_partitions=self.K)
                fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner,}, trust_remote_code=True,)
                for i in range(self.K):
                    partition = fds.load_partition(i, "train")
                    X_train = partition['img']
                    y_train = partition['label']  
                    X_train = np.stack([np.array(img) for img in X_train])  # shape (K, 32, 32, 3) or (K, 28, 28, 3)
                    X_train = X_train.astype('float32') / 255.0
                    num_classes = 10  # classes CIFAR10 or Fasion-MNIST
                    y_train = to_categorical(y_train, num_classes=num_classes)
                    clients_dataset.append([X_train, y_train])     
            elif self.method == 'MNIST':
                partitioner = IidPartitioner(num_partitions=self.K)
                fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": partitioner,}, trust_remote_code=True,)
                for i in range(self.K):
                    partition = fds.load_partition(i, "train")
                    X_train = partition['img']
                    y_train = partition['label']
                    X_train = np.stack([np.array(img) for img in X_train])  # shape (K, 32, 32, 3) or (K, 28, 28, 3)
                    X_train = X_train.astype('float32') / 255.0
                    num_classes = 10  # classes CIFAR10 or Fasion-MNIST
                    y_train = to_categorical(y_train, num_classes=num_classes)
                    clients_dataset.append([X_train, y_train])       
            else:
                raise ValueError("method must be 'LR', 'CIFAR10' or 'MNIST'.")
        elif self.case == 'non_i_i_d':
            if self.method == 'LR':
                # Sort samples by target y
                sorted_indices = np.argsort(self.y_train)
                # Split sorted indices into K partitions (so each has samples with similar y values)
                splits = np.array_split(sorted_indices, self.K)
                for idxs in splits:
                    X_train = self.X_train[idxs]
                    y_train = self.y_train[idxs]
                    clients_dataset.append([X_train, y_train])
            elif self.method == 'CIFAR10':
                pathological_partitioner = PathologicalPartitioner(num_partitions=self.K, partition_by="label", num_classes_per_partition=2)
                fds = FederatedDataset(dataset="cifar10", partitioners={"train": pathological_partitioner}, trust_remote_code=True,)
                for i in range(self.K):
                    partition = fds.load_partition(i, "train")
                    X_train = partition['img']
                    y_train = partition['label']
                    X_train = np.stack([np.array(img) for img in X_train])  # shape (K, 32, 32, 3) or (K, 28, 28, 3)
                    X_train = X_train.astype('float32') / 255.0
                    num_classes = 10  # classes CIFAR10 or Fasion-MNIST
                    y_train = to_categorical(y_train, num_classes=num_classes)
                    clients_dataset.append([X_train, y_train])       
            elif self.method == 'MNIST':
                pathological_partitioner = PathologicalPartitioner(num_partitions=self.K, partition_by="label", num_classes_per_partition=2)
                fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": pathological_partitioner}, trust_remote_code=True,)
                for i in range(self.K):
                    partition = fds.load_partition(i, "train")
                    X_train = partition['img']
                    y_train = partition['label']
                    X_train = np.stack([np.array(img) for img in X_train])  # shape (K, 32, 32, 3) or (K, 28, 28, 3)
                    X_train = X_train.astype('float32') / 255.0
                    num_classes = 10  # classes CIFAR10 or Fasion-MNIST
                    y_train = to_categorical(y_train, num_classes=num_classes)
                    clients_dataset.append([X_train, y_train])    
            else:
                raise ValueError("method must be 'LR', 'CIFAR10' or 'MNIST'.")
        else:
            raise ValueError("case must be 'i_i_d' or 'non_i_i_d'")
        return clients_dataset


    def set_clients(self):
        clients_dataset = self.partition_data()
        clients = []
        for i in range(self.K):
            X_train, y_train = clients_dataset[i]
            clients.append(Client(X_train=X_train, y_train=y_train, method=(self.method), learning_rate=self.learning_rate))
        return clients

    def compute_total_loss(self):
        """
        Computes the weighted average of clients' LOCAL losses.
        This is a metric for overall system performance, not for updating weights.
        """
        self.loss = sum(client.loss * ((client.dataset_size)/(self.dataset_size)) for client in (self.clients))
        return self.loss

    def update_model(self, selected_clients):
        """
        Updates the server's global model with the aggregated weights from clients.
        """
        aggregated_weights = [client.get_local_model_weights() for client in self.clients]

        if aggregated_weights is None:
            print(f"Warning: No aggregated weights provided for {self.method} model update.")
            return

        if self.method in  ['CIFAR10', 'MNIST']:
            avg_weights = [
                          sum(client_weights[layer] * client.dataset_size for client_weights, client in zip(aggregated_weights, self.clients)) / self.dataset_size
                          for layer in range(len(aggregated_weights[0]))
                          ]
            self.model.set_weights(avg_weights)
        else:
            # Sklearn LR doesn't have a direct set_weights, need to assign coef_ and intercept_
            # Check if coef_ and intercept_ are valid for the model's dimensions
            total_clients = len(aggregated_weights)
            avg_coef = sum(client.dataset_size * w[0] for client, w in zip(self.clients, aggregated_weights)) / self.dataset_size
            avg_intercept = sum(client.dataset_size * w[1] for client, w in zip(self.clients, aggregated_weights)) / self.dataset_size
            avg_intercept = sum(w[1] for w in aggregated_weights) / total_clients
            self.model.coef_ = avg_coef
            self.model.intercept_ = avg_intercept
        return self.propagate_global_model_to_clients(selected_clients)

    def propagate_global_model_to_clients(self, selected_clients):
        """Sends the current global model weights to all clients."""
        if self.method in ['CIFAR10', 'MNIST']:
            weights = self.model.get_weights()
            for client in self.clients:
                client.set_weights(weights)
        else:
            lr_params = (self.model.coef_, self.model.intercept_)
            for client in self.clients:
                client.set_weights(lr_params) # CNN specific, can be None for LR context
        return self.local_train(selected_clients)

    def local_train(self, selected_clients):
        for client in selected_clients:
            client.local_train()
        return self.evaluate_accuracy()
    
    def evaluate_accuracy(self):
        """Evaluates the global CNN model on the global test set."""
        if self.method in ['CIFAR10', 'MNIST']:
            loss, accuracy, precision, recall = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        else:
            """Evaluates the global LR model on the global test set for classification."""
            y_pred_continuous = self.model.predict(self.X_test)
    
            # Round predictions to get class labels (assumes labels are integers)
            y_pred_classes = np.round(y_pred_continuous).astype(int)
            y_true_classes = self.y_test.astype(int)

            # Clip to valid class range (e.g., 0-9) if needed
            y_pred_classes = np.clip(y_pred_classes, 0, NUM_CLASSES - 1)

            # Calculate metrics
            loss = mean_squared_error(self.y_test, y_pred_continuous)
            accuracy = accuracy_score(y_true_classes, y_pred_classes)
            precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
            recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

        return loss, accuracy * 100, precision, recall
    
    def cs_ucb(self, training_time):
    #"""CS_UCB algorithm for i.i.d dataset partition to clients
    #input: server, method in ['CIFAR10', 'MNIST', 'LR'], training_time: for how long do you want to train
    #output: accuracy(list to time), loss(list to time), times"""
        time = 0
        times = []
        K_tag = self.clients
        K_2_tag = list()
        if self.case == 'non_i_i_d':
            raise ValueError("cs-ucb algorithm is used only for i.i.d case.")
        reward = 0
        rewards = []
        optimal_reward = 0
        regret = []
        loss = []
        accuracy = []
        flag = False
        while time < training_time:
            if self.method == 'LR':
                all_client_combinations = list(itertools.combinations(self.clients, self.m))
                opt_value = float('-inf')
                for clients in all_client_combinations:
                    sorted_clients = sorted(clients, key=lambda client: client.t)
                    T_max_optimal = (sorted_clients[-1]).t
                    miu = 0
                    opt_value_temp = 0
                    for client in sorted_clients:
                        miu += (client.t)
                    miu = (1/self.m)*miu
                    opt_value_temp = -miu/T_max_optimal + 1
                    if opt_value_temp > opt_value:
                        opt_value = opt_value_temp
                opt_value = -opt_value*T_max_optimal + T_max_optimal
                optimal_reward += opt_value
            T_max = 0
            for client in self.clients:
                if client.t > T_max:
                    T_max = client.t
            for client in self.clients:
                client.cs_ucb_r = -(client.t)/(T_max)+1
            selected_clients = list()
            if len(K_tag) != 0:
                if len(K_tag) < self.m:
                    random_subset = random.sample(K_2_tag, self.m-len(K_tag))
                    selected_clients = K_tag + random_subset
                    K_tag = list()
                    if len(selected_clients) != self.m:
                        raise ValueError("error in cs-ucb algorith, selected_clients != m.")
                else:
                    selected_clients = random.sample(K_tag, self.m)
                    K_tag = set(K_tag) - set(selected_clients)
                    K_2_tag = set(K_2_tag) | set(selected_clients)
                    K_tag = list(K_tag)
                    K_2_tag = list(K_2_tag)
            else:
                for client in K_2_tag:
                    client.cs_ucb_s = (client.cs_ucb_y)+math.sqrt((((self.m)+1)*math.log(time))/(client.cs_ucb_z))
                sorted_clients = sorted(K_2_tag, key=lambda client: client.cs_ucb_s, reverse=True)
                selected_clients = sorted_clients[:self.m]
            for client in selected_clients:
                client.cs_ucb_y = ((client.cs_ucb_y)*(client.cs_ucb_z)+client.cs_ucb_r)/(client.cs_ucb_z + 1)
                client.cs_ucb_z += 1
            if not flag:
                model_loss, model_accuracy, _, _ = self.update_model(selected_clients=selected_clients)
                if model_accuracy > 90:
                    flag = True
            else:
                model_loss = loss[-1]
                model_accuracy = accuracy[-1]
            loss.append(model_loss)
            accuracy.append(model_accuracy)
            sorted_clients_t = sorted(selected_clients, key=lambda client: client.t, reverse=True)
            time += (sorted_clients_t[0]).t
            times.append(time)
            if self.method == 'LR':
                T_max = (sorted_clients_t[0]).t
                r_mean = 0
                for client in selected_clients:
                    T_hat = (client.t)/T_max
                    r_mean += T_hat
                r_mean = -(1/(self.m))*r_mean+1
                rewards.append(r_mean)
                reward_mean = sum(rewards)/len(rewards)
                reward += reward_mean
                regret.append(optimal_reward - reward)
        if self.method == 'LR':
            return loss, regret, times
        else:
            return loss, accuracy, times

    def cs_ucb_q(self, training_time, beta=0.5):
        if self.case != 'non_i_i_d':
            raise ValueError("The cs-ucb-q algorithm is designed only for non i.i.d case.")
        time = 0
        times = []
        reward = 0
        rewards = []
        optimal_reward = 0
        regret = []
        loss = []
        accuracy = []
        flag = False
        while time < training_time:
            if self.method == 'LR':
                all_client_combinations = list(itertools.combinations(self.clients, self.m))
                opt_value = float('-inf')
                for clients in all_client_combinations:
                    sorted_clients = sorted(clients, key=lambda client: client.t)
                    T_max_optimal = (sorted_clients[-1]).t
                    miu = 0
                    opt_value_temp = 0
                    for client in sorted_clients:
                        miu += (client.t)
                    miu = (1/self.m)*miu
                    opt_value_temp = -miu/T_max_optimal + 1
                    if opt_value_temp > opt_value:
                        opt_value = opt_value_temp
                opt_value = -opt_value*T_max_optimal + T_max_optimal
                optimal_reward += opt_value
            T_max = 0
            for client in self.clients:
                if client.t > T_max:
                    T_max = client.t
            for client in self.clients:
                client.cs_ucb_q_r = -(client.t)/(T_max)+1
            for client in self.clients:
                client.cs_ucb_q_c = (client.dataset_size)/(self.dataset_size)
                if client.cs_ucb_q_z > 0:
                    client.cs_ucb_q_y_hat = (client.cs_ucb_q_y) + math.sqrt(2*math.log(time+1)/(client.cs_ucb_q_z))
                    if client.cs_ucb_q_y_hat < 1:
                        client.cs_ucb_q_y_hat = 1
                else:
                    client.cs_ucb_q_y_hat = 1
                client.cs_ucb_q_D = client.cs_ucb_q_D + client.cs_ucb_q_c - client.cs_ucb_q_b
                if client.cs_ucb_q_D < 0:
                    client.cs_ucb_q_D = 0
                client.cs_ucb_q_s = (1 - beta)*(client.cs_ucb_q_y_hat)+(beta)*client.cs_ucb_q_D
                client.cs_ucb_q_b = 0
            sorted_clients = sorted(self.clients, key=lambda client: client.cs_ucb_q_s, reverse=True)
            selected_clients = sorted_clients[:self.m]
            for client in selected_clients:
                client.cs_ucb_q_b = 1
                client.cs_ucb_q_y = ((client.cs_ucb_q_y)*(client.cs_ucb_q_z)+(client.cs_ucb_q_r))/((client.cs_ucb_q_z)+1)
                client.cs_ucb_q_z += 1
            if not flag:
                model_loss, model_accuracy, _, _ = self.update_model(selected_clients=selected_clients)
                if model_accuracy > 90:
                    flag = True
            else:
                model_loss = loss[-1]
                model_accuracy = accuracy[-1]
            loss.append(model_loss)
            accuracy.append(model_accuracy)
            sorted_clients_t = sorted(selected_clients, key=lambda client: client.t, reverse=True)
            time += (sorted_clients_t[0]).t
            times.append(time)
            if self.method == 'LR':
                T_max = (sorted_clients_t[0]).t
                r_mean = 0
                for client in selected_clients:
                    T_hat = (client.t)/T_max
                    r_mean += T_hat
                r_mean = -(1/(self.m))*r_mean+1
                rewards.append(r_mean)
                reward_mean = sum(rewards)/len(rewards)
                reward += reward_mean
                regret.append(optimal_reward - reward)
        if self.method == 'LR':
            return loss, regret, times
        else:
            return loss, accuracy, times

    def rbcs_f(self, training_time, alpha=0.1, beta=0.15, lamda=1, V=5):
        time = 0
        times = []
        reward = 0
        rewards = []
        optimal_reward = 0
        optimal_rewards = []
        regret = []
        loss = []
        accuracy = []
        flag = False
        total_reward = 0
        for client in self.clients:
            client.rbcs_f_H = lamda*client.rbcs_f_H
        while time < training_time:
            if self.method == 'LR':
                optimal_selection = sorted(self.clients, key=lambda client: client.t)
                optimal_selection = optimal_selection[:(self.m)]
                optimal_reward = (optimal_selection[-1]).t
            for client in self.clients:
                inv_H = np.linalg.inv(client.rbcs_f_H)
                client.rbcs_f_theta = inv_H@(client.rbcs_f_b)
                client.rbcs_f_T_hat = (client.rbcs_f_c)@(client.rbcs_f_b)
                client.rbcs_f_T = client.rbcs_f_T_hat - alpha*math.sqrt((client.rbcs_f_c)@(inv_H)@(client.rbcs_f_c))
            selected_clients = []
            z = float('inf')
            for client in self.clients:
                S = []
                T_max = 0
                z_hat = 0
                for client_hat in self.clients:
                    if client_hat.rbcs_f_T <= client.rbcs_f_T:
                        S.append(client_hat)
                        if client_hat.t > T_max:
                            T_max = client_hat.t
                        z_hat += client_hat.rbcs_f_z
                    if len(S) == self.m:
                        z_hat += V*T_max
                        break
                if z_hat < z:
                    selected_clients = S
                    z = z_hat
            
            for client in self.clients:
                if client in selected_clients:
                    client.rbcs_f_x = 1
                else:
                    client.rbcs_f_x = 0
                client.rbcs_f_z = client.rbcs_f_z + beta - client.rbcs_f_x
                if client.rbcs_f_z < 0:
                    client.rbcs_f_z = 0
                client.rbcs_f_H = client.rbcs_f_H + (client.rbcs_f_x)*np.outer(client.rbcs_f_c, client.rbcs_f_c)
                client.rbcs_f_b = client.rbcs_f_b + (client.rbcs_f_x)*(client.t)*(client.rbcs_f_c)
            if self.method == 'LR':
                selected_clients = sorted(selected_clients, key=lambda client: client.t)
                T_max_reward = (selected_clients[-1]).t
                rewards.append(T_max_reward)
                reward = sum(rewards)/len(rewards)
                optimal_rewards.append(optimal_reward)
                optimal_reward = sum(optimal_rewards)/len(optimal_rewards)
                total_reward += (reward - optimal_reward)/len(rewards)
                regret.append(total_reward)
            if not flag:
                model_loss, model_accuracy, _, _ = self.update_model(selected_clients=selected_clients)
                if model_accuracy > 90:
                    flag = True
            else:
                model_loss = loss[-1]
                model_accuracy = accuracy[-1]
            loss.append(model_loss)
            accuracy.append(model_accuracy)
            sorted_clients_t = sorted(selected_clients, key=lambda client: client.t, reverse=True)
            time += (sorted_clients_t[0]).t
            times.append(time)
        if self.method == 'LR':
            return loss, regret, times
        else:
            return loss, accuracy, times

    def power_of_choice(self, training_time):
        time = 0
        times = []
        loss = []
        accuracy = []
        flag = False
        while time < training_time:
            sorted_clients = sorted(self.clients, key=lambda client: client.t)
            selected_clients = sorted_clients[:(self.m)]
            if not flag:
                model_loss, model_accuracy, _, _ = self.update_model(selected_clients=selected_clients)
                if model_accuracy > 90:
                    flag = True
            else:
                model_loss = loss[-1]
                model_accuracy = accuracy[-1]
            loss.append(model_loss)
            accuracy.append(model_accuracy)
            sorted_clients_t = sorted(selected_clients, key=lambda client: client.t, reverse=True)
            time += (sorted_clients_t[0]).t
            times.append(time)
        return loss, accuracy, times

    def random_selection(self, training_time):
        time = 0
        times = []
        probabilities = []
        loss = []
        accuracy = []
        flag = False
        for client in self.clients:
            client.probability = (client.dataset_size)/(self.dataset_size)
            probabilities.append(client.probability)
        while time < training_time:
            selected_clients = random.choices(
                population=self.clients,
                weights=probabilities,
                k=(self.m)
            )
            if not flag:
                model_loss, model_accuracy, _, _ = self.update_model(selected_clients=selected_clients)
                if model_accuracy > 90:
                    flag = True
            else:
                model_loss = loss[-1]
                model_accuracy = accuracy[-1]
            loss.append(model_loss)
            accuracy.append(model_accuracy)
            sorted_clients_t = sorted(selected_clients, key=lambda client: client.t, reverse=True)
            time += (sorted_clients_t[0]).t
            times.append(time)
        return loss, accuracy, times

    def bsfl(self, training_time, alpha=3, beta=1.2):
        def sign(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0
        time = 0
        times = []
        loss = []
        accuracy = []
        regret = []
        reward = 0
        rewards = []
        opt_reward = 0
        opt_rewards = []
        flag = False
        regret_val = 0
        while time < training_time:
            if self.method == 'LR':
                all_client_combinations = list(itertools.combinations(self.clients, self.m))
                opt_value = float('-inf')
                sorted_clients = sorted(self.clients, key=lambda client: client.t)
                T_min_optimal = (sorted_clients[0]).t  
                for clients in all_client_combinations:
                    miu = 0
                    for client in clients:
                        miu += client.t
                    miu = (1/self.m)*miu
                    miu = T_min_optimal/miu
                    opt_value_temp = miu
                    for client in clients:
                        opt_value_temp += (alpha/(self.m))*(client.bsfl_g)
                        if opt_value_temp > opt_value:
                            opt_value = opt_value_temp
            for client in self.clients:
                if self.case == 'i_i_d':
                    x = (self.m)/(self.K)-(client.bsfl_c)/(time+1)
                else:
                    x = ((self.m)*(client.dataset_size))/(self.dataset_size)-(client.bsfl_c)/(time+1)
                client.bsfl_g = (abs(x)**beta)*sign(x)
            if time == 0:
                selected_clients = self.clients[:(self.m)]
            else:
                sorted_clients = sorted(selected_clients, key=lambda c: c.bsfl_ucb)
                suggested_client = sorted_clients[0]
                sorted_clients.pop(0)
                A = float('-inf')
                for client in self.clients:
                    sorted_clients.insert(0, client)
                    update_clients = sorted(sorted_clients, key=lambda c: c.bsfl_ucb)
                    A_hat = (update_clients[0]).bsfl_ucb + (alpha/(self.m))*sum(c.bsfl_g for c in update_clients)
                    if A_hat > A:
                        A = A_hat
                        selected_clients = update_clients
                    sorted_clients.pop(0)
                sorted_clients.insert(0, suggested_client)
                sorted_clients = sorted(sorted_clients, key=lambda c: c.bsfl_g)
                for client in self.clients:
                    sorted_clients.insert(0, client)
                    update_clients = sorted(sorted_clients, key=lambda c: c.bsfl_ucb)
                    A_hat = (update_clients[0]).bsfl_ucb +(alpha/(self.m))*sum(c.bsfl_g for c in update_clients)
                    if A_hat > A:
                        A = A_hat
                        selected_clients = update_clients
                    sorted_clients.pop(0)
            if not flag:
                model_loss, model_accuracy, _, _ = self.update_model(selected_clients=selected_clients)
                if model_accuracy > 90:
                    flag = True
            else:
                model_loss = loss[-1]
                model_accuracy = accuracy[-1]
            loss.append(model_loss)
            accuracy.append(model_accuracy)
            sorted_clients_t = sorted(selected_clients, key=lambda client: client.t, reverse=True)
            time += (sorted_clients_t[0]).t
            times.append(time)
            T_min = float('inf')
            T_max = float('-inf')
            for client in selected_clients:
                if client.t < T_min:
                    T_min = client.t
                if client.t > T_max:
                    T_max = client.t
            for client in selected_clients:
                client.bsfl_miu = ((client.bsfl_miu)*(client.bsfl_c)+(T_min/(client.t))/((client.bsfl_c)+1))
                client.bsfl_c += 1
                client.bsfl_ucb = client.bsfl_miu + math.sqrt((((self.m)+1)*math.log(time+1))/(client.bsfl_c))
            if self.method == 'LR':
                reward_val = T_min/T_max + (alpha/(self.m))*sum(c.bsfl_g for c in selected_clients)
                rewards.append(reward_val)
                reward = sum(rewards)/len(rewards)
                opt_rewards.append(opt_value)
                opt_reward = sum(opt_rewards)/len(opt_rewards)
                regret_val += opt_reward - reward
                regret.append(regret_val)
        if self.method == 'LR':
            return loss, regret, times
        else:
            return loss, accuracy, times

def normalize_loss_by_max(loss_list):
    if not loss_list:
        return []
    max_loss = max(loss_list) 
    if max_loss == 0:
        print("Warning: max loss is 0.")
        return [0.0 for _ in loss_list]
        
    normalized_list = [l / max_loss for l in loss_list]
    return normalized_list

if __name__ == '__main__':
    #------------------------------------linear regression i.i.d case--------------------------------------
    server = Server(K=LR_CLIENTS, m=LR_ACTIVE, method='LR', server_case='i_i_d', learning_rate=0.005)
    print("starting algorithms for LR case i.i.d with syntetic data")
    print("starting CS-UCB algorithm..")
    cs_ucb_loss, cs_ucb_regret, cs_ucb_times = server.cs_ucb(training_time=200)
    cs_ucb_loss = normalize_loss_by_max(cs_ucb_loss)
    print("CS-UCB algorithm is done.")
    server.reset_model()
    print("starting RBCS-F algorithm..")
    rbcs_loss, rbcs_regret, rbcs_times = server.rbcs_f(training_time=200)
    rbcs_loss = normalize_loss_by_max(rbcs_loss)
    print("RBCS-F algorithm is done.")
    server.reset_model()
    print("starting BSFL algorithm..")
    bfsl_loss, bfsl_regret, bsfl_times = server.bsfl(training_time=200, alpha=3, beta=1.2)
    bfsl_loss = normalize_loss_by_max(bfsl_loss)
    print("BSFL algorithm is done.")
    server.reset_model()
    print("starting Power-Of-Choice algorithm..")
    poc_loss, _ , poc_times= server.power_of_choice(training_time=200)
    poc_loss = normalize_loss_by_max(poc_loss)
    print("Power-Of-Choice algorithm is done.")
    server.reset_model()
    print("starting Random-Selection algorithm..")
    random_loss, _, random_times = server.random_selection(training_time=200)
    random_loss = normalize_loss_by_max(random_loss)
    print("Random-Selection algorithm is done.")

    # regret vs iteration figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_regret, label='CS-UCB', color='blue')
    plt.plot(rbcs_times, rbcs_regret, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_regret, label='BSFL', color='purple')

    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.title('Linear regression i.i.d Regret vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # loss vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_loss, label='CS-UCB', color='blue')
    plt.plot(rbcs_times, rbcs_loss, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_loss, label='BSFL', color='purple')
    plt.plot(poc_times, poc_loss, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_loss, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Linear regression i.i.d Loss vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #----------------------------------------Linear Regression non i.i.d case-----------------------------
    del server
    server = Server(K=LR_CLIENTS, m=LR_ACTIVE, method='LR', server_case='non_i_i_d', learning_rate=0.003)
    print("starting algorithms for LR case non i.i.d with syntetic data")
    print("starting CS-UCB-Q algorithm..")
    cs_ucb_loss, cs_ucb_regret, cs_ucb_times = server.cs_ucb_q(training_time=130, beta=0.5)
    cs_ucb_loss = normalize_loss_by_max(cs_ucb_loss)
    print("CS-UCB-Q algorithm is done.")
    server.reset_model()
    print("starting RBCS-F algorithm..")
    rbcs_loss, rbcs_regret, rbcs_times = server.rbcs_f(training_time=130)
    rbcs_loss = normalize_loss_by_max(rbcs_loss)
    print("RBCS-F algorithm is done.")
    server.reset_model()
    print("starting BSFL algorithm..")
    bfsl_loss, bfsl_regret, bsfl_times = server.bsfl(training_time=130, alpha=3, beta=1.2)
    bfsl_loss = normalize_loss_by_max(bfsl_loss)
    print("BSFL algorithm is done.")
    server.reset_model()
    print("starting Power-Of-Choice algorithm..")
    poc_loss, _, poc_times= server.power_of_choice(training_time=130)
    poc_loss = normalize_loss_by_max(poc_loss)
    print("Power-Of-Choice algorithm is done.")
    server.reset_model()
    print("starting Random-Selection algorithm..")
    random_loss, _, random_times = server.random_selection(training_time=130)
    random_loss = normalize_loss_by_max(random_loss)
    print("Random-Selection algorithm is done.")

    # regret vs iteration figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_regret, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_regret, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_regret, label='BSFL', color='purple')

    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.title('Linear Regression non i.i.d Regret vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # loss vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_loss, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_loss, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_loss, label='BSFL', color='purple')
    plt.plot(poc_times, poc_loss, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_loss, label='Random-Selection', color='red')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Linear Regression non i.i.d Loss vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    #------------------------- CNN CIFAR10 i.i.d case ------------------------------------
    del server
    server = Server(K=CNN_CLIENTS, m=CNN_ACTIVE, method='CIFAR10', server_case='i_i_d', learning_rate=0.00009)
    print("starting algorithms for CNN case i.i.d with CIFAR10 dataset")
    print("starting CS-UCB algorithm..")
    cs_ucb_loss, cs_ucb_accuracy, cs_ucb_times = server.cs_ucb(training_time=400)
    cs_ucb_loss = normalize_loss_by_max(cs_ucb_loss)
    print("CS-UCB algorithm is done.")
    server.reset_model()
    print("starting RBCS-F algorithm..")
    rbcs_loss, rbcs_accuracy, rbcs_times = server.rbcs_f(training_time=400)
    rbcs_loss = normalize_loss_by_max(rbcs_loss)
    print("RBCS-F algorithm is done.")
    server.reset_model()
    print("starting BSFL algorithm..")
    bfsl_loss, bfsl_accuracy, bsfl_times = server.bsfl(training_time=400, alpha=2, beta=1)
    bfsl_loss = normalize_loss_by_max(bfsl_loss)
    print("BSFL algorithm is done.")
    server.reset_model()
    print("starting Power-Of-Choice algorithm..")
    poc_loss, poc_accuracy, poc_times= server.power_of_choice(training_time=400)
    poc_loss = normalize_loss_by_max(poc_loss)
    print("Power-Of-Choice algorithm is done.")
    server.reset_model()
    print("starting Random-Selection algorithm..")
    random_loss, random_accuracy, random_times = server.random_selection(training_time=400)
    random_loss = normalize_loss_by_max(random_loss)
    print("Random-Selection algorithm is done.")

    # accuracy vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_accuracy, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_accuracy, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_accuracy, label='BSFL', color='purple')
    plt.plot(poc_times, poc_accuracy, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_accuracy, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('CNN CIFAR10 i.i.d Accuracy vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # loss vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_loss, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_loss, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_loss, label='BSFL', color='purple')
    plt.plot(poc_times, poc_loss, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_loss, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('CNN CIFAR10 dataset i.i.d Loss vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #------------------------- CNN CIFAR10 non i.i.d case ------------------------------------
    del server
    server = Server(K=CNN_CLIENTS, m=CNN_ACTIVE, method='CIFAR10', server_case='non_i_i_d', learning_rate=0.000005)
    print("starting algorithms for CNN case non i.i.d with CIFAR10 dataset")
    print("starting CS-UCB-Q algorithm..")
    cs_ucb_loss, cs_ucb_accuracy, cs_ucb_times = server.cs_ucb_q(training_time=200)
    cs_ucb_loss = normalize_loss_by_max(cs_ucb_loss)
    print("CS-UCB-Q algorithm is done.")
    server.reset_model()
    print("starting RBCS-F algorithm..")
    rbcs_loss, rbcs_accuracy, rbcs_times = server.rbcs_f(training_time=200)
    rbcs_loss = normalize_loss_by_max(rbcs_loss)
    print("RBCS-F algorithm is done.")
    server.reset_model()
    print("starting BSFL algorithm..")
    bfsl_loss, bfsl_accuracy, bsfl_times = server.bsfl(training_time=200, alpha=3, beta=1.2)
    bfsl_loss = normalize_loss_by_max(bfsl_loss)
    print("BSFL algorithm is done.")
    server.reset_model()
    print("starting Power-Of-Choice algorithm..")
    poc_loss, poc_accuracy, poc_times= server.power_of_choice(training_time=200)
    poc_loss = normalize_loss_by_max(poc_loss)
    print("Power-Of-Choice algorithm is done.")
    server.reset_model()
    print("starting Random-Selection algorithm..")
    random_loss, random_accuracy, random_times = server.random_selection(training_time=200)
    random_loss = normalize_loss_by_max(random_loss)
    print("Random-Selection algorithm is done.")

    # accuracy vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_accuracy, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_accuracy, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_accuracy, label='BSFL', color='purple')
    plt.plot(poc_times, poc_accuracy, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_accuracy, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('CNN CIFAR10 dataset non i.i.d Accuracy vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # loss vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_loss, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_loss, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_loss, label='BSFL', color='purple')
    plt.plot(poc_times, poc_loss, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_loss, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('CNN CIFAR10 dataset non i.i.d Loss vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #------------------------- CNN Fashion-MNIST i.i.d case ------------------------------------
    del server
    server = Server(K=CNN_CLIENTS, m=CNN_ACTIVE, method='MNIST', server_case='i_i_d', learning_rate=0.00005)
    print("starting algorithms for CNN case i.i.d with Fashion-MNIST dataset")
    print("starting CS-UCB algorithm..")
    cs_ucb_loss, cs_ucb_accuracy, cs_ucb_times = server.cs_ucb(training_time=400)
    cs_ucb_loss = normalize_loss_by_max(cs_ucb_loss)
    print("CS-UCB algorithm is done.")
    server.reset_model()
    print("starting RBCS-F algorithm..")
    rbcs_loss, rbcs_accuracy, rbcs_times = server.rbcs_f(training_time=400)
    rbcs_loss = normalize_loss_by_max(rbcs_loss)
    print("RBCS-F algorithm is done.")
    server.reset_model()
    print("starting BSFL algorithm..")
    bfsl_loss, bfsl_accuracy, bsfl_times = server.bsfl(training_time=400, alpha=2, beta=1)
    bfsl_loss = normalize_loss_by_max(bfsl_loss)
    print("BSFL algorithm is done.")
    server.reset_model()
    print("starting Power-Of-Choice algorithm..")
    poc_loss, poc_accuracy, poc_times= server.power_of_choice(training_time=400)
    poc_loss = normalize_loss_by_max(poc_loss)
    print("Power-Of-Choice algorithm is done.")
    server.reset_model()
    print("starting Random-Selection algorithm..")
    random_loss, random_accuracy, random_times = server.random_selection(training_time=400)
    random_loss = normalize_loss_by_max(random_loss)
    print("Random-Selection algorithm is done.")

    # accuracy vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_accuracy, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_accuracy, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_accuracy, label='BSFL', color='purple')
    plt.plot(poc_times, poc_accuracy, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_accuracy, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('CNN Fashion-MNIST i.i.d Accuracy vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # loss vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_loss, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_loss, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_loss, label='BSFL', color='purple')
    plt.plot(poc_times, poc_loss, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_loss, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('CNN Fashion-MNIST dataset i.i.d Loss vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #------------------------- CNN Fashin-MNIST non i.i.d case ------------------------------------
    del server
    server = Server(K=CNN_CLIENTS, m=CNN_ACTIVE, method='MNIST', server_case='non_i_i_d', learning_rate=0.00001)
    print("starting algorithms for CNN case non i.i.d with Fashion-MNIST dataset")
    print("starting CS-UCB-Q algorithm..")
    cs_ucb_loss, cs_ucb_accuracy, cs_ucb_times = server.cs_ucb_q(training_time=200)
    cs_ucb_loss = normalize_loss_by_max(cs_ucb_loss)
    print("CS-UCB-Q algorithm is done.")
    server.reset_model()
    print("starting RBCS-F algorithm..")
    rbcs_loss, rbcs_accuracy, rbcs_times = server.rbcs_f(training_time=200)
    rbcs_loss = normalize_loss_by_max(rbcs_loss)
    print("RBCS-F algorithm is done.")
    server.reset_model()
    print("starting BSFL algorithm..")
    bfsl_loss, bfsl_accuracy, bsfl_times = server.bsfl(training_time=200, alpha=3, beta=1.2)
    bfsl_loss = normalize_loss_by_max(bfsl_loss)
    print("BSFL algorithm is done.")
    server.reset_model()
    print("starting Power-Of-Choice algorithm..")
    poc_loss, poc_accuracy, poc_times= server.power_of_choice(training_time=200)
    poc_loss = normalize_loss_by_max(poc_loss)
    print("Power-Of-Choice algorithm is done.")
    server.reset_model()
    print("starting Random-Selection algorithm..")
    random_loss, random_accuracy, random_times = server.random_selection(training_time=200)
    random_loss = normalize_loss_by_max(random_loss)
    print("Random-Selection algorithm is done.")

    # accuracy vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_accuracy, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_accuracy, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_accuracy, label='BSFL', color='purple')
    plt.plot(poc_times, poc_accuracy, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_accuracy, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('CNN Fashion-MNIST dataset non i.i.d Accuracy vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # loss vs time figure
    plt.figure(figsize=(12, 6))
    plt.plot(cs_ucb_times, cs_ucb_loss, label='CS-UCB-Q', color='blue')
    plt.plot(rbcs_times, rbcs_loss, label='RBCS-F', color='green')
    plt.plot(bsfl_times, bfsl_loss, label='BSFL', color='purple')
    plt.plot(poc_times, poc_loss, label='Power-Of-Choice', color='orange')
    plt.plot(random_times, random_loss, label='Random-Selection', color='red')

    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('CNN Fashion-MNIST dataset non i.i.d Loss vs Time for Different Algorithms')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()