import copy
import numpy as np
from environment_ag_20220402 import Env
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from tensorflow.keras.optimizers import Adam

# Paths to input files
network_file = "simple_network.txt"
train_file = "simple_train.txt"

# Initialize environment
env = Env(network_file, train_file,600)

n_blocks = env.n_blocks
state_size = 2 + n_blocks * n_blocks + 9 * n_blocks
action_size = n_blocks + 1


# Rebuild the model architecture
def build_model():
    model = Sequential()
    model.add(Dense(1024,
                    input_dim=state_size,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    kernel_initializer="random_uniform",
                    bias_initializer="zeros"))
    model.add(Dense(512,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Dense(256,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Dense(128,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Dense(action_size,
                    activation='linear',
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model


# Load the model architecture and weights
model = build_model()
model.load_weights("./20220402_DQN_49_12_9600.h5")

print("Starting inference...")

state = env.reset()
done = False
total_reward = 0
steps = 0

while not done:
    normalized_state = np.reshape(state, [1, state_size])
    q_values = model.predict(normalized_state)[0]

    # Greedy action
    action = np.argmax(q_values)

    next_state, reward, is_terminal = env.step(action)

    total_reward += reward
    steps += 1
    # --- New print statements for more detailed output ---
    print("--------------------------------------------------")
    print(f"Step {steps}:")
    print(f"  Current state shape: {state.shape}")
    print(f"  Predicted Q-values: {np.round(q_values, 2)}")
    print(f"  Chosen Action (Block ID): {action}")
    print(f"  Reward received: {reward}")
    print(f"  Is terminal? {is_terminal}")
    print("--------------------------------------------------")
    state = copy.deepcopy(next_state)

print(f"\nInference finished in {steps} steps. Total reward: {total_reward}")