# API Documentation

## Neural Network (NN) Class

### Overview
The `NN` class implements a modern, configurable neural network with advanced features for deep reinforcement learning.

### Constructors

#### `NN(int inputs, int hiddens, int outputs)`
Creates a neural network with default configuration.

**Parameters:**
- `inputs` (int): Number of input neurons
- `hiddens` (int): Number of hidden neurons per layer
- `outputs` (int): Number of output neurons

**Example:**
```csharp
NN network = new NN(14, 64, 66);
```

#### `NN(NetworkConfig config)`
Creates a neural network with custom configuration.

**Parameters:**
- `config` (NetworkConfig): Configuration object with all network parameters

**Example:**
```csharp
NN.NetworkConfig config = new NN.NetworkConfig
{
    numInputs = 14,
    numHiddens = 64,
    numOutputs = 66,
    numHiddenLayers = 3,
    learningRate = 0.001f,
    useBatchNormalization = true,
    activationType = NN.ActivationType.ReLU
};
NN network = new NN(config);
```

### Core Methods

#### `List<float> calcNet(float[] input)`
Performs forward pass through the network.

**Parameters:**
- `input` (float[]): Input vector

**Returns:**
- `List<float>`: Network output (Q-values)

**Throws:**
- `ArgumentException`: If input size doesn't match network input size

**Example:**
```csharp
float[] state = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f};
List<float> qValues = network.calcNet(state);
int bestAction = qValues.IndexOf(qValues.Max());
```

#### `void Train(Transition transition)`
Trains the network on a single transition.

**Parameters:**
- `transition` (Transition): State-action-reward transition

**Example:**
```csharp
NN.Transition transition = new NN.Transition
{
    state = currentState,
    action = selectedAction,
    reward = receivedReward
};
network.Train(transition);
```

### Performance Methods

#### `float GetCurrentLoss()`
Gets the current training loss.

**Returns:**
- `float`: Current loss value

#### `float GetAverageLoss()`
Gets the moving average loss.

**Returns:**
- `float`: Average loss over recent training steps

#### `List<float> GetLossHistory()`
Gets the complete loss history.

**Returns:**
- `List<float>`: List of all recorded loss values

#### `int GetTrainingSteps()`
Gets the number of training steps performed.

**Returns:**
- `int`: Total training steps

#### `string GetNetworkInfo()`
Gets a summary of network information.

**Returns:**
- `string`: Formatted network information

### Persistence Methods

#### `void Save(StreamWriter writer)`
Saves the network to a stream.

**Parameters:**
- `writer` (StreamWriter): Output stream

**Example:**
```csharp
using (StreamWriter writer = new StreamWriter("model.txt"))
{
    network.Save(writer);
}
```

#### `void Load(StreamReader reader)`
Loads the network from a stream.

**Parameters:**
- `reader` (StreamReader): Input stream

**Example:**
```csharp
using (StreamReader reader = new StreamReader("model.txt"))
{
    network.Load(reader);
}
```

## Reinforcement Learning (RL) Class

### Overview
The `RL` class implements advanced deep Q-learning with modern techniques like Double DQN, target networks, and prioritized experience replay.

### Core Methods

#### `int Act(float[] state)`
Selects an action using epsilon-greedy policy.

**Parameters:**
- `state` (float[]): Current state vector

**Returns:**
- `int`: Selected action index

**Example:**
```csharp
float[] state = GetCurrentState();
int action = agent.Act(state);
ExecuteAction(action);
```

#### `void Observe(Transition transition)`
Learns from a state-action-reward transition.

**Parameters:**
- `transition` (Transition): Complete transition with next state and terminal flag

**Example:**
```csharp
RL.Transition transition = new RL.Transition
{
    state = currentState,
    action = selectedAction,
    reward = receivedReward,
    nextState = nextState,
    isTerminal = isEpisodeOver
};
agent.Observe(transition);
```

### Episode Management

#### `void StartEpisode()`
Begins a new episode, resetting episode-specific metrics.

**Example:**
```csharp
agent.StartEpisode();
// ... perform actions ...
agent.EndEpisode();
```

#### `void EndEpisode()`
Ends the current episode, updating performance statistics.

**Example:**
```csharp
if (isEpisodeOver)
{
    agent.EndEpisode();
}
```

### Persistence Methods

#### `void SaveModel()`
Saves the complete agent state to a file.

**Example:**
```csharp
agent.SaveModel(); // Saves to Application.persistentDataPath/rl_model.txt
```

#### `void LoadModel()`
Loads the complete agent state from a file.

**Example:**
```csharp
agent.LoadModel(); // Loads from Application.persistentDataPath/rl_model.txt
```

#### `void ResetAgent()`
Resets the agent to initial state.

**Example:**
```csharp
agent.ResetAgent();
```

### Performance Methods

#### `float GetEpsilon()`
Gets the current exploration rate.

**Returns:**
- `float`: Current epsilon value

#### `int GetTrainingSteps()`
Gets the total number of training steps.

**Returns:**
- `int`: Total training steps

#### `float GetCurrentReward()`
Gets the current episode reward.

**Returns:**
- `float`: Current episode reward

#### `float GetAverageReward()`
Gets the average reward over recent episodes.

**Returns:**
- `float`: Average reward

#### `float GetAverageLength()`
Gets the average episode length.

**Returns:**
- `float`: Average episode length

#### `int GetExperienceBufferSize()`
Gets the current size of the experience replay buffer.

**Returns:**
- `int`: Number of stored experiences

#### `string GetAgentInfo()`
Gets a summary of agent information.

**Returns:**
- `string`: Formatted agent information

## Configuration Classes

### NetworkConfig

#### Properties
- `numInputs` (int): Number of input neurons
- `numHiddens` (int): Number of hidden neurons per layer
- `numOutputs` (int): Number of output neurons
- `numHiddenLayers` (int): Number of hidden layers
- `learningRate` (float): Learning rate for training
- `batchSize` (int): Batch size for training
- `experienceReplaySize` (int): Size of experience replay buffer
- `useBatchNormalization` (bool): Enable batch normalization
- `useDropout` (bool): Enable dropout regularization
- `dropoutRate` (float): Dropout rate (0.0-1.0)
- `activationType` (ActivationType): Activation function type
- `weightInitType` (WeightInitType): Weight initialization method
- `optimizerType` (OptimizerType): Optimizer type

### RLConfig

#### Properties
- `epsilonStart` (float): Initial exploration rate
- `epsilonEnd` (float): Final exploration rate
- `epsilonDecay` (float): Exploration decay rate
- `epsilonDecaySteps` (int): Steps for epsilon decay
- `learningRate` (float): Learning rate
- `gamma` (float): Discount factor
- `targetUpdateFrequency` (int): Target network update frequency
- `experienceReplaySize` (int): Experience replay buffer size
- `batchSize` (int): Training batch size
- `useDoubleDQN` (bool): Enable Double DQN
- `useTargetNetwork` (bool): Enable target network
- `usePrioritizedReplay` (bool): Enable prioritized experience replay
- `prioritizedReplayAlpha` (float): Prioritized replay alpha parameter
- `prioritizedReplayBeta` (float): Prioritized replay beta parameter
- `prioritizedReplayBetaIncrement` (float): Beta increment rate
- `networkConfig` (NetworkConfig): Neural network configuration

## Enums

### ActivationType
- `Sigmoid`: Sigmoid activation function
- `ReLU`: Rectified Linear Unit
- `LeakyReLU`: Leaky Rectified Linear Unit
- `Tanh`: Hyperbolic tangent

### WeightInitType
- `Random`: Random weight initialization
- `Xavier`: Xavier/Glorot initialization
- `He`: He initialization

### OptimizerType
- `SGD`: Stochastic Gradient Descent
- `Adam`: Adam optimizer
- `RMSprop`: RMSprop optimizer

## Data Structures

### Transition (NN)
```csharp
public struct Transition
{
    public float[] state;    // Current state
    public int action;       // Action taken
    public float reward;     // Reward received
}
```

### Transition (RL)
```csharp
public struct Transition
{
    public float[] state;      // Current state
    public int action;         // Action taken
    public float reward;       // Reward received
    public float[] nextState;  // Next state
    public bool isTerminal;    // Episode termination flag
    public float priority;     // Priority for replay (internal)
    public int index;          // Buffer index (internal)
}
```

## Error Handling

### Common Exceptions
- `ArgumentException`: Invalid input parameters
- `InvalidOperationException`: Invalid operation state
- `FileNotFoundException`: Save/load file not found
- `FormatException`: Invalid file format

### Best Practices
1. Always validate input parameters
2. Check for null references
3. Handle file I/O exceptions
4. Use try-catch blocks for critical operations
5. Log errors for debugging

## Performance Considerations

### Memory Usage
- Experience replay buffer: ~4MB for 10K transitions
- Neural network weights: ~1MB for 64x64x66 network
- Total memory footprint: ~10-20MB typical

### Training Speed
- Forward pass: ~0.1ms per sample
- Backward pass: ~0.5ms per sample
- Batch training: ~10ms per batch of 32
- Target network update: ~1ms

### Optimization Tips
1. Use appropriate batch sizes (32-128)
2. Enable batch normalization for stability
3. Use Adam optimizer for faster convergence
4. Monitor memory usage with large buffers
5. Consider async training for large networks
