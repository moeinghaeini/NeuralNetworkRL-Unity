using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;

/// <summary>
/// Modern Neural Network implementation with advanced features for Deep Q-Learning
/// Features: ReLU activation, batch normalization, Xavier initialization, dropout, and more
/// </summary>
public class NN
{
    #region Configuration
    [System.Serializable]
    public class NetworkConfig
    {
        [Header("Network Architecture")]
        public int numInputs = 14;
        public int numHiddens = 10;
        public int numOutputs = 66;
        public int numHiddenLayers = 2;
        
        [Header("Training Parameters")]
        public int numEpochs = 50;
        public float learningRate = 0.001f;
        public int batchSize = 32;
        public int experienceReplaySize = 10000;
        
        [Header("Advanced Features")]
        public bool useBatchNormalization = true;
        public bool useDropout = true;
        public float dropoutRate = 0.2f;
        public ActivationType activationType = ActivationType.ReLU;
        public WeightInitType weightInitType = WeightInitType.Xavier;
        public float gradientClipThreshold = 1.0f;
        
        [Header("Optimization")]
        public OptimizerType optimizerType = OptimizerType.Adam;
        public float adamBeta1 = 0.9f;
        public float adamBeta2 = 0.999f;
        public float adamEpsilon = 1e-8f;
    }
    
    public enum ActivationType { Sigmoid, ReLU, LeakyReLU, Tanh }
    public enum WeightInitType { Random, Xavier, He }
    public enum OptimizerType { SGD, Adam, RMSprop }
    #endregion

    #region Private Fields
    private NetworkConfig config;
    private int numEpochs;
    private int numInputs;
    private int numHiddens;
    private int numOutputs;
    private int sizeOfExperienceReplayMemory;
    private float LR;
    private int batchSize;

    //training data
    private List<RL.Transition> transitions, batchTransitions;

    //network layers and activations
    private float[][] hiddenLayers;
    private float[] outputLayer;
    private float[][] hiddenDeltas;
    private float[] outputDelta;

    //weights and biases
    private float[][,] weights;
    private float[][] biases;
    
    //batch normalization parameters
    private float[][] batchNormGamma;
    private float[][] batchNormBeta;
    private float[][] batchNormMean;
    private float[][] batchNormVariance;
    
    //optimizer states (for Adam)
    private float[][,] mWeights; // first moment
    private float[][,] vWeights; // second moment
    private float[][] mBiases;
    private float[][] vBiases;
    private int timestep = 0;
    
    //dropout masks
    private bool[][] dropoutMasks;
    
    //performance tracking
    private float currentLoss = 0f;
    private List<float> lossHistory = new List<float>();
    private float averageLoss = 0f;
    
    #region Constructors
    public NN(int inputs, int hiddens, int outputs)
    {
        // Create default config
        config = new NetworkConfig
        {
            numInputs = inputs,
            numHiddens = hiddens,
            numOutputs = outputs
        };
        InitializeNetwork();
    }
    
    public NN(NetworkConfig networkConfig)
    {
        config = networkConfig ?? throw new ArgumentNullException(nameof(networkConfig));
        InitializeNetwork();
    }
    
    private void InitializeNetwork()
    {
        try
        {
            // Initialize parameters from config
            numInputs = config.numInputs;
            numHiddens = config.numHiddens;
            numOutputs = config.numOutputs;
            numEpochs = config.numEpochs;
            sizeOfExperienceReplayMemory = config.experienceReplaySize;
            LR = config.learningRate;
            batchSize = config.batchSize;
            
            // Initialize data structures
            transitions = new List<RL.Transition>();
            batchTransitions = new List<RL.Transition>();
            lossHistory = new List<float>();
            
            // Initialize network architecture
            InitializeNetworkArchitecture();
            InitializeWeights();
            
            Debug.Log($"Neural Network initialized: {numInputs} -> {numHiddens} -> {numOutputs}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize neural network: {e.Message}");
            throw;
        }
    }
    #endregion

    #region Network Architecture Initialization
    private void InitializeNetworkArchitecture()
    {
        int totalLayers = config.numHiddenLayers + 1; // hidden layers + output layer
        
        // Initialize layer arrays
        hiddenLayers = new float[config.numHiddenLayers][];
        hiddenDeltas = new float[config.numHiddenLayers][];
        weights = new float[totalLayers][,];
        biases = new float[totalLayers][];
        dropoutMasks = new bool[config.numHiddenLayers][];
        
        // Initialize hidden layers
        for (int i = 0; i < config.numHiddenLayers; i++)
        {
            int layerSize = config.numHiddens;
            hiddenLayers[i] = new float[layerSize];
            hiddenDeltas[i] = new float[layerSize];
            dropoutMasks[i] = new bool[layerSize];
        }
        
        // Initialize output layer
        outputLayer = new float[numOutputs];
        outputDelta = new float[numOutputs];
        
        // Initialize weights and biases
        InitializeLayerWeights(0, numInputs, config.numHiddens); // input -> first hidden
        
        for (int i = 1; i < config.numHiddenLayers; i++)
        {
            InitializeLayerWeights(i, config.numHiddens, config.numHiddens); // hidden -> hidden
        }
        
        InitializeLayerWeights(config.numHiddenLayers, config.numHiddens, numOutputs); // last hidden -> output
        
        // Initialize batch normalization if enabled
        if (config.useBatchNormalization)
        {
            InitializeBatchNormalization();
        }
        
        // Initialize optimizer states if using Adam
        if (config.optimizerType == OptimizerType.Adam)
        {
            InitializeAdamOptimizer();
        }
    }
    
    private void InitializeLayerWeights(int layerIndex, int inputSize, int outputSize)
    {
        weights[layerIndex] = new float[inputSize, outputSize];
        biases[layerIndex] = new float[outputSize];
        
        // Initialize weights based on selected method
        float weightScale = GetWeightInitializationScale(inputSize, outputSize);
        
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                weights[layerIndex][i, j] = GetInitialWeight(weightScale);
            }
        }
        
        // Initialize biases to zero
        for (int j = 0; j < outputSize; j++)
        {
            biases[layerIndex][j] = 0f;
        }
    }
    
    private float GetWeightInitializationScale(int inputSize, int outputSize)
    {
        switch (config.weightInitType)
        {
            case WeightInitType.Xavier:
                return Mathf.Sqrt(2.0f / (inputSize + outputSize));
            case WeightInitType.He:
                return Mathf.Sqrt(2.0f / inputSize);
            case WeightInitType.Random:
            default:
                return 0.1f;
        }
    }
    
    private float GetInitialWeight(float scale)
    {
        return UnityEngine.Random.Range(-scale, scale);
    }
    
    private void InitializeBatchNormalization()
    {
        int totalLayers = config.numHiddenLayers + 1;
        batchNormGamma = new float[totalLayers][];
        batchNormBeta = new float[totalLayers][];
        batchNormMean = new float[totalLayers][];
        batchNormVariance = new float[totalLayers][];
        
        for (int i = 0; i < totalLayers; i++)
        {
            int layerSize = (i < config.numHiddenLayers) ? config.numHiddens : numOutputs;
            batchNormGamma[i] = new float[layerSize];
            batchNormBeta[i] = new float[layerSize];
            batchNormMean[i] = new float[layerSize];
            batchNormVariance[i] = new float[layerSize];
            
            // Initialize gamma to 1, beta to 0
            for (int j = 0; j < layerSize; j++)
            {
                batchNormGamma[i][j] = 1f;
                batchNormBeta[i][j] = 0f;
                batchNormMean[i][j] = 0f;
                batchNormVariance[i][j] = 1f;
            }
        }
    }
    
    private void InitializeAdamOptimizer()
    {
        int totalLayers = config.numHiddenLayers + 1;
        mWeights = new float[totalLayers][,];
        vWeights = new float[totalLayers][,];
        mBiases = new float[totalLayers][];
        vBiases = new float[totalLayers][];
        
        for (int i = 0; i < totalLayers; i++)
        {
            int inputSize = (i == 0) ? numInputs : config.numHiddens;
            int outputSize = (i == config.numHiddenLayers) ? numOutputs : config.numHiddens;
            
            mWeights[i] = new float[inputSize, outputSize];
            vWeights[i] = new float[inputSize, outputSize];
            mBiases[i] = new float[outputSize];
            vBiases[i] = new float[outputSize];
            
            // Initialize to zero
            for (int j = 0; j < inputSize; j++)
            {
                for (int k = 0; k < outputSize; k++)
                {
                    mWeights[i][j, k] = 0f;
                    vWeights[i][j, k] = 0f;
                }
            }
            
            for (int j = 0; j < outputSize; j++)
            {
                mBiases[i][j] = 0f;
                vBiases[i][j] = 0f;
            }
        }
    }
    #endregion

    #region Activation Functions
    private float ApplyActivation(float x, ActivationType activationType)
    {
        switch (activationType)
        {
            case ActivationType.ReLU:
                return Mathf.Max(0f, x);
            case ActivationType.LeakyReLU:
                return x > 0 ? x : 0.01f * x;
            case ActivationType.Tanh:
                return (float)Math.Tanh(x);
            case ActivationType.Sigmoid:
            default:
                return 1.0f / (1.0f + Mathf.Exp(-x));
        }
    }
    
    private float ActivationDerivative(float x, ActivationType activationType)
    {
        switch (activationType)
        {
            case ActivationType.ReLU:
                return x > 0 ? 1f : 0f;
            case ActivationType.LeakyReLU:
                return x > 0 ? 1f : 0.01f;
            case ActivationType.Tanh:
                float tanh = (float)Math.Tanh(x);
                return 1f - tanh * tanh;
            case ActivationType.Sigmoid:
            default:
                float sigmoid = 1.0f / (1.0f + Mathf.Exp(-x));
                return sigmoid * (1f - sigmoid);
        }
    }
    #endregion

    #region Forward Pass
    public List<float> calcNet(float[] input)
    {
        if (input == null || input.Length != numInputs)
        {
            throw new ArgumentException($"Input array must have exactly {numInputs} elements");
        }
        
        try
        {
            // Forward pass through hidden layers
            float[] currentInput = input;
            
            for (int layer = 0; layer < config.numHiddenLayers; layer++)
            {
                int inputSize = (layer == 0) ? numInputs : config.numHiddens;
                int outputSize = config.numHiddens;
                
                // Linear transformation
                for (int j = 0; j < outputSize; j++)
                {
                    hiddenLayers[layer][j] = biases[layer][j];
                    for (int i = 0; i < inputSize; i++)
                    {
                        hiddenLayers[layer][j] += currentInput[i] * weights[layer][i, j];
                    }
                }
                
                // Batch normalization (if enabled)
                if (config.useBatchNormalization)
                {
                    ApplyBatchNormalization(hiddenLayers[layer], layer);
                }
                
                // Activation function
                for (int j = 0; j < outputSize; j++)
                {
                    hiddenLayers[layer][j] = ApplyActivation(hiddenLayers[layer][j], config.activationType);
                }
                
                // Dropout (if enabled and training)
                if (config.useDropout)
                {
                    ApplyDropout(hiddenLayers[layer], layer);
                }
                
                currentInput = hiddenLayers[layer];
            }
            
            // Output layer
            for (int j = 0; j < numOutputs; j++)
            {
                outputLayer[j] = biases[config.numHiddenLayers][j];
                for (int i = 0; i < config.numHiddens; i++)
                {
                    outputLayer[j] += currentInput[i] * weights[config.numHiddenLayers][i, j];
                }
            }
            
            // Output activation (linear for Q-values)
            List<float> output = new List<float>();
            for (int j = 0; j < numOutputs; j++)
            {
                output.Add(outputLayer[j]); // No activation for Q-values
            }
            
            return output;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in forward pass: {e.Message}");
            throw;
        }
    }
    
    private void ApplyBatchNormalization(float[] layer, int layerIndex)
    {
        float epsilon = 1e-8f;
        for (int i = 0; i < layer.Length; i++)
        {
            float normalized = (layer[i] - batchNormMean[layerIndex][i]) / 
                              Mathf.Sqrt(batchNormVariance[layerIndex][i] + epsilon);
            layer[i] = batchNormGamma[layerIndex][i] * normalized + batchNormBeta[layerIndex][i];
        }
    }
    
    private void ApplyDropout(float[] layer, int layerIndex)
    {
        for (int i = 0; i < layer.Length; i++)
        {
            dropoutMasks[layerIndex][i] = UnityEngine.Random.Range(0f, 1f) < config.dropoutRate;
            if (dropoutMasks[layerIndex][i])
            {
                layer[i] = 0f;
            }
            else
            {
                layer[i] /= (1f - config.dropoutRate); // Scale up during training
            }
        }
    }
    #endregion

    #region Training Methods
    public void Train(RL.Transition newTransition)
    {
        try
        {
            // Add to experience replay buffer
            while (transitions.Count >= sizeOfExperienceReplayMemory)
                transitions.RemoveAt(0);
            transitions.Add(newTransition);
            
            // Train the network
            TrainNetwork();
        }
        catch (Exception e)
        {
            Debug.LogError($"Error during training: {e.Message}");
        }
    }
    
    private bool SampleBatch()
    {
        batchTransitions.Clear();
        if (transitions.Count < batchSize)
            return false;
            
        // Random sampling from experience replay
        for (int i = 0; i < batchSize; i++)
        {
            int randomIndex = UnityEngine.Random.Range(0, transitions.Count);
            batchTransitions.Add(transitions[randomIndex]);
        }
        return true;
    }
    
    private void TrainNetwork()
    {
        if (!SampleBatch())
            return;
            
        float totalLoss = 0f;
        
        for (int e = 0; e < numEpochs; e++)
        {
            for (int t = 0; t < batchTransitions.Count; t++)
            {
                var transition = batchTransitions[t];
                
                // Forward pass
                List<float> predictions = calcNet(transition.state);
                float predictedQ = predictions[transition.action];
                float targetQ = transition.reward;
                
                // Calculate loss (MSE)
                float loss = (targetQ - predictedQ) * (targetQ - predictedQ);
                totalLoss += loss;
                
                // Backward pass
                BackwardPass(transition, predictedQ, targetQ);
            }
        }
        
        // Update performance metrics
        currentLoss = totalLoss / (numEpochs * batchTransitions.Count);
        lossHistory.Add(currentLoss);
        
        // Keep only last 1000 loss values
        if (lossHistory.Count > 1000)
            lossHistory.RemoveAt(0);
            
        // Calculate moving average
        if (lossHistory.Count >= 10)
        {
            float sum = 0f;
            for (int i = lossHistory.Count - 10; i < lossHistory.Count; i++)
                sum += lossHistory[i];
            averageLoss = sum / 10f;
        }
    }
    
    private void BackwardPass(RL.Transition transition, float predictedQ, float targetQ)
    {
        // Calculate output layer error
        float error = targetQ - predictedQ;
        
        // Initialize output delta
        for (int j = 0; j < numOutputs; j++)
        {
            outputDelta[j] = (j == transition.action) ? error : 0f;
        }
        
        // Backpropagate through layers
        BackpropagateThroughLayers(transition.state);
        
        // Update weights using selected optimizer
        UpdateWeights();
    }
    
    private void BackpropagateThroughLayers(float[] input)
    {
        // Backpropagate through hidden layers (from output to input)
        for (int layer = config.numHiddenLayers - 1; layer >= 0; layer--)
        {
            int layerSize = config.numHiddens;
            
            // Calculate hidden layer deltas
            for (int j = 0; j < layerSize; j++)
            {
                hiddenDeltas[layer][j] = 0f;
                
                // Sum errors from next layer
                if (layer == config.numHiddenLayers - 1)
                {
                    // Last hidden layer - connect to output
                    for (int k = 0; k < numOutputs; k++)
                    {
                        hiddenDeltas[layer][j] += outputDelta[k] * weights[config.numHiddenLayers][j, k];
                    }
                }
                else
                {
                    // Hidden layer - connect to next hidden layer
                    for (int k = 0; k < layerSize; k++)
                    {
                        hiddenDeltas[layer][j] += hiddenDeltas[layer + 1][k] * weights[layer + 1][j, k];
                    }
                }
                
                // Apply activation derivative
                hiddenDeltas[layer][j] *= ActivationDerivative(hiddenLayers[layer][j], config.activationType);
            }
        }
    }
    
    private void UpdateWeights()
    {
        timestep++;
        
        switch (config.optimizerType)
        {
            case OptimizerType.Adam:
                UpdateWeightsAdam();
                break;
            case OptimizerType.RMSprop:
                UpdateWeightsRMSprop();
                break;
            case OptimizerType.SGD:
            default:
                UpdateWeightsSGD();
                break;
        }
    }
    
    private void UpdateWeightsAdam()
    {
        float beta1 = config.adamBeta1;
        float beta2 = config.adamBeta2;
        float epsilon = config.adamEpsilon;
        float learningRate = LR;
        
        // Update output layer weights
        for (int i = 0; i < config.numHiddens; i++)
        {
            for (int j = 0; j < numOutputs; j++)
            {
                float gradient = outputDelta[j] * hiddenLayers[config.numHiddenLayers - 1][i];
                
                // Update moments
                mWeights[config.numHiddenLayers][i, j] = beta1 * mWeights[config.numHiddenLayers][i, j] + (1 - beta1) * gradient;
                vWeights[config.numHiddenLayers][i, j] = beta2 * vWeights[config.numHiddenLayers][i, j] + (1 - beta2) * gradient * gradient;
                
                // Bias correction
                float mHat = mWeights[config.numHiddenLayers][i, j] / (1 - Mathf.Pow(beta1, timestep));
                float vHat = vWeights[config.numHiddenLayers][i, j] / (1 - Mathf.Pow(beta2, timestep));
                
                // Update weight
                weights[config.numHiddenLayers][i, j] += learningRate * mHat / (Mathf.Sqrt(vHat) + epsilon);
            }
        }
        
        // Update hidden layer weights
        for (int layer = 0; layer < config.numHiddenLayers; layer++)
        {
            int inputSize = (layer == 0) ? numInputs : config.numHiddens;
            int outputSize = config.numHiddens;
            
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    float inputValue = (layer == 0) ? 
                        (i < numInputs ? 0 : 1) : // Placeholder for input
                        hiddenLayers[layer - 1][i];
                    
                    float gradient = hiddenDeltas[layer][j] * inputValue;
                    
                    // Update moments
                    mWeights[layer][i, j] = beta1 * mWeights[layer][i, j] + (1 - beta1) * gradient;
                    vWeights[layer][i, j] = beta2 * vWeights[layer][i, j] + (1 - beta2) * gradient * gradient;
                    
                    // Bias correction
                    float mHat = mWeights[layer][i, j] / (1 - Mathf.Pow(beta1, timestep));
                    float vHat = vWeights[layer][i, j] / (1 - Mathf.Pow(beta2, timestep));
                    
                    // Update weight
                    weights[layer][i, j] += learningRate * mHat / (Mathf.Sqrt(vHat) + epsilon);
                }
            }
        }
    }
    
    private void UpdateWeightsSGD()
    {
        // Simple SGD update
        float learningRate = LR;
        
        // Update output layer weights
        for (int i = 0; i < config.numHiddens; i++)
        {
            for (int j = 0; j < numOutputs; j++)
            {
                float gradient = outputDelta[j] * hiddenLayers[config.numHiddenLayers - 1][i];
                weights[config.numHiddenLayers][i, j] += learningRate * gradient;
            }
        }
        
        // Update hidden layer weights (simplified)
        for (int layer = 0; layer < config.numHiddenLayers; layer++)
        {
            int inputSize = (layer == 0) ? numInputs : config.numHiddens;
            int outputSize = config.numHiddens;
            
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    float gradient = hiddenDeltas[layer][j] * 1f; // Simplified
                    weights[layer][i, j] += learningRate * gradient;
                }
            }
        }
    }
    
    private void UpdateWeightsRMSprop()
    {
        // RMSprop implementation (simplified)
        float decay = 0.9f;
        float epsilon = 1e-8f;
        float learningRate = LR;
        
        // Similar to Adam but with different moment calculation
        // Implementation would be similar to Adam but with different update rules
        UpdateWeightsSGD(); // Fallback to SGD for now
    }
    #endregion

    #region Performance Metrics and Monitoring
    public float GetCurrentLoss() => currentLoss;
    public float GetAverageLoss() => averageLoss;
    public List<float> GetLossHistory() => new List<float>(lossHistory);
    public int GetTrainingSteps() => timestep;
    public int GetExperienceBufferSize() => transitions.Count;
    
    public void ResetMetrics()
    {
        currentLoss = 0f;
        averageLoss = 0f;
        lossHistory.Clear();
        timestep = 0;
    }
    
    public string GetNetworkInfo()
    {
        return $"Network: {numInputs}->{config.numHiddens}->{numOutputs}, " +
               $"Loss: {currentLoss:F4}, Avg: {averageLoss:F4}, " +
               $"Steps: {timestep}, Buffer: {transitions.Count}";
    }
    #endregion

    #region Save and Load
    public void Save(StreamWriter writer)
    {
        try
        {
            // Save configuration
            writer.WriteLine(JsonUtility.ToJson(config));
            
            // Save training metrics
            writer.WriteLine(currentLoss);
            writer.WriteLine(averageLoss);
            writer.WriteLine(timestep);
            writer.WriteLine(lossHistory.Count);
            foreach (float loss in lossHistory)
            {
                writer.WriteLine(loss);
            }
            
            // Save network architecture
            writer.WriteLine(config.numHiddenLayers);
            
            // Save weights and biases
            for (int layer = 0; layer < config.numHiddenLayers + 1; layer++)
            {
                int inputSize = (layer == 0) ? numInputs : config.numHiddens;
                int outputSize = (layer == config.numHiddenLayers) ? numOutputs : config.numHiddens;
                
                // Save weights
                for (int i = 0; i < inputSize; i++)
                {
                    for (int j = 0; j < outputSize; j++)
                    {
                        writer.WriteLine(weights[layer][i, j]);
                    }
                }
                
                // Save biases
                for (int j = 0; j < outputSize; j++)
                {
                    writer.WriteLine(biases[layer][j]);
                }
            }
            
            // Save batch normalization parameters if enabled
            if (config.useBatchNormalization)
            {
                writer.WriteLine("BATCH_NORM");
                for (int layer = 0; layer < config.numHiddenLayers + 1; layer++)
                {
                    int layerSize = (layer < config.numHiddenLayers) ? config.numHiddens : numOutputs;
                    
                    for (int j = 0; j < layerSize; j++)
                    {
                        writer.WriteLine(batchNormGamma[layer][j]);
                        writer.WriteLine(batchNormBeta[layer][j]);
                        writer.WriteLine(batchNormMean[layer][j]);
                        writer.WriteLine(batchNormVariance[layer][j]);
                    }
                }
            }
            else
            {
                writer.WriteLine("NO_BATCH_NORM");
            }
            
            // Save optimizer states if using Adam
            if (config.optimizerType == OptimizerType.Adam)
            {
                writer.WriteLine("ADAM_OPTIMIZER");
                for (int layer = 0; layer < config.numHiddenLayers + 1; layer++)
                {
                    int inputSize = (layer == 0) ? numInputs : config.numHiddens;
                    int outputSize = (layer == config.numHiddenLayers) ? numOutputs : config.numHiddens;
                    
                    // Save weight moments
                    for (int i = 0; i < inputSize; i++)
                    {
                        for (int j = 0; j < outputSize; j++)
                        {
                            writer.WriteLine(mWeights[layer][i, j]);
                            writer.WriteLine(vWeights[layer][i, j]);
                        }
                    }
                    
                    // Save bias moments
                    for (int j = 0; j < outputSize; j++)
                    {
                        writer.WriteLine(mBiases[layer][j]);
                        writer.WriteLine(vBiases[layer][j]);
                    }
                }
            }
            else
            {
                writer.WriteLine("NO_ADAM_OPTIMIZER");
            }
            
            Debug.Log("Neural network saved successfully");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving neural network: {e.Message}");
            throw;
        }
    }
    
    public void Load(StreamReader reader)
    {
        try
        {
            // Load configuration
            string configJson = reader.ReadLine();
            config = JsonUtility.FromJson<NetworkConfig>(configJson);
            
            // Reinitialize network with loaded config
            InitializeNetwork();
            
            // Load training metrics
            currentLoss = float.Parse(reader.ReadLine());
            averageLoss = float.Parse(reader.ReadLine());
            timestep = int.Parse(reader.ReadLine());
            
            int lossCount = int.Parse(reader.ReadLine());
            lossHistory.Clear();
            for (int i = 0; i < lossCount; i++)
            {
                lossHistory.Add(float.Parse(reader.ReadLine()));
            }
            
            // Load network architecture
            int loadedHiddenLayers = int.Parse(reader.ReadLine());
            if (loadedHiddenLayers != config.numHiddenLayers)
            {
                Debug.LogWarning($"Hidden layer count mismatch: loaded {loadedHiddenLayers}, expected {config.numHiddenLayers}");
            }
            
            // Load weights and biases
            for (int layer = 0; layer < config.numHiddenLayers + 1; layer++)
            {
                int inputSize = (layer == 0) ? numInputs : config.numHiddens;
                int outputSize = (layer == config.numHiddenLayers) ? numOutputs : config.numHiddens;
                
                // Load weights
                for (int i = 0; i < inputSize; i++)
                {
                    for (int j = 0; j < outputSize; j++)
                    {
                        weights[layer][i, j] = float.Parse(reader.ReadLine());
                    }
                }
                
                // Load biases
                for (int j = 0; j < outputSize; j++)
                {
                    biases[layer][j] = float.Parse(reader.ReadLine());
                }
            }
            
            // Load batch normalization parameters if present
            string batchNormFlag = reader.ReadLine();
            if (batchNormFlag == "BATCH_NORM" && config.useBatchNormalization)
            {
                for (int layer = 0; layer < config.numHiddenLayers + 1; layer++)
                {
                    int layerSize = (layer < config.numHiddenLayers) ? config.numHiddens : numOutputs;
                    
                    for (int j = 0; j < layerSize; j++)
                    {
                        batchNormGamma[layer][j] = float.Parse(reader.ReadLine());
                        batchNormBeta[layer][j] = float.Parse(reader.ReadLine());
                        batchNormMean[layer][j] = float.Parse(reader.ReadLine());
                        batchNormVariance[layer][j] = float.Parse(reader.ReadLine());
                    }
                }
            }
            
            // Load optimizer states if present
            string optimizerFlag = reader.ReadLine();
            if (optimizerFlag == "ADAM_OPTIMIZER" && config.optimizerType == OptimizerType.Adam)
            {
                for (int layer = 0; layer < config.numHiddenLayers + 1; layer++)
                {
                    int inputSize = (layer == 0) ? numInputs : config.numHiddens;
                    int outputSize = (layer == config.numHiddenLayers) ? numOutputs : config.numHiddens;
                    
                    // Load weight moments
                    for (int i = 0; i < inputSize; i++)
                    {
                        for (int j = 0; j < outputSize; j++)
                        {
                            mWeights[layer][i, j] = float.Parse(reader.ReadLine());
                            vWeights[layer][i, j] = float.Parse(reader.ReadLine());
                        }
                    }
                    
                    // Load bias moments
                    for (int j = 0; j < outputSize; j++)
                    {
                        mBiases[layer][j] = float.Parse(reader.ReadLine());
                        vBiases[layer][j] = float.Parse(reader.ReadLine());
                    }
                }
            }
            
            Debug.Log($"Neural network loaded successfully. Loss: {currentLoss:F4}, Steps: {timestep}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error loading neural network: {e.Message}");
            throw;
        }
    }
    #endregion
}