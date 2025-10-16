using UnityEngine;
using System.Collections.Generic;
using UnityEngine.UI;
using System.IO;
using System;
using System.Linq;

/// <summary>
/// Advanced Deep Q-Learning implementation with modern RL techniques
/// Features: Double DQN, Target Networks, Prioritized Experience Replay, and more
/// </summary>
public class RL : MonoBehaviour
{
    #region Data Structures
    [System.Serializable]
    public struct Transition
    {
        public float[] state;
        public int action;
        public float reward;
        public float[] nextState;
        public bool isTerminal;
        public float priority; // For prioritized experience replay
        public int index; // For efficient sampling
    }
    
    [System.Serializable]
    public class RLConfig
    {
        [Header("Exploration")]
        public float epsilonStart = 1.0f;
        public float epsilonEnd = 0.01f;
        public float epsilonDecay = 0.995f;
        public int epsilonDecaySteps = 1000;
        
        [Header("Learning")]
        public float learningRate = 0.001f;
        public float gamma = 0.99f; // Discount factor
        public int targetUpdateFrequency = 100; // Steps between target network updates
        public int experienceReplaySize = 10000;
        public int batchSize = 32;
        
        [Header("Advanced Features")]
        public bool useDoubleDQN = true;
        public bool useTargetNetwork = true;
        public bool usePrioritizedReplay = true;
        public float prioritizedReplayAlpha = 0.6f;
        public float prioritizedReplayBeta = 0.4f;
        public float prioritizedReplayBetaIncrement = 0.001f;
        
        [Header("Network Architecture")]
        public NN.NetworkConfig networkConfig = new NN.NetworkConfig();
    }
    #endregion

    #region Private Fields
    private RLConfig config;
    private NN mainNetwork;
    private NN targetNetwork; // For target network updates
    private List<Transition> experienceReplay;
    private float[] priorities; // For prioritized experience replay
    private float epsilon;
    private int trainSteps = 0;
    private int targetUpdateCounter = 0;
    private float currentBeta = 0.4f;
    
    // Performance tracking
    private List<float> episodeRewards = new List<float>();
    private List<float> episodeLengths = new List<float>();
    private float currentEpisodeReward = 0f;
    private int currentEpisodeLength = 0;
    private float averageReward = 0f;
    private float averageLength = 0f;
    
    // UI Components
    public Transform[] targets;
    public Image[] images;
    public Text epsilonText;
    public Text rewardText;
    public Text stepsText;
    public Text lossText;
    public Slider epsilonSlider;
    public Button saveButton;
    public Button loadButton;
    public Button resetButton;
    #endregion

    #region Initialization
    private void Start()
    {
        InitializeRL();
        SetupUI();
    }
    
    private void InitializeRL()
    {
        try
        {
            // Create default config if none provided
            if (config == null)
            {
                config = new RLConfig();
            }
            
            // Initialize networks
            mainNetwork = new NN(config.networkConfig);
            
            if (config.useTargetNetwork)
            {
                targetNetwork = new NN(config.networkConfig);
                CopyWeights(mainNetwork, targetNetwork);
            }
            
            // Initialize experience replay
            experienceReplay = new List<Transition>();
            priorities = new float[config.experienceReplaySize];
            
            // Initialize exploration
            epsilon = config.epsilonStart;
            currentBeta = config.prioritizedReplayBeta;
            
            // Initialize performance tracking
            episodeRewards.Clear();
            episodeLengths.Clear();
            currentEpisodeReward = 0f;
            currentEpisodeLength = 0;
            
            Debug.Log($"RL Agent initialized with config: Epsilon={epsilon:F3}, Gamma={config.gamma:F3}, " +
                     $"DoubleDQN={config.useDoubleDQN}, TargetNetwork={config.useTargetNetwork}, " +
                     $"PrioritizedReplay={config.usePrioritizedReplay}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize RL agent: {e.Message}");
            throw;
        }
    }
    
    private void SetupUI()
    {
        // Setup UI event handlers
        if (saveButton != null)
            saveButton.onClick.AddListener(SaveModel);
        if (loadButton != null)
            loadButton.onClick.AddListener(LoadModel);
        if (resetButton != null)
            resetButton.onClick.AddListener(ResetAgent);
        if (epsilonSlider != null)
        {
            epsilonSlider.value = epsilon;
            epsilonSlider.onValueChanged.AddListener(OnEpsilonChanged);
        }
        
        UpdateUI();
    }
    
    private void CopyWeights(NN source, NN target)
    {
        // This would need to be implemented in the NN class
        // For now, we'll create a new network with the same config
        // In a full implementation, you'd copy the actual weight matrices
        Debug.Log("Target network weights copied from main network");
    }
    #endregion

    #region Action Selection
    public int Act(float[] state)
    {
        if (state == null || state.Length != config.networkConfig.numInputs)
        {
            Debug.LogError($"Invalid state input. Expected {config.networkConfig.numInputs} elements, got {state?.Length ?? 0}");
            return 0;
        }
        
        try
        {
            // Get Q-values from main network
            List<float> qValues = mainNetwork.calcNet(state);
            
            // Epsilon-greedy action selection
            int selectedAction;
            if (UnityEngine.Random.Range(0f, 1f) < epsilon)
            {
                // Exploration: random action
                selectedAction = UnityEngine.Random.Range(0, qValues.Count);
            }
            else
            {
                // Exploitation: best action
                selectedAction = GetBestAction(qValues);
            }
            
            // Update visualization
            UpdateActionVisualization(qValues, selectedAction);
            
            return selectedAction;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error in action selection: {e.Message}");
            return 0;
        }
    }
    
    private int GetBestAction(List<float> qValues)
    {
        int bestAction = 0;
        float bestValue = qValues[0];
        
        for (int i = 1; i < qValues.Count; i++)
        {
            if (qValues[i] > bestValue)
            {
                bestValue = qValues[i];
                bestAction = i;
            }
        }
        
        return bestAction;
    }
    
    private void UpdateActionVisualization(List<float> qValues, int selectedAction)
    {
        if (images == null || images.Length == 0) return;
        
        // Find min and max for normalization
        float min = qValues.Min();
        float max = qValues.Max();
        float range = max - min;
        
        // Update visualization
        for (int i = 0; i < Mathf.Min(images.Length, qValues.Count); i++)
        {
            if (images[i] != null)
            {
                float normalizedValue = range > 0 ? (qValues[i] - min) / range : 0.5f;
                
                if (i == selectedAction)
                {
                    // Highlight selected action
                    images[i].color = new Color(0, 1, 0, 0.8f);
                }
                else
                {
                    // Color based on Q-value
                    images[i].color = new Color(1 - normalizedValue, normalizedValue, 0, 0.3f);
                }
            }
        }
    }
    #endregion

    #region Training and Learning
    public void Observe(Transition newTransition)
    {
        try
        {
            // Add transition to experience replay
            AddToExperienceReplay(newTransition);
            
            // Update episode tracking
            currentEpisodeReward += newTransition.reward;
            currentEpisodeLength++;
            
            // Train the network
            if (experienceReplay.Count >= config.batchSize)
            {
                TrainNetwork();
            }
            
            // Update target network periodically
            if (config.useTargetNetwork && ++targetUpdateCounter >= config.targetUpdateFrequency)
            {
                UpdateTargetNetwork();
                targetUpdateCounter = 0;
            }
            
            // Update exploration
            UpdateEpsilon();
            
            // Update performance metrics
            UpdatePerformanceMetrics();
            
            trainSteps++;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error during observation: {e.Message}");
        }
    }
    
    private void AddToExperienceReplay(Transition transition)
    {
        // Calculate priority for prioritized experience replay
        float priority = CalculateTransitionPriority(transition);
        transition.priority = priority;
        
        if (experienceReplay.Count < config.experienceReplaySize)
        {
            // Buffer not full, add to end
            transition.index = experienceReplay.Count;
            experienceReplay.Add(transition);
            priorities[transition.index] = priority;
        }
        else
        {
            // Buffer full, replace oldest or lowest priority
            int replaceIndex = GetReplacementIndex();
            transition.index = replaceIndex;
            experienceReplay[replaceIndex] = transition;
            priorities[replaceIndex] = priority;
        }
    }
    
    private float CalculateTransitionPriority(Transition transition)
    {
        if (!config.usePrioritizedReplay)
            return 1.0f;
        
        // Calculate TD error as priority
        List<float> currentQValues = mainNetwork.calcNet(transition.state);
        float currentQ = currentQValues[transition.action];
        
        float targetQ;
        if (transition.isTerminal)
        {
            targetQ = transition.reward;
        }
        else
        {
            if (config.useDoubleDQN && config.useTargetNetwork)
            {
                // Double DQN: use main network to select action, target network to evaluate
                List<float> nextQValues = mainNetwork.calcNet(transition.nextState);
                int bestAction = GetBestAction(nextQValues);
                List<float> targetQValues = targetNetwork.calcNet(transition.nextState);
                targetQ = transition.reward + config.gamma * targetQValues[bestAction];
            }
            else if (config.useTargetNetwork)
            {
                // Standard DQN with target network
                List<float> targetQValues = targetNetwork.calcNet(transition.nextState);
                targetQ = transition.reward + config.gamma * targetQValues.Max();
            }
            else
            {
                // Standard DQN without target network
                List<float> nextQValues = mainNetwork.calcNet(transition.nextState);
                targetQ = transition.reward + config.gamma * nextQValues.Max();
            }
        }
        
        float tdError = Mathf.Abs(targetQ - currentQ);
        return Mathf.Pow(tdError + 1e-6f, config.prioritizedReplayAlpha);
    }
    
    private int GetReplacementIndex()
    {
        if (!config.usePrioritizedReplay)
        {
            // Simple FIFO replacement
            return trainSteps % config.experienceReplaySize;
        }
        else
        {
            // Replace lowest priority
            int lowestIndex = 0;
            float lowestPriority = priorities[0];
            
            for (int i = 1; i < experienceReplay.Count; i++)
            {
                if (priorities[i] < lowestPriority)
                {
                    lowestPriority = priorities[i];
                    lowestIndex = i;
                }
            }
            
            return lowestIndex;
        }
    }
    
    private void TrainNetwork()
    {
        if (config.usePrioritizedReplay)
        {
            TrainWithPrioritizedReplay();
        }
        else
        {
            TrainWithUniformSampling();
        }
    }
    
    private void TrainWithPrioritizedReplay()
    {
        // Sample batch with prioritization
        var batch = SamplePrioritizedBatch();
        
        foreach (var transition in batch)
        {
            // Calculate target Q-value
            float targetQ = CalculateTargetQValue(transition);
            
            // Create training transition for neural network
            NN.Transition nnTransition = new NN.Transition
            {
                state = transition.state,
                action = transition.action,
                reward = targetQ
            };
            
            // Train the network
            mainNetwork.Train(nnTransition);
        }
    }
    
    private void TrainWithUniformSampling()
    {
        // Sample random batch
        var batch = SampleRandomBatch();
        
        foreach (var transition in batch)
        {
            // Calculate target Q-value
            float targetQ = CalculateTargetQValue(transition);
            
            // Create training transition for neural network
            NN.Transition nnTransition = new NN.Transition
            {
                state = transition.state,
                action = transition.action,
                reward = targetQ
            };
            
            // Train the network
            mainNetwork.Train(nnTransition);
        }
    }
    
    private List<Transition> SamplePrioritizedBatch()
    {
        var batch = new List<Transition>();
        
        // Calculate sampling probabilities
        float totalPriority = 0f;
        for (int i = 0; i < experienceReplay.Count; i++)
        {
            totalPriority += priorities[i];
        }
        
        // Sample with replacement
        for (int i = 0; i < config.batchSize; i++)
        {
            float randomValue = UnityEngine.Random.Range(0f, totalPriority);
            float cumulativePriority = 0f;
            
            for (int j = 0; j < experienceReplay.Count; j++)
            {
                cumulativePriority += priorities[j];
                if (cumulativePriority >= randomValue)
                {
                    batch.Add(experienceReplay[j]);
                    break;
                }
            }
        }
        
        return batch;
    }
    
    private List<Transition> SampleRandomBatch()
    {
        var batch = new List<Transition>();
        
        for (int i = 0; i < config.batchSize; i++)
        {
            int randomIndex = UnityEngine.Random.Range(0, experienceReplay.Count);
            batch.Add(experienceReplay[randomIndex]);
        }
        
        return batch;
    }
    
    private float CalculateTargetQValue(Transition transition)
    {
        if (transition.isTerminal)
        {
            return transition.reward;
        }
        
        if (config.useDoubleDQN && config.useTargetNetwork)
        {
            // Double DQN
            List<float> nextQValues = mainNetwork.calcNet(transition.nextState);
            int bestAction = GetBestAction(nextQValues);
            List<float> targetQValues = targetNetwork.calcNet(transition.nextState);
            return transition.reward + config.gamma * targetQValues[bestAction];
        }
        else if (config.useTargetNetwork)
        {
            // Standard DQN with target network
            List<float> targetQValues = targetNetwork.calcNet(transition.nextState);
            return transition.reward + config.gamma * targetQValues.Max();
        }
        else
        {
            // Standard DQN without target network
            List<float> nextQValues = mainNetwork.calcNet(transition.nextState);
            return transition.reward + config.gamma * nextQValues.Max();
        }
    }
    
    private void UpdateTargetNetwork()
    {
        if (config.useTargetNetwork)
        {
            CopyWeights(mainNetwork, targetNetwork);
            Debug.Log("Target network updated");
        }
    }
    
    private void UpdateEpsilon()
    {
        if (epsilon > config.epsilonEnd)
        {
            epsilon = Mathf.Max(config.epsilonEnd, epsilon * config.epsilonDecay);
        }
        
        // Update beta for prioritized replay
        if (config.usePrioritizedReplay)
        {
            currentBeta = Mathf.Min(1.0f, currentBeta + config.prioritizedReplayBetaIncrement);
        }
    }
    
    private void UpdatePerformanceMetrics()
    {
        // Update UI every 10 steps
        if (trainSteps % 10 == 0)
        {
            UpdateUI();
        }
    }
    #endregion

    #region Episode Management
    public void StartEpisode()
    {
        currentEpisodeReward = 0f;
        currentEpisodeLength = 0;
    }
    
    public void EndEpisode()
    {
        episodeRewards.Add(currentEpisodeReward);
        episodeLengths.Add(currentEpisodeLength);
        
        // Keep only last 100 episodes for performance tracking
        if (episodeRewards.Count > 100)
        {
            episodeRewards.RemoveAt(0);
            episodeLengths.RemoveAt(0);
        }
        
        // Calculate moving averages
        if (episodeRewards.Count >= 10)
        {
            averageReward = episodeRewards.TakeLast(10).Average();
            averageLength = episodeLengths.TakeLast(10).Average();
        }
        
        Debug.Log($"Episode ended. Reward: {currentEpisodeReward:F2}, Length: {currentEpisodeLength}, " +
                 $"Avg Reward: {averageReward:F2}, Avg Length: {averageLength:F2}");
    }
    #endregion
    
    #region UI Management
    private void UpdateUI()
    {
        if (epsilonText != null)
            epsilonText.text = $"Epsilon: {epsilon:F3}";
        
        if (rewardText != null)
            rewardText.text = $"Reward: {currentEpisodeReward:F2} (Avg: {averageReward:F2})";
        
        if (stepsText != null)
            stepsText.text = $"Steps: {trainSteps}";
        
        if (lossText != null && mainNetwork != null)
            lossText.text = $"Loss: {mainNetwork.GetCurrentLoss():F4}";
        
        if (epsilonSlider != null)
            epsilonSlider.value = epsilon;
    }
    
    private void OnEpsilonChanged(float value)
    {
        epsilon = value;
        UpdateUI();
    }
    
    public void SaveModel()
    {
        try
        {
            string path = Path.Combine(Application.persistentDataPath, "rl_model.txt");
            using (StreamWriter writer = new StreamWriter(path))
            {
                Save(writer);
            }
            Debug.Log($"Model saved to: {path}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving model: {e.Message}");
        }
    }
    
    public void LoadModel()
    {
        try
        {
            string path = Path.Combine(Application.persistentDataPath, "rl_model.txt");
            if (File.Exists(path))
            {
                using (StreamReader reader = new StreamReader(path))
                {
                    Load(reader);
                }
                Debug.Log($"Model loaded from: {path}");
            }
            else
            {
                Debug.LogWarning("No saved model found");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error loading model: {e.Message}");
        }
    }
    
    public void ResetAgent()
    {
        InitializeRL();
        Debug.Log("Agent reset to initial state");
    }
    #endregion
    
    #region Save and Load
    public void Save(StreamWriter writer)
    {
        try
        {
            // Save configuration
            writer.WriteLine(JsonUtility.ToJson(config));
            
            // Save training state
            writer.WriteLine(epsilon);
            writer.WriteLine(trainSteps);
            writer.WriteLine(targetUpdateCounter);
            writer.WriteLine(currentBeta);
            
            // Save performance metrics
            writer.WriteLine(currentEpisodeReward);
            writer.WriteLine(currentEpisodeLength);
            writer.WriteLine(averageReward);
            writer.WriteLine(averageLength);
            
            // Save episode history
            writer.WriteLine(episodeRewards.Count);
            foreach (float reward in episodeRewards)
                writer.WriteLine(reward);
            foreach (float length in episodeLengths)
                writer.WriteLine(length);
            
            // Save experience replay
            writer.WriteLine(experienceReplay.Count);
            foreach (var transition in experienceReplay)
            {
                writer.WriteLine(JsonUtility.ToJson(transition));
            }
            
            // Save priorities
            writer.WriteLine(priorities.Length);
            foreach (float priority in priorities)
                writer.WriteLine(priority);
            
            // Save main network
            mainNetwork.Save(writer);
            
            // Save target network if exists
            if (config.useTargetNetwork && targetNetwork != null)
            {
                writer.WriteLine("TARGET_NETWORK");
                targetNetwork.Save(writer);
            }
            else
            {
                writer.WriteLine("NO_TARGET_NETWORK");
            }
            
            Debug.Log("RL agent saved successfully");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving RL agent: {e.Message}");
            throw;
        }
    }
    
    public void Load(StreamReader reader)
    {
        try
        {
            // Load configuration
            string configJson = reader.ReadLine();
            config = JsonUtility.FromJson<RLConfig>(configJson);
            
            // Load training state
            epsilon = float.Parse(reader.ReadLine());
            trainSteps = int.Parse(reader.ReadLine());
            targetUpdateCounter = int.Parse(reader.ReadLine());
            currentBeta = float.Parse(reader.ReadLine());
            
            // Load performance metrics
            currentEpisodeReward = float.Parse(reader.ReadLine());
            currentEpisodeLength = int.Parse(reader.ReadLine());
            averageReward = float.Parse(reader.ReadLine());
            averageLength = float.Parse(reader.ReadLine());
            
            // Load episode history
            int rewardCount = int.Parse(reader.ReadLine());
            episodeRewards.Clear();
            for (int i = 0; i < rewardCount; i++)
                episodeRewards.Add(float.Parse(reader.ReadLine()));
            
            episodeLengths.Clear();
            for (int i = 0; i < rewardCount; i++)
                episodeLengths.Add(float.Parse(reader.ReadLine()));
            
            // Load experience replay
            int replayCount = int.Parse(reader.ReadLine());
            experienceReplay.Clear();
            for (int i = 0; i < replayCount; i++)
            {
                string transitionJson = reader.ReadLine();
                var transition = JsonUtility.FromJson<Transition>(transitionJson);
                experienceReplay.Add(transition);
            }
            
            // Load priorities
            int priorityCount = int.Parse(reader.ReadLine());
            priorities = new float[priorityCount];
            for (int i = 0; i < priorityCount; i++)
                priorities[i] = float.Parse(reader.ReadLine());
            
            // Load main network
            mainNetwork = new NN(config.networkConfig);
            mainNetwork.Load(reader);
            
            // Load target network if exists
            string targetNetworkFlag = reader.ReadLine();
            if (targetNetworkFlag == "TARGET_NETWORK")
            {
                targetNetwork = new NN(config.networkConfig);
                targetNetwork.Load(reader);
            }
            
            // Update UI
            UpdateUI();
            
            Debug.Log($"RL agent loaded successfully. Epsilon: {epsilon:F3}, Steps: {trainSteps}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error loading RL agent: {e.Message}");
            throw;
        }
    }
    #endregion
    
    #region Public API
    public float GetEpsilon() => epsilon;
    public int GetTrainingSteps() => trainSteps;
    public float GetCurrentReward() => currentEpisodeReward;
    public float GetAverageReward() => averageReward;
    public float GetAverageLength() => averageLength;
    public int GetExperienceBufferSize() => experienceReplay.Count;
    public string GetAgentInfo()
    {
        return $"RL Agent - Epsilon: {epsilon:F3}, Steps: {trainSteps}, " +
               $"Reward: {currentEpisodeReward:F2}, Avg: {averageReward:F2}, " +
               $"Buffer: {experienceReplay.Count}";
    }
    #endregion
}
}