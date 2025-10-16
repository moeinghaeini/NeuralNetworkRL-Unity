# Usage Examples

## Basic Example: Simple Game Agent

### Setup
```csharp
using UnityEngine;

public class SimpleGameAgent : MonoBehaviour
{
    private RL agent;
    private float[] currentState;
    private int currentAction;
    
    void Start()
    {
        // Get or add RL component
        agent = GetComponent<RL>();
        if (agent == null)
        {
            agent = gameObject.AddComponent<RL>();
        }
        
        // Start first episode
        agent.StartEpisode();
    }
    
    void Update()
    {
        // Get current state (example: position, velocity, etc.)
        currentState = GetCurrentState();
        
        // Get action from agent
        currentAction = agent.Act(currentState);
        
        // Execute action
        ExecuteAction(currentAction);
        
        // Get reward and next state
        float reward = CalculateReward();
        float[] nextState = GetNextState();
        bool isTerminal = IsEpisodeOver();
        
        // Create transition
        RL.Transition transition = new RL.Transition
        {
            state = currentState,
            action = currentAction,
            reward = reward,
            nextState = nextState,
            isTerminal = isTerminal
        };
        
        // Learn from experience
        agent.Observe(transition);
        
        // End episode if terminal
        if (isTerminal)
        {
            agent.EndEpisode();
            agent.StartEpisode(); // Start new episode
        }
    }
    
    private float[] GetCurrentState()
    {
        // Example: return position, velocity, and other relevant features
        return new float[] {
            transform.position.x,
            transform.position.y,
            transform.position.z,
            GetComponent<Rigidbody>().velocity.x,
            GetComponent<Rigidbody>().velocity.y,
            GetComponent<Rigidbody>().velocity.z,
            // ... more state features
        };
    }
    
    private void ExecuteAction(int action)
    {
        // Example: move in different directions based on action
        Vector3 direction = Vector3.zero;
        switch (action)
        {
            case 0: direction = Vector3.forward; break;
            case 1: direction = Vector3.back; break;
            case 2: direction = Vector3.left; break;
            case 3: direction = Vector3.right; break;
            // ... more actions
        }
        
        transform.Translate(direction * Time.deltaTime * 5f);
    }
    
    private float CalculateReward()
    {
        // Example: reward based on distance to target
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        return -distanceToTarget; // Negative reward for distance
    }
    
    private float[] GetNextState()
    {
        return GetCurrentState(); // Same as current state for next step
    }
    
    private bool IsEpisodeOver()
    {
        // Example: episode ends when reaching target or timeout
        return Vector3.Distance(transform.position, target.position) < 1f ||
               Time.time - episodeStartTime > 30f;
    }
}
```

## Advanced Example: Custom Configuration

### Custom Network Architecture
```csharp
public class AdvancedRLAgent : MonoBehaviour
{
    private RL agent;
    
    void Start()
    {
        // Create custom network configuration
        NN.NetworkConfig networkConfig = new NN.NetworkConfig
        {
            numInputs = 20,           // More complex state
            numHiddens = 128,         // Larger hidden layer
            numOutputs = 8,           // More actions
            numHiddenLayers = 4,      // Deeper network
            learningRate = 0.0005f,   // Lower learning rate
            batchSize = 64,           // Larger batches
            experienceReplaySize = 50000, // Larger buffer
            useBatchNormalization = true,
            useDropout = true,
            dropoutRate = 0.3f,
            activationType = NN.ActivationType.LeakyReLU,
            weightInitType = NN.WeightInitType.He,
            optimizerType = NN.OptimizerType.Adam
        };
        
        // Create custom RL configuration
        RL.RLConfig rlConfig = new RL.RLConfig
        {
            epsilonStart = 0.9f,
            epsilonEnd = 0.05f,
            epsilonDecay = 0.999f,
            gamma = 0.95f,
            targetUpdateFrequency = 200,
            experienceReplaySize = 50000,
            batchSize = 64,
            useDoubleDQN = true,
            useTargetNetwork = true,
            usePrioritizedReplay = true,
            prioritizedReplayAlpha = 0.7f,
            prioritizedReplayBeta = 0.5f,
            networkConfig = networkConfig
        };
        
        // Initialize agent with custom config
        agent = gameObject.AddComponent<RL>();
        // Note: You would need to modify RL class to accept config in constructor
    }
}
```

## Example: Training with Curriculum Learning

### Progressive Difficulty
```csharp
public class CurriculumLearningAgent : MonoBehaviour
{
    private RL agent;
    private int currentDifficulty = 0;
    private float[] difficultyThresholds = { 50f, 100f, 200f, 500f };
    
    void Start()
    {
        agent = GetComponent<RL>();
        agent.StartEpisode();
    }
    
    void Update()
    {
        // Standard RL loop
        float[] state = GetCurrentState();
        int action = agent.Act(state);
        ExecuteAction(action);
        
        float reward = CalculateReward();
        float[] nextState = GetNextState();
        bool isTerminal = IsEpisodeOver();
        
        RL.Transition transition = new RL.Transition
        {
            state = state,
            action = action,
            reward = reward,
            nextState = nextState,
            isTerminal = isTerminal
        };
        
        agent.Observe(transition);
        
        if (isTerminal)
        {
            agent.EndEpisode();
            
            // Check if ready for next difficulty
            if (agent.GetAverageReward() > difficultyThresholds[currentDifficulty])
            {
                currentDifficulty++;
                IncreaseDifficulty();
                Debug.Log($"Increased difficulty to level {currentDifficulty}");
            }
            
            agent.StartEpisode();
        }
    }
    
    private void IncreaseDifficulty()
    {
        // Example: increase obstacle density, speed, etc.
        switch (currentDifficulty)
        {
            case 1:
                // Add more obstacles
                break;
            case 2:
                // Increase movement speed
                break;
            case 3:
                // Add dynamic obstacles
                break;
        }
    }
}
```

## Example: Multi-Agent Training

### Multiple Agents Learning Together
```csharp
public class MultiAgentManager : MonoBehaviour
{
    public GameObject[] agentPrefabs;
    private RL[] agents;
    private int currentAgentIndex = 0;
    
    void Start()
    {
        agents = new RL[agentPrefabs.Length];
        
        for (int i = 0; i < agentPrefabs.Length; i++)
        {
            GameObject agentObj = Instantiate(agentPrefabs[i]);
            agents[i] = agentObj.GetComponent<RL>();
            agents[i].StartEpisode();
        }
    }
    
    void Update()
    {
        // Train one agent per frame to distribute computation
        RL currentAgent = agents[currentAgentIndex];
        
        float[] state = GetCurrentState(currentAgentIndex);
        int action = currentAgent.Act(state);
        ExecuteAction(currentAgentIndex, action);
        
        float reward = CalculateReward(currentAgentIndex);
        float[] nextState = GetNextState(currentAgentIndex);
        bool isTerminal = IsEpisodeOver(currentAgentIndex);
        
        RL.Transition transition = new RL.Transition
        {
            state = state,
            action = action,
            reward = reward,
            nextState = nextState,
            isTerminal = isTerminal
        };
        
        currentAgent.Observe(transition);
        
        if (isTerminal)
        {
            currentAgent.EndEpisode();
            currentAgent.StartEpisode();
        }
        
        // Move to next agent
        currentAgentIndex = (currentAgentIndex + 1) % agents.Length;
    }
    
    public void SaveAllAgents()
    {
        for (int i = 0; i < agents.Length; i++)
        {
            // Save each agent with unique filename
            string path = Path.Combine(Application.persistentDataPath, $"agent_{i}.txt");
            using (StreamWriter writer = new StreamWriter(path))
            {
                agents[i].Save(writer);
            }
        }
    }
}
```

## Example: Hyperparameter Tuning

### Automated Parameter Search
```csharp
public class HyperparameterTuner : MonoBehaviour
{
    [System.Serializable]
    public class ParameterRange
    {
        public float learningRateMin = 0.0001f;
        public float learningRateMax = 0.01f;
        public int batchSizeMin = 16;
        public int batchSizeMax = 128;
        public float gammaMin = 0.9f;
        public float gammaMax = 0.99f;
    }
    
    public ParameterRange parameterRange;
    public int numTrials = 10;
    public int episodesPerTrial = 100;
    
    private List<float> bestRewards = new List<float>();
    private RL.RLConfig bestConfig;
    
    void Start()
    {
        StartCoroutine(RunHyperparameterSearch());
    }
    
    IEnumerator RunHyperparameterSearch()
    {
        for (int trial = 0; trial < numTrials; trial++)
        {
            // Generate random parameters
            RL.RLConfig config = GenerateRandomConfig();
            
            // Create and train agent
            RL agent = CreateAgent(config);
            float averageReward = TrainAgent(agent, episodesPerTrial);
            
            // Record results
            bestRewards.Add(averageReward);
            if (averageReward > bestRewards.Max())
            {
                bestConfig = config;
            }
            
            Debug.Log($"Trial {trial}: Reward = {averageReward:F2}");
            
            yield return new WaitForSeconds(0.1f); // Prevent freezing
        }
        
        Debug.Log($"Best configuration found with reward: {bestRewards.Max():F2}");
        LogBestConfig();
    }
    
    private RL.RLConfig GenerateRandomConfig()
    {
        return new RL.RLConfig
        {
            learningRate = Random.Range(parameterRange.learningRateMin, parameterRange.learningRateMax),
            batchSize = Random.Range(parameterRange.batchSizeMin, parameterRange.batchSizeMax),
            gamma = Random.Range(parameterRange.gammaMin, parameterRange.gammaMax),
            // ... other parameters
        };
    }
    
    private float TrainAgent(RL agent, int episodes)
    {
        List<float> episodeRewards = new List<float>();
        
        for (int episode = 0; episode < episodes; episode++)
        {
            agent.StartEpisode();
            
            // Run episode
            while (!IsEpisodeOver())
            {
                float[] state = GetCurrentState();
                int action = agent.Act(state);
                ExecuteAction(action);
                
                float reward = CalculateReward();
                float[] nextState = GetNextState();
                bool isTerminal = IsEpisodeOver();
                
                RL.Transition transition = new RL.Transition
                {
                    state = state,
                    action = action,
                    reward = reward,
                    nextState = nextState,
                    isTerminal = isTerminal
                };
                
                agent.Observe(transition);
                
                if (isTerminal)
                {
                    agent.EndEpisode();
                    episodeRewards.Add(agent.GetCurrentReward());
                    break;
                }
            }
        }
        
        return episodeRewards.Average();
    }
}
```

## Example: Real-time Monitoring

### Performance Dashboard
```csharp
public class PerformanceMonitor : MonoBehaviour
{
    public Text epsilonText;
    public Text rewardText;
    public Text lossText;
    public Text stepsText;
    public Slider epsilonSlider;
    public Button saveButton;
    public Button loadButton;
    public Button resetButton;
    
    private RL agent;
    private LineRenderer lossGraph;
    private List<float> lossHistory = new List<float>();
    
    void Start()
    {
        agent = GetComponent<RL>();
        SetupUI();
        StartCoroutine(UpdateMetrics());
    }
    
    private void SetupUI()
    {
        // Setup button listeners
        saveButton.onClick.AddListener(() => agent.SaveModel());
        loadButton.onClick.AddListener(() => agent.LoadModel());
        resetButton.onClick.AddListener(() => agent.ResetAgent());
        
        // Setup epsilon slider
        epsilonSlider.onValueChanged.AddListener(OnEpsilonChanged);
        
        // Setup loss graph
        lossGraph = gameObject.AddComponent<LineRenderer>();
        lossGraph.material = new Material(Shader.Find("Sprites/Default"));
        lossGraph.color = Color.red;
        lossGraph.startWidth = 0.1f;
        lossGraph.endWidth = 0.1f;
    }
    
    private void OnEpsilonChanged(float value)
    {
        // Allow manual epsilon adjustment
        // Note: You would need to add a SetEpsilon method to RL class
    }
    
    IEnumerator UpdateMetrics()
    {
        while (true)
        {
            // Update text displays
            epsilonText.text = $"Epsilon: {agent.GetEpsilon():F3}";
            rewardText.text = $"Reward: {agent.GetCurrentReward():F2} (Avg: {agent.GetAverageReward():F2})";
            stepsText.text = $"Steps: {agent.GetTrainingSteps()}";
            
            // Update loss display and graph
            float currentLoss = agent.GetCurrentLoss();
            lossText.text = $"Loss: {currentLoss:F4}";
            
            lossHistory.Add(currentLoss);
            if (lossHistory.Count > 100)
            {
                lossHistory.RemoveAt(0);
            }
            
            UpdateLossGraph();
            
            yield return new WaitForSeconds(0.1f);
        }
    }
    
    private void UpdateLossGraph()
    {
        if (lossHistory.Count < 2) return;
        
        lossGraph.positionCount = lossHistory.Count;
        
        float minLoss = lossHistory.Min();
        float maxLoss = lossHistory.Max();
        float range = maxLoss - minLoss;
        
        for (int i = 0; i < lossHistory.Count; i++)
        {
            float x = (float)i / (lossHistory.Count - 1);
            float y = range > 0 ? (lossHistory[i] - minLoss) / range : 0.5f;
            lossGraph.SetPosition(i, new Vector3(x * 10f, y * 5f, 0f));
        }
    }
}
```

## Example: State Preprocessing

### Feature Engineering
```csharp
public class StatePreprocessor : MonoBehaviour
{
    private float[] rawState;
    private float[] processedState;
    
    public float[] PreprocessState(float[] rawState)
    {
        // Normalize position to [-1, 1]
        float[] normalizedPos = new float[3];
        normalizedPos[0] = Mathf.Clamp(rawState[0] / 10f, -1f, 1f);
        normalizedPos[1] = Mathf.Clamp(rawState[1] / 10f, -1f, 1f);
        normalizedPos[2] = Mathf.Clamp(rawState[2] / 10f, -1f, 1f);
        
        // Normalize velocity
        float[] normalizedVel = new float[3];
        normalizedVel[0] = Mathf.Clamp(rawState[3] / 20f, -1f, 1f);
        normalizedVel[1] = Mathf.Clamp(rawState[4] / 20f, -1f, 1f);
        normalizedVel[2] = Mathf.Clamp(rawState[5] / 20f, -1f, 1f);
        
        // Calculate relative features
        float distanceToTarget = Vector3.Distance(
            new Vector3(rawState[0], rawState[1], rawState[2]),
            targetPosition
        );
        float normalizedDistance = Mathf.Clamp(distanceToTarget / 50f, 0f, 1f);
        
        // Combine all features
        processedState = new float[] {
            normalizedPos[0], normalizedPos[1], normalizedPos[2],
            normalizedVel[0], normalizedVel[1], normalizedVel[2],
            normalizedDistance,
            // ... more processed features
        };
        
        return processedState;
    }
}
```

## Example: Reward Shaping

### Sophisticated Reward Function
```csharp
public class RewardShaper : MonoBehaviour
{
    public float distanceRewardWeight = -0.1f;
    public float velocityRewardWeight = 0.01f;
    public float timePenaltyWeight = -0.01f;
    public float collisionPenalty = -10f;
    public float goalReward = 100f;
    
    private Vector3 lastPosition;
    private float episodeStartTime;
    
    public float CalculateReward(Vector3 currentPosition, Vector3 targetPosition, 
                                bool hasCollided, bool reachedGoal)
    {
        float reward = 0f;
        
        // Distance-based reward (closer to target = higher reward)
        float distanceToTarget = Vector3.Distance(currentPosition, targetPosition);
        reward += distanceToTarget * distanceRewardWeight;
        
        // Velocity-based reward (encourage movement toward target)
        Vector3 directionToTarget = (targetPosition - currentPosition).normalized;
        Vector3 currentVelocity = (currentPosition - lastPosition) / Time.deltaTime;
        float velocityAlignment = Vector3.Dot(directionToTarget, currentVelocity.normalized);
        reward += velocityAlignment * velocityRewardWeight;
        
        // Time penalty (encourage efficiency)
        reward += timePenaltyWeight;
        
        // Collision penalty
        if (hasCollided)
        {
            reward += collisionPenalty;
        }
        
        // Goal reward
        if (reachedGoal)
        {
            reward += goalReward;
        }
        
        lastPosition = currentPosition;
        return reward;
    }
}
```

These examples demonstrate various ways to use the NeuralNetworkRL-Unity system for different scenarios, from simple game agents to complex multi-agent systems with hyperparameter tuning and performance monitoring.
