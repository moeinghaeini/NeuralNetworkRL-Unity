using UnityEngine;
using System.IO;
using System;

/// <summary>
/// Configuration manager for NeuralNetworkRL-Unity
/// Handles loading, saving, and validation of configuration files
/// </summary>
public class ConfigManager : MonoBehaviour
{
    [Header("Configuration Files")]
    public string configFileName = "rl_config.json";
    public string defaultConfigFileName = "default_config.json";
    
    [Header("Auto-load Settings")]
    public bool loadConfigOnStart = true;
    public bool saveConfigOnExit = true;
    public bool validateConfigOnLoad = true;
    
    [Header("Current Configuration")]
    public RL.RLConfig currentConfig;
    
    private string configPath;
    private string defaultConfigPath;
    
    #region Unity Lifecycle
    private void Awake()
    {
        InitializePaths();
        LoadDefaultConfig();
    }
    
    private void Start()
    {
        if (loadConfigOnStart)
        {
            LoadConfig();
        }
    }
    
    private void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus && saveConfigOnExit)
        {
            SaveConfig();
        }
    }
    
    private void OnApplicationFocus(bool hasFocus)
    {
        if (!hasFocus && saveConfigOnExit)
        {
            SaveConfig();
        }
    }
    
    private void OnDestroy()
    {
        if (saveConfigOnExit)
        {
            SaveConfig();
        }
    }
    #endregion
    
    #region Initialization
    private void InitializePaths()
    {
        configPath = Path.Combine(Application.persistentDataPath, configFileName);
        defaultConfigPath = Path.Combine(Application.streamingAssetsPath, defaultConfigFileName);
        
        // Create streaming assets directory if it doesn't exist
        if (!Directory.Exists(Application.streamingAssetsPath))
        {
            Directory.CreateDirectory(Application.streamingAssetsPath);
        }
    }
    
    private void LoadDefaultConfig()
    {
        try
        {
            if (File.Exists(defaultConfigPath))
            {
                string json = File.ReadAllText(defaultConfigPath);
                currentConfig = JsonUtility.FromJson<RL.RLConfig>(json);
                Debug.Log("Default configuration loaded successfully");
            }
            else
            {
                // Create default configuration
                currentConfig = CreateDefaultConfig();
                SaveDefaultConfig();
                Debug.Log("Default configuration created");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error loading default config: {e.Message}");
            currentConfig = CreateDefaultConfig();
        }
    }
    
    private RL.RLConfig CreateDefaultConfig()
    {
        return new RL.RLConfig
        {
            // Exploration settings
            epsilonStart = 1.0f,
            epsilonEnd = 0.01f,
            epsilonDecay = 0.995f,
            epsilonDecaySteps = 1000,
            
            // Learning settings
            learningRate = 0.001f,
            gamma = 0.99f,
            targetUpdateFrequency = 100,
            experienceReplaySize = 10000,
            batchSize = 32,
            
            // Advanced features
            useDoubleDQN = true,
            useTargetNetwork = true,
            usePrioritizedReplay = true,
            prioritizedReplayAlpha = 0.6f,
            prioritizedReplayBeta = 0.4f,
            prioritizedReplayBetaIncrement = 0.001f,
            
            // Network configuration
            networkConfig = new NN.NetworkConfig
            {
                numInputs = 14,
                numHiddens = 64,
                numOutputs = 66,
                numHiddenLayers = 2,
                learningRate = 0.001f,
                batchSize = 32,
                experienceReplaySize = 10000,
                useBatchNormalization = true,
                useDropout = true,
                dropoutRate = 0.2f,
                activationType = NN.ActivationType.ReLU,
                weightInitType = NN.WeightInitType.Xavier,
                optimizerType = NN.OptimizerType.Adam,
                gradientClipThreshold = 1.0f,
                adamBeta1 = 0.9f,
                adamBeta2 = 0.999f,
                adamEpsilon = 1e-8f
            }
        };
    }
    #endregion
    
    #region Configuration Management
    public void LoadConfig()
    {
        try
        {
            if (File.Exists(configPath))
            {
                string json = File.ReadAllText(configPath);
                var loadedConfig = JsonUtility.FromJson<RL.RLConfig>(json);
                
                if (validateConfigOnLoad && ValidateConfig(loadedConfig))
                {
                    currentConfig = loadedConfig;
                    Debug.Log("Configuration loaded successfully");
                }
                else
                {
                    Debug.LogWarning("Loaded configuration failed validation, using default");
                }
            }
            else
            {
                Debug.Log("No configuration file found, using default");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error loading configuration: {e.Message}");
        }
    }
    
    public void SaveConfig()
    {
        try
        {
            if (validateConfigOnLoad && !ValidateConfig(currentConfig))
            {
                Debug.LogWarning("Current configuration failed validation, not saving");
                return;
            }
            
            string json = JsonUtility.ToJson(currentConfig, true);
            File.WriteAllText(configPath, json);
            Debug.Log($"Configuration saved to: {configPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving configuration: {e.Message}");
        }
    }
    
    public void SaveDefaultConfig()
    {
        try
        {
            string json = JsonUtility.ToJson(currentConfig, true);
            File.WriteAllText(defaultConfigPath, json);
            Debug.Log($"Default configuration saved to: {defaultConfigPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving default configuration: {e.Message}");
        }
    }
    
    public void ResetToDefault()
    {
        currentConfig = CreateDefaultConfig();
        Debug.Log("Configuration reset to default");
    }
    #endregion
    
    #region Configuration Validation
    public bool ValidateConfig(RL.RLConfig config)
    {
        if (config == null)
        {
            Debug.LogError("Configuration is null");
            return false;
        }
        
        // Validate exploration parameters
        if (config.epsilonStart < 0f || config.epsilonStart > 1f)
        {
            Debug.LogError($"Invalid epsilonStart: {config.epsilonStart}");
            return false;
        }
        
        if (config.epsilonEnd < 0f || config.epsilonEnd > 1f)
        {
            Debug.LogError($"Invalid epsilonEnd: {config.epsilonEnd}");
            return false;
        }
        
        if (config.epsilonDecay <= 0f || config.epsilonDecay >= 1f)
        {
            Debug.LogError($"Invalid epsilonDecay: {config.epsilonDecay}");
            return false;
        }
        
        // Validate learning parameters
        if (config.learningRate <= 0f || config.learningRate > 1f)
        {
            Debug.LogError($"Invalid learningRate: {config.learningRate}");
            return false;
        }
        
        if (config.gamma < 0f || config.gamma > 1f)
        {
            Debug.LogError($"Invalid gamma: {config.gamma}");
            return false;
        }
        
        if (config.batchSize <= 0 || config.batchSize > 1000)
        {
            Debug.LogError($"Invalid batchSize: {config.batchSize}");
            return false;
        }
        
        if (config.experienceReplaySize <= 0 || config.experienceReplaySize > 1000000)
        {
            Debug.LogError($"Invalid experienceReplaySize: {config.experienceReplaySize}");
            return false;
        }
        
        // Validate network configuration
        if (!ValidateNetworkConfig(config.networkConfig))
        {
            return false;
        }
        
        return true;
    }
    
    private bool ValidateNetworkConfig(NN.NetworkConfig networkConfig)
    {
        if (networkConfig == null)
        {
            Debug.LogError("Network configuration is null");
            return false;
        }
        
        if (networkConfig.numInputs <= 0)
        {
            Debug.LogError($"Invalid numInputs: {networkConfig.numInputs}");
            return false;
        }
        
        if (networkConfig.numHiddens <= 0)
        {
            Debug.LogError($"Invalid numHiddens: {networkConfig.numHiddens}");
            return false;
        }
        
        if (networkConfig.numOutputs <= 0)
        {
            Debug.LogError($"Invalid numOutputs: {networkConfig.numOutputs}");
            return false;
        }
        
        if (networkConfig.numHiddenLayers <= 0 || networkConfig.numHiddenLayers > 10)
        {
            Debug.LogError($"Invalid numHiddenLayers: {networkConfig.numHiddenLayers}");
            return false;
        }
        
        if (networkConfig.learningRate <= 0f || networkConfig.learningRate > 1f)
        {
            Debug.LogError($"Invalid network learningRate: {networkConfig.learningRate}");
            return false;
        }
        
        if (networkConfig.dropoutRate < 0f || networkConfig.dropoutRate >= 1f)
        {
            Debug.LogError($"Invalid dropoutRate: {networkConfig.dropoutRate}");
            return false;
        }
        
        return true;
    }
    #endregion
    
    #region Configuration Presets
    public void LoadPreset(string presetName)
    {
        switch (presetName.ToLower())
        {
            case "fast":
                LoadFastPreset();
                break;
            case "stable":
                LoadStablePreset();
                break;
            case "exploration":
                LoadExplorationPreset();
                break;
            case "exploitation":
                LoadExploitationPreset();
                break;
            default:
                Debug.LogWarning($"Unknown preset: {presetName}");
                break;
        }
    }
    
    private void LoadFastPreset()
    {
        currentConfig.learningRate = 0.01f;
        currentConfig.batchSize = 64;
        currentConfig.targetUpdateFrequency = 50;
        currentConfig.networkConfig.learningRate = 0.01f;
        currentConfig.networkConfig.batchSize = 64;
        Debug.Log("Fast preset loaded");
    }
    
    private void LoadStablePreset()
    {
        currentConfig.learningRate = 0.0001f;
        currentConfig.batchSize = 16;
        currentConfig.targetUpdateFrequency = 200;
        currentConfig.networkConfig.learningRate = 0.0001f;
        currentConfig.networkConfig.batchSize = 16;
        currentConfig.networkConfig.useBatchNormalization = true;
        currentConfig.networkConfig.useDropout = true;
        Debug.Log("Stable preset loaded");
    }
    
    private void LoadExplorationPreset()
    {
        currentConfig.epsilonStart = 1.0f;
        currentConfig.epsilonEnd = 0.1f;
        currentConfig.epsilonDecay = 0.99f;
        Debug.Log("Exploration preset loaded");
    }
    
    private void LoadExploitationPreset()
    {
        currentConfig.epsilonStart = 0.1f;
        currentConfig.epsilonEnd = 0.001f;
        currentConfig.epsilonDecay = 0.999f;
        Debug.Log("Exploitation preset loaded");
    }
    #endregion
    
    #region Public API
    public RL.RLConfig GetConfig() => currentConfig;
    
    public void SetConfig(RL.RLConfig config)
    {
        if (ValidateConfig(config))
        {
            currentConfig = config;
            Debug.Log("Configuration updated");
        }
        else
        {
            Debug.LogError("Invalid configuration provided");
        }
    }
    
    public string GetConfigAsJson() => JsonUtility.ToJson(currentConfig, true);
    
    public void LoadConfigFromJson(string json)
    {
        try
        {
            var config = JsonUtility.FromJson<RL.RLConfig>(json);
            SetConfig(config);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error loading config from JSON: {e.Message}");
        }
    }
    
    public void ExportConfig(string filePath)
    {
        try
        {
            string json = JsonUtility.ToJson(currentConfig, true);
            File.WriteAllText(filePath, json);
            Debug.Log($"Configuration exported to: {filePath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error exporting configuration: {e.Message}");
        }
    }
    
    public void ImportConfig(string filePath)
    {
        try
        {
            if (File.Exists(filePath))
            {
                string json = File.ReadAllText(filePath);
                LoadConfigFromJson(json);
                Debug.Log($"Configuration imported from: {filePath}");
            }
            else
            {
                Debug.LogError($"Configuration file not found: {filePath}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error importing configuration: {e.Message}");
        }
    }
    #endregion
}
