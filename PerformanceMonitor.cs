using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;
using System.Linq;

/// <summary>
/// Comprehensive performance monitoring and logging system for NeuralNetworkRL-Unity
/// Tracks metrics, generates reports, and provides real-time monitoring
/// </summary>
public class PerformanceMonitor : MonoBehaviour
{
    [Header("Monitoring Settings")]
    public bool enableMonitoring = true;
    public bool enableLogging = true;
    public bool enableRealTimeDisplay = true;
    public float updateInterval = 0.1f;
    
    [Header("Logging Settings")]
    public string logFileName = "performance_log.txt";
    public bool logToFile = true;
    public bool logToConsole = true;
    public int maxLogEntries = 10000;
    
    [Header("Metrics Tracking")]
    public bool trackTrainingMetrics = true;
    public bool trackEpisodeMetrics = true;
    public bool trackSystemMetrics = true;
    public bool trackMemoryUsage = true;
    
    [Header("UI Components")]
    public Text performanceText;
    public LineRenderer lossGraph;
    public LineRenderer rewardGraph;
    public Slider performanceSlider;
    
    // Performance data structures
    private List<PerformanceEntry> performanceHistory = new List<PerformanceEntry>();
    private List<float> lossHistory = new List<float>();
    private List<float> rewardHistory = new List<float>();
    private List<float> epsilonHistory = new List<float>();
    private List<float> fpsHistory = new List<float>();
    
    // Current metrics
    private float currentFPS = 0f;
    private float averageFPS = 0f;
    private long memoryUsage = 0;
    private float cpuUsage = 0f;
    
    // Monitoring state
    private float lastUpdateTime = 0f;
    private int frameCount = 0;
    private float frameTime = 0f;
    
    // Logging
    private StreamWriter logWriter;
    private string logPath;
    
    #region Unity Lifecycle
    private void Start()
    {
        InitializeMonitoring();
        SetupLogging();
        SetupUI();
    }
    
    private void Update()
    {
        if (enableMonitoring)
        {
            UpdateSystemMetrics();
            
            if (Time.time - lastUpdateTime >= updateInterval)
            {
                UpdatePerformanceMetrics();
                lastUpdateTime = Time.time;
            }
        }
    }
    
    private void OnDestroy()
    {
        CloseLogging();
    }
    
    private void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus)
        {
            SavePerformanceReport();
        }
    }
    #endregion
    
    #region Initialization
    private void InitializeMonitoring()
    {
        performanceHistory.Clear();
        lossHistory.Clear();
        rewardHistory.Clear();
        epsilonHistory.Clear();
        fpsHistory.Clear();
        
        lastUpdateTime = Time.time;
        frameCount = 0;
        frameTime = 0f;
        
        Debug.Log("Performance monitoring initialized");
    }
    
    private void SetupLogging()
    {
        if (enableLogging && logToFile)
        {
            logPath = Path.Combine(Application.persistentDataPath, logFileName);
            try
            {
                logWriter = new StreamWriter(logPath, true);
                LogMessage("Performance monitoring started", LogLevel.Info);
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to setup logging: {e.Message}");
            }
        }
    }
    
    private void SetupUI()
    {
        if (enableRealTimeDisplay)
        {
            SetupPerformanceGraphs();
            UpdatePerformanceDisplay();
        }
    }
    
    private void SetupPerformanceGraphs()
    {
        if (lossGraph != null)
        {
            lossGraph.material = new Material(Shader.Find("Sprites/Default"));
            lossGraph.color = Color.red;
            lossGraph.startWidth = 0.1f;
            lossGraph.endWidth = 0.1f;
        }
        
        if (rewardGraph != null)
        {
            rewardGraph.material = new Material(Shader.Find("Sprites/Default"));
            rewardGraph.color = Color.green;
            rewardGraph.startWidth = 0.1f;
            rewardGraph.endWidth = 0.1f;
        }
    }
    #endregion
    
    #region Metrics Collection
    private void UpdateSystemMetrics()
    {
        frameCount++;
        frameTime += Time.unscaledDeltaTime;
        
        // Calculate FPS
        if (frameTime >= 1.0f)
        {
            currentFPS = frameCount / frameTime;
            fpsHistory.Add(currentFPS);
            
            if (fpsHistory.Count > 100)
            {
                fpsHistory.RemoveAt(0);
            }
            
            averageFPS = fpsHistory.Average();
            
            frameCount = 0;
            frameTime = 0f;
        }
        
        // Memory usage
        if (trackMemoryUsage)
        {
            memoryUsage = System.GC.GetTotalMemory(false);
        }
    }
    
    private void UpdatePerformanceMetrics()
    {
        // Find RL agent in scene
        RL agent = FindObjectOfType<RL>();
        if (agent == null) return;
        
        // Collect agent metrics
        float currentLoss = agent.GetCurrentLoss();
        float currentReward = agent.GetCurrentReward();
        float currentEpsilon = agent.GetEpsilon();
        int trainingSteps = agent.GetTrainingSteps();
        
        // Update histories
        if (trackTrainingMetrics)
        {
            lossHistory.Add(currentLoss);
            if (lossHistory.Count > 1000)
            {
                lossHistory.RemoveAt(0);
            }
        }
        
        if (trackEpisodeMetrics)
        {
            rewardHistory.Add(currentReward);
            if (rewardHistory.Count > 1000)
            {
                rewardHistory.RemoveAt(0);
            }
            
            epsilonHistory.Add(currentEpsilon);
            if (epsilonHistory.Count > 1000)
            {
                epsilonHistory.RemoveAt(0);
            }
        }
        
        // Create performance entry
        PerformanceEntry entry = new PerformanceEntry
        {
            timestamp = Time.time,
            fps = currentFPS,
            averageFPS = averageFPS,
            memoryUsage = memoryUsage,
            trainingLoss = currentLoss,
            episodeReward = currentReward,
            epsilon = currentEpsilon,
            trainingSteps = trainingSteps
        };
        
        performanceHistory.Add(entry);
        
        // Limit history size
        if (performanceHistory.Count > maxLogEntries)
        {
            performanceHistory.RemoveAt(0);
        }
        
        // Update UI
        if (enableRealTimeDisplay)
        {
            UpdatePerformanceDisplay();
            UpdatePerformanceGraphs();
        }
        
        // Log metrics
        if (enableLogging)
        {
            LogPerformanceEntry(entry);
        }
    }
    #endregion
    
    #region UI Updates
    private void UpdatePerformanceDisplay()
    {
        if (performanceText == null) return;
        
        RL agent = FindObjectOfType<RL>();
        if (agent == null) return;
        
        string performanceInfo = $"FPS: {currentFPS:F1} (Avg: {averageFPS:F1})\n" +
                               $"Memory: {memoryUsage / 1024 / 1024:F1} MB\n" +
                               $"Loss: {agent.GetCurrentLoss():F4}\n" +
                               $"Reward: {agent.GetCurrentReward():F2}\n" +
                               $"Epsilon: {agent.GetEpsilon():F3}\n" +
                               $"Steps: {agent.GetTrainingSteps()}\n" +
                               $"Buffer: {agent.GetExperienceBufferSize()}";
        
        performanceText.text = performanceInfo;
    }
    
    private void UpdatePerformanceGraphs()
    {
        UpdateLossGraph();
        UpdateRewardGraph();
    }
    
    private void UpdateLossGraph()
    {
        if (lossGraph == null || lossHistory.Count < 2) return;
        
        lossGraph.positionCount = lossHistory.Count;
        
        float minLoss = lossHistory.Min();
        float maxLoss = lossHistory.Max();
        float range = maxLoss - minLoss;
        
        if (range <= 0) range = 1f;
        
        for (int i = 0; i < lossHistory.Count; i++)
        {
            float x = (float)i / (lossHistory.Count - 1) * 10f;
            float y = (lossHistory[i] - minLoss) / range * 5f;
            lossGraph.SetPosition(i, new Vector3(x, y, 0f));
        }
    }
    
    private void UpdateRewardGraph()
    {
        if (rewardGraph == null || rewardHistory.Count < 2) return;
        
        rewardGraph.positionCount = rewardHistory.Count;
        
        float minReward = rewardHistory.Min();
        float maxReward = rewardHistory.Max();
        float range = maxReward - minReward;
        
        if (range <= 0) range = 1f;
        
        for (int i = 0; i < rewardHistory.Count; i++)
        {
            float x = (float)i / (rewardHistory.Count - 1) * 10f;
            float y = (rewardHistory[i] - minReward) / range * 5f;
            rewardGraph.SetPosition(i, new Vector3(x, y + 6f, 0f)); // Offset above loss graph
        }
    }
    #endregion
    
    #region Logging
    private void LogMessage(string message, LogLevel level)
    {
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        string logEntry = $"[{timestamp}] [{level}] {message}";
        
        if (logToConsole)
        {
            switch (level)
            {
                case LogLevel.Error:
                    Debug.LogError(logEntry);
                    break;
                case LogLevel.Warning:
                    Debug.LogWarning(logEntry);
                    break;
                default:
                    Debug.Log(logEntry);
                    break;
            }
        }
        
        if (logToFile && logWriter != null)
        {
            try
            {
                logWriter.WriteLine(logEntry);
                logWriter.Flush();
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to write to log file: {e.Message}");
            }
        }
    }
    
    private void LogPerformanceEntry(PerformanceEntry entry)
    {
        string logEntry = $"PERF: FPS={entry.fps:F1}, Memory={entry.memoryUsage / 1024 / 1024:F1}MB, " +
                         $"Loss={entry.trainingLoss:F4}, Reward={entry.episodeReward:F2}, " +
                         $"Epsilon={entry.epsilon:F3}, Steps={entry.trainingSteps}";
        LogMessage(logEntry, LogLevel.Info);
    }
    
    private void CloseLogging()
    {
        if (logWriter != null)
        {
            try
            {
                LogMessage("Performance monitoring stopped", LogLevel.Info);
                logWriter.Close();
                logWriter = null;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error closing log file: {e.Message}");
            }
        }
    }
    #endregion
    
    #region Reports and Analysis
    public void SavePerformanceReport()
    {
        try
        {
            string reportPath = Path.Combine(Application.persistentDataPath, "performance_report.txt");
            using (StreamWriter writer = new StreamWriter(reportPath))
            {
                writer.WriteLine("=== NeuralNetworkRL-Unity Performance Report ===");
                writer.WriteLine($"Generated: {DateTime.Now}");
                writer.WriteLine($"Total Entries: {performanceHistory.Count}");
                writer.WriteLine();
                
                // Summary statistics
                if (performanceHistory.Count > 0)
                {
                    var latest = performanceHistory.Last();
                    writer.WriteLine("=== Current Status ===");
                    writer.WriteLine($"FPS: {latest.fps:F1} (Average: {latest.averageFPS:F1})");
                    writer.WriteLine($"Memory Usage: {latest.memoryUsage / 1024 / 1024:F1} MB");
                    writer.WriteLine($"Training Loss: {latest.trainingLoss:F4}");
                    writer.WriteLine($"Episode Reward: {latest.episodeReward:F2}");
                    writer.WriteLine($"Epsilon: {latest.epsilon:F3}");
                    writer.WriteLine($"Training Steps: {latest.trainingSteps}");
                    writer.WriteLine();
                }
                
                // Performance trends
                if (fpsHistory.Count > 0)
                {
                    writer.WriteLine("=== Performance Trends ===");
                    writer.WriteLine($"Average FPS: {fpsHistory.Average():F1}");
                    writer.WriteLine($"Min FPS: {fpsHistory.Min():F1}");
                    writer.WriteLine($"Max FPS: {fpsHistory.Max():F1}");
                    writer.WriteLine();
                }
                
                if (lossHistory.Count > 0)
                {
                    writer.WriteLine("=== Training Trends ===");
                    writer.WriteLine($"Average Loss: {lossHistory.Average():F4}");
                    writer.WriteLine($"Min Loss: {lossHistory.Min():F4}");
                    writer.WriteLine($"Max Loss: {lossHistory.Max():F4}");
                    writer.WriteLine();
                }
                
                if (rewardHistory.Count > 0)
                {
                    writer.WriteLine("=== Reward Trends ===");
                    writer.WriteLine($"Average Reward: {rewardHistory.Average():F2}");
                    writer.WriteLine($"Min Reward: {rewardHistory.Min():F2}");
                    writer.WriteLine($"Max Reward: {rewardHistory.Max():F2}");
                    writer.WriteLine();
                }
                
                // Detailed history (last 100 entries)
                writer.WriteLine("=== Recent Performance History ===");
                writer.WriteLine("Timestamp\tFPS\tMemory(MB)\tLoss\tReward\tEpsilon\tSteps");
                
                int startIndex = Math.Max(0, performanceHistory.Count - 100);
                for (int i = startIndex; i < performanceHistory.Count; i++)
                {
                    var entry = performanceHistory[i];
                    writer.WriteLine($"{entry.timestamp:F1}\t{entry.fps:F1}\t" +
                                   $"{entry.memoryUsage / 1024 / 1024:F1}\t" +
                                   $"{entry.trainingLoss:F4}\t{entry.episodeReward:F2}\t" +
                                   $"{entry.epsilon:F3}\t{entry.trainingSteps}");
                }
            }
            
            Debug.Log($"Performance report saved to: {reportPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving performance report: {e.Message}");
        }
    }
    
    public PerformanceSummary GetPerformanceSummary()
    {
        if (performanceHistory.Count == 0)
        {
            return new PerformanceSummary();
        }
        
        var latest = performanceHistory.Last();
        
        return new PerformanceSummary
        {
            currentFPS = latest.fps,
            averageFPS = latest.averageFPS,
            memoryUsageMB = latest.memoryUsage / 1024 / 1024,
            currentLoss = latest.trainingLoss,
            averageLoss = lossHistory.Count > 0 ? lossHistory.Average() : 0f,
            currentReward = latest.episodeReward,
            averageReward = rewardHistory.Count > 0 ? rewardHistory.Average() : 0f,
            currentEpsilon = latest.epsilon,
            trainingSteps = latest.trainingSteps,
            totalEntries = performanceHistory.Count
        };
    }
    #endregion
    
    #region Public API
    public void StartMonitoring()
    {
        enableMonitoring = true;
        LogMessage("Performance monitoring started", LogLevel.Info);
    }
    
    public void StopMonitoring()
    {
        enableMonitoring = false;
        LogMessage("Performance monitoring stopped", LogLevel.Info);
    }
    
    public void ClearHistory()
    {
        performanceHistory.Clear();
        lossHistory.Clear();
        rewardHistory.Clear();
        epsilonHistory.Clear();
        fpsHistory.Clear();
        LogMessage("Performance history cleared", LogLevel.Info);
    }
    
    public void ExportData(string filePath)
    {
        try
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine("Timestamp,FPS,AverageFPS,MemoryMB,Loss,Reward,Epsilon,Steps");
                
                foreach (var entry in performanceHistory)
                {
                    writer.WriteLine($"{entry.timestamp},{entry.fps},{entry.averageFPS}," +
                                   $"{entry.memoryUsage / 1024 / 1024},{entry.trainingLoss}," +
                                   $"{entry.episodeReward},{entry.epsilon},{entry.trainingSteps}");
                }
            }
            
            Debug.Log($"Performance data exported to: {filePath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error exporting performance data: {e.Message}");
        }
    }
    #endregion
}

#region Data Structures
[System.Serializable]
public class PerformanceEntry
{
    public float timestamp;
    public float fps;
    public float averageFPS;
    public long memoryUsage;
    public float trainingLoss;
    public float episodeReward;
    public float epsilon;
    public int trainingSteps;
}

[System.Serializable]
public class PerformanceSummary
{
    public float currentFPS;
    public float averageFPS;
    public float memoryUsageMB;
    public float currentLoss;
    public float averageLoss;
    public float currentReward;
    public float averageReward;
    public float currentEpsilon;
    public int trainingSteps;
    public int totalEntries;
}

public enum LogLevel
{
    Info,
    Warning,
    Error
}
#endregion
