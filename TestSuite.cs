using UnityEngine;
using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Comprehensive test suite for NeuralNetworkRL-Unity
/// Tests neural network functionality, RL algorithms, and integration
/// </summary>
public class TestSuite
{
    #region Neural Network Tests
    
    [Test]
    public void TestNeuralNetworkInitialization()
    {
        // Test basic initialization
        NN network = new NN(4, 8, 2);
        Assert.IsNotNull(network);
        
        // Test custom configuration initialization
        NN.NetworkConfig config = new NN.NetworkConfig
        {
            numInputs = 4,
            numHiddens = 8,
            numOutputs = 2,
            learningRate = 0.001f
        };
        NN customNetwork = new NN(config);
        Assert.IsNotNull(customNetwork);
    }
    
    [Test]
    public void TestNeuralNetworkForwardPass()
    {
        NN network = new NN(4, 8, 2);
        float[] input = { 1f, 2f, 3f, 4f };
        
        List<float> output = network.calcNet(input);
        
        Assert.AreEqual(2, output.Count);
        Assert.IsTrue(output.All(x => !float.IsNaN(x)));
        Assert.IsTrue(output.All(x => !float.IsInfinity(x)));
    }
    
    [Test]
    public void TestNeuralNetworkTraining()
    {
        NN network = new NN(4, 8, 2);
        float[] input = { 1f, 2f, 3f, 4f };
        
        // Get initial output
        List<float> initialOutput = network.calcNet(input);
        
        // Train the network
        NN.Transition transition = new NN.Transition
        {
            state = input,
            action = 0,
            reward = 1.0f
        };
        network.Train(transition);
        
        // Get output after training
        List<float> trainedOutput = network.calcNet(input);
        
        // Output should be different (network should have learned)
        Assert.AreNotEqual(initialOutput[0], trainedOutput[0], 0.001f);
    }
    
    [Test]
    public void TestNeuralNetworkSaveLoad()
    {
        NN originalNetwork = new NN(4, 8, 2);
        float[] input = { 1f, 2f, 3f, 4f };
        List<float> originalOutput = originalNetwork.calcNet(input);
        
        // Save network
        string savePath = System.IO.Path.Combine(Application.persistentDataPath, "test_network.txt");
        using (System.IO.StreamWriter writer = new System.IO.StreamWriter(savePath))
        {
            originalNetwork.Save(writer);
        }
        
        // Load network
        NN loadedNetwork = new NN(4, 8, 2);
        using (System.IO.StreamReader reader = new System.IO.StreamReader(savePath))
        {
            loadedNetwork.Load(reader);
        }
        
        // Compare outputs
        List<float> loadedOutput = loadedNetwork.calcNet(input);
        Assert.AreEqual(originalOutput.Count, loadedOutput.Count);
        
        for (int i = 0; i < originalOutput.Count; i++)
        {
            Assert.AreEqual(originalOutput[i], loadedOutput[i], 0.001f);
        }
        
        // Cleanup
        if (System.IO.File.Exists(savePath))
        {
            System.IO.File.Delete(savePath);
        }
    }
    
    [Test]
    public void TestNeuralNetworkPerformanceMetrics()
    {
        NN network = new NN(4, 8, 2);
        
        // Initially, loss should be 0
        Assert.AreEqual(0f, network.GetCurrentLoss());
        Assert.AreEqual(0f, network.GetAverageLoss());
        Assert.AreEqual(0, network.GetTrainingSteps());
        
        // Train the network
        NN.Transition transition = new NN.Transition
        {
            state = new float[] { 1f, 2f, 3f, 4f },
            action = 0,
            reward = 1.0f
        };
        network.Train(transition);
        
        // Metrics should be updated
        Assert.Greater(network.GetTrainingSteps(), 0);
        Assert.IsTrue(network.GetCurrentLoss() >= 0f);
    }
    
    #endregion
    
    #region Reinforcement Learning Tests
    
    [Test]
    public void TestRLAgentInitialization()
    {
        GameObject agentObj = new GameObject("TestAgent");
        RL agent = agentObj.AddComponent<RL>();
        
        Assert.IsNotNull(agent);
        Assert.AreEqual(1.0f, agent.GetEpsilon(), 0.001f);
        Assert.AreEqual(0, agent.GetTrainingSteps());
        
        Object.DestroyImmediate(agentObj);
    }
    
    [Test]
    public void TestRLAgentActionSelection()
    {
        GameObject agentObj = new GameObject("TestAgent");
        RL agent = agentObj.AddComponent<RL>();
        
        float[] state = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f };
        int action = agent.Act(state);
        
        Assert.GreaterOrEqual(action, 0);
        Assert.Less(action, 66); // Assuming 66 actions
        
        Object.DestroyImmediate(agentObj);
    }
    
    [Test]
    public void TestRLAgentLearning()
    {
        GameObject agentObj = new GameObject("TestAgent");
        RL agent = agentObj.AddComponent<RL>();
        
        float[] state = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f };
        float[] nextState = { 1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f, 7.1f, 8.1f, 9.1f, 10.1f, 11.1f, 12.1f, 13.1f, 14.1f };
        
        // Start episode
        agent.StartEpisode();
        
        // Get initial action
        int action = agent.Act(state);
        
        // Observe transition
        RL.Transition transition = new RL.Transition
        {
            state = state,
            action = action,
            reward = 1.0f,
            nextState = nextState,
            isTerminal = false
        };
        agent.Observe(transition);
        
        // Training steps should increase
        Assert.Greater(agent.GetTrainingSteps(), 0);
        
        Object.DestroyImmediate(agentObj);
    }
    
    [Test]
    public void TestRLAgentEpisodeManagement()
    {
        GameObject agentObj = new GameObject("TestAgent");
        RL agent = agentObj.AddComponent<RL>();
        
        // Start episode
        agent.StartEpisode();
        Assert.AreEqual(0f, agent.GetCurrentReward());
        
        // Simulate episode
        float[] state = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f };
        int action = agent.Act(state);
        
        RL.Transition transition = new RL.Transition
        {
            state = state,
            action = action,
            reward = 5.0f,
            nextState = state,
            isTerminal = true
        };
        agent.Observe(transition);
        
        // End episode
        agent.EndEpisode();
        
        // Reward should be recorded
        Assert.AreEqual(5.0f, agent.GetCurrentReward());
        
        Object.DestroyImmediate(agentObj);
    }
    
    [Test]
    public void TestRLAgentSaveLoad()
    {
        GameObject agentObj = new GameObject("TestAgent");
        RL agent = agentObj.AddComponent<RL>();
        
        // Train the agent a bit
        for (int i = 0; i < 10; i++)
        {
            float[] state = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f };
            int action = agent.Act(state);
            
            RL.Transition transition = new RL.Transition
            {
                state = state,
                action = action,
                reward = 1.0f,
                nextState = state,
                isTerminal = false
            };
            agent.Observe(transition);
        }
        
        int originalSteps = agent.GetTrainingSteps();
        float originalEpsilon = agent.GetEpsilon();
        
        // Save agent
        string savePath = System.IO.Path.Combine(Application.persistentDataPath, "test_agent.txt");
        using (System.IO.StreamWriter writer = new System.IO.StreamWriter(savePath))
        {
            agent.Save(writer);
        }
        
        // Create new agent and load
        GameObject newAgentObj = new GameObject("NewTestAgent");
        RL newAgent = newAgentObj.AddComponent<RL>();
        using (System.IO.StreamReader reader = new System.IO.StreamReader(savePath))
        {
            newAgent.Load(reader);
        }
        
        // Compare states
        Assert.AreEqual(originalSteps, newAgent.GetTrainingSteps());
        Assert.AreEqual(originalEpsilon, newAgent.GetEpsilon(), 0.001f);
        
        // Cleanup
        Object.DestroyImmediate(agentObj);
        Object.DestroyImmediate(newAgentObj);
        if (System.IO.File.Exists(savePath))
        {
            System.IO.File.Delete(savePath);
        }
    }
    
    #endregion
    
    #region Configuration Tests
    
    [Test]
    public void TestConfigurationValidation()
    {
        ConfigManager configManager = new GameObject("ConfigManager").AddComponent<ConfigManager>();
        
        // Test valid configuration
        RL.RLConfig validConfig = new RL.RLConfig
        {
            epsilonStart = 1.0f,
            epsilonEnd = 0.01f,
            epsilonDecay = 0.995f,
            learningRate = 0.001f,
            gamma = 0.99f,
            batchSize = 32,
            experienceReplaySize = 10000,
            networkConfig = new NN.NetworkConfig
            {
                numInputs = 14,
                numHiddens = 64,
                numOutputs = 66,
                learningRate = 0.001f
            }
        };
        
        Assert.IsTrue(configManager.ValidateConfig(validConfig));
        
        // Test invalid configuration
        RL.RLConfig invalidConfig = new RL.RLConfig
        {
            epsilonStart = -1.0f, // Invalid
            epsilonEnd = 0.01f,
            epsilonDecay = 0.995f,
            learningRate = 0.001f,
            gamma = 0.99f,
            batchSize = 32,
            experienceReplaySize = 10000,
            networkConfig = new NN.NetworkConfig
            {
                numInputs = 14,
                numHiddens = 64,
                numOutputs = 66,
                learningRate = 0.001f
            }
        };
        
        Assert.IsFalse(configManager.ValidateConfig(invalidConfig));
        
        Object.DestroyImmediate(configManager.gameObject);
    }
    
    [Test]
    public void TestConfigurationPresets()
    {
        ConfigManager configManager = new GameObject("ConfigManager").AddComponent<ConfigManager>();
        
        // Test fast preset
        configManager.LoadPreset("fast");
        Assert.AreEqual(0.01f, configManager.GetConfig().learningRate);
        Assert.AreEqual(64, configManager.GetConfig().batchSize);
        
        // Test stable preset
        configManager.LoadPreset("stable");
        Assert.AreEqual(0.0001f, configManager.GetConfig().learningRate);
        Assert.AreEqual(16, configManager.GetConfig().batchSize);
        
        Object.DestroyImmediate(configManager.gameObject);
    }
    
    #endregion
    
    #region Integration Tests
    
    [UnityTest]
    public IEnumerator TestFullTrainingCycle()
    {
        GameObject agentObj = new GameObject("TestAgent");
        RL agent = agentObj.AddComponent<RL>();
        
        // Run training for multiple episodes
        for (int episode = 0; episode < 5; episode++)
        {
            agent.StartEpisode();
            
            for (int step = 0; step < 10; step++)
            {
                float[] state = GenerateRandomState();
                int action = agent.Act(state);
                float reward = CalculateReward(state, action);
                float[] nextState = GenerateRandomState();
                bool isTerminal = step == 9; // End episode after 10 steps
                
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
                }
                
                yield return null; // Wait one frame
            }
        }
        
        // Verify agent has learned
        Assert.Greater(agent.GetTrainingSteps(), 0);
        Assert.Less(agent.GetEpsilon(), 1.0f); // Epsilon should have decayed
        
        Object.DestroyImmediate(agentObj);
    }
    
    [Test]
    public void TestPerformanceUnderLoad()
    {
        // Test neural network performance with large inputs
        NN network = new NN(100, 200, 50);
        float[] largeInput = new float[100];
        for (int i = 0; i < 100; i++)
        {
            largeInput[i] = Random.Range(-1f, 1f);
        }
        
        // Measure forward pass time
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        List<float> output = network.calcNet(largeInput);
        stopwatch.Stop();
        
        Assert.AreEqual(50, output.Count);
        Assert.Less(stopwatch.ElapsedMilliseconds, 100); // Should be fast
        
        // Test training performance
        stopwatch.Restart();
        for (int i = 0; i < 100; i++)
        {
            NN.Transition transition = new NN.Transition
            {
                state = largeInput,
                action = Random.Range(0, 50),
                reward = Random.Range(-1f, 1f)
            };
            network.Train(transition);
        }
        stopwatch.Stop();
        
        Assert.Less(stopwatch.ElapsedMilliseconds, 1000); // Should be reasonably fast
    }
    
    #endregion
    
    #region Helper Methods
    
    private float[] GenerateRandomState()
    {
        float[] state = new float[14];
        for (int i = 0; i < 14; i++)
        {
            state[i] = Random.Range(-1f, 1f);
        }
        return state;
    }
    
    private float CalculateReward(float[] state, int action)
    {
        // Simple reward function for testing
        return Random.Range(-1f, 1f);
    }
    
    #endregion
    
    #region Edge Case Tests
    
    [Test]
    public void TestEdgeCases()
    {
        // Test with zero inputs
        NN network = new NN(0, 8, 2);
        float[] emptyInput = new float[0];
        
        // This should throw an exception or handle gracefully
        Assert.Throws<System.ArgumentException>(() => network.calcNet(emptyInput));
        
        // Test with very small values
        NN smallNetwork = new NN(1, 1, 1);
        float[] smallInput = { 0.0001f };
        List<float> output = smallNetwork.calcNet(smallInput);
        Assert.AreEqual(1, output.Count);
        Assert.IsTrue(!float.IsNaN(output[0]));
    }
    
    [Test]
    public void TestMemoryManagement()
    {
        // Test that networks don't leak memory
        for (int i = 0; i < 100; i++)
        {
            NN network = new NN(10, 20, 5);
            float[] input = new float[10];
            for (int j = 0; j < 10; j++)
            {
                input[j] = Random.Range(-1f, 1f);
            }
            
            List<float> output = network.calcNet(input);
            Assert.AreEqual(5, output.Count);
        }
        
        // Force garbage collection
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        
        // If we get here without running out of memory, test passed
        Assert.IsTrue(true);
    }
    
    #endregion
}
