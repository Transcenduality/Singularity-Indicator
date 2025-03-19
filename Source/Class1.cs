using System;
using System.Windows.Media;
using ATAS.Indicators;
using System.Drawing;
using System.Collections.Generic;
using System.ComponentModel;
using System.Collections.Concurrent;
using System.Windows;
using static System.Net.Mime.MediaTypeNames;
using System.IO;

public class SingularityHybridNN : Indicator
{
    private int inputSize = 17;
    private int[] hiddenSizes = { 128, 256, 512 };
    private int outputSize = 1;
    private int numberOfNetworks = 5;
    private double initialLearningRate = 0.005;
    private double adaptiveLearningRate;
    private double mutationRate = 0.01;
    private double adaptiveMutationRate;
    private double proliferationThreshold = 0.85;
    private double l2Regularization = 0.0005;
    private ValueDataSeries shortTermPrediction;
    private ValueDataSeries mediumTermPrediction;
    private ValueDataSeries longTermPrediction;

    private int shortTermHorizon = 5;
    private int mediumTermHorizon = 15;
    private int longTermHorizon = 30;
    private Random rand = new Random();

    private int lastEnqueuedBarShortTerm = -1;
    private int lastEnqueuedBarMediumTerm = -1;
    private int lastEnqueuedBarLongTerm = -1;

    private int lastProcessedBarBuffers = -1;
    private int lastProcessedBarLiveCandle = -1;
    private decimal lastLiveCandleClose = 0;

    private NeuralNetwork shortTermNetwork;
    private NeuralNetwork mediumTermNetwork;
    private NeuralNetwork longTermNetwork;

    private ConcurrentQueue<(int BarIndex, double[] Input, double ActualPrice, double Weight)> shortTermBuffer = new ConcurrentQueue<(int, double[], double, double)>();
    private ConcurrentQueue<(int BarIndex, double[] Input, double ActualPrice, double Weight)> mediumTermBuffer = new ConcurrentQueue<(int, double[], double, double)>();
    private ConcurrentQueue<(int BarIndex, double[] Input, double ActualPrice, double Weight)> longTermBuffer = new ConcurrentQueue<(int, double[], double, double)>();

    private ValueDataSeries shortTermAbovePrediction;
    private ValueDataSeries shortTermBelowPrediction;
    private ValueDataSeries mediumTermAbovePrediction;
    private ValueDataSeries mediumTermBelowPrediction;
    private ValueDataSeries longTermAbovePrediction;
    private ValueDataSeries longTermBelowPrediction;

    private double bestValidationLoss = double.MaxValue;
    private int epochsWithoutImprovement = 0;
    private int patience = 10;
    private bool isTrainingStopped = false;

    private List<(double[] Input, double ActualPrice)> validationData = new List<(double[], double)>();
    private int validationStartBar = -1;
    private int validationSize = 100;

    private int liveCandleCount = 0;

    private double[][][] trainedWeightsShortTerm;
    private double[][] trainedBiasesShortTerm;

    private double[][][] trainedWeightsMediumTerm;
    private double[][] trainedBiasesMediumTerm;

    private double[][][] trainedWeightsLongTerm;
    private double[][] trainedBiasesLongTerm;

    private double lastVWAPValue = 0;

    private int lastProcessedCandle = -1; // Initialize with -1 or any out-of-bounds value

    public SingularityHybridNN() : base(true)
    {
        InitializeDataSeries();
        InitializeEarlyStopping();
    }

    private void InitializeEarlyStopping()
    {
        bestValidationLoss = double.MaxValue;
        epochsWithoutImprovement = 0;
        patience = 10;
    }
    private void InitializeNetworks(int bar)
    {
        double currentPrice = (double)GetCandle(bar).Close;

        shortTermNetwork = new NeuralNetwork(inputSize, hiddenSizes, outputSize, new Random(rand.Next()), currentPrice);
        mediumTermNetwork = new NeuralNetwork(inputSize, hiddenSizes, outputSize, new Random(rand.Next()), currentPrice);
        longTermNetwork = new NeuralNetwork(inputSize, hiddenSizes, outputSize, new Random(rand.Next()), currentPrice);

        double decayRate = 0.0005;
        adaptiveLearningRate = initialLearningRate * Math.Exp(-decayRate * bar);
        adaptiveMutationRate = mutationRate;
    }

    private ValueDataSeries gradientShortTerm;
    private ValueDataSeries gradientMediumTerm;
    private ValueDataSeries gradientLongTerm;

    private void InitializeDataSeries()
    {
        shortTermPrediction = new ValueDataSeries("5 Candles Ahead Prediction");
        mediumTermPrediction = new ValueDataSeries("15 Candles Ahead Prediction");
        longTermPrediction = new ValueDataSeries("30 Candles Ahead Prediction");

        gradientShortTerm = new ValueDataSeries("Short-Term Gradient");
        gradientMediumTerm = new ValueDataSeries("Medium-Term Gradient");
        gradientLongTerm = new ValueDataSeries("Long-Term Gradient");

        DataSeries.Add(shortTermPrediction);
        DataSeries.Add(mediumTermPrediction);
        DataSeries.Add(longTermPrediction);

        DataSeries.Add(gradientShortTerm);
        DataSeries.Add(gradientMediumTerm);
        DataSeries.Add(gradientLongTerm);
    }

    // Track last processed live bar and add clearing logic for live prediction updates
    private int lastProcessedLiveBar = -1;
    private bool isCandleClosed = false;

    // Updated OnCalculate method for live candle handling
    protected override void OnCalculate(int bar, decimal value)
    {
        // Use 'bar' as the current candle or bar index
        int currentCandle = bar;

        // Detect a new candle by comparing with the last processed candle
        if (currentCandle != lastProcessedCandle)
        {
            // Clear buffers to prevent stacking from the previous candle
            shortTermBuffer.Clear();
            mediumTermBuffer.Clear();
            longTermBuffer.Clear();

            // Update the last processed candle to the current candle
            lastProcessedCandle = currentCandle;
        }

        int BarsCount = ChartInfo.PriceChartContainer.TotalBars;
        int lookback = 50;

        double[] input = null;
        decimal currentPrice = 0;
        double predictedShortTermPrice = 0;
        double predictedMediumTermPrice = 0;
        double predictedLongTermPrice = 0;

        // Initialize networks at lookback
        if (bar == lookback)
        {
            InitializeNetworks(bar);
        }

        if (bar >= lookback)
        {
            input = ExtractFeatures(bar, lookback);  // Extract features
            currentPrice = GetCandle(bar).Close;

            // VWAP-based bias correction
            double VWAPValue = CalculateVWAP(bar);
            double biasCorrectionFactor = DetectTrendReversal(VWAPValue, (double)currentPrice, bar);

            // Apply VWAP-based bias correction
            predictedShortTermPrice = shortTermNetwork.Forward(input)[0] + biasCorrectionFactor;
            predictedMediumTermPrice = mediumTermNetwork.Forward(input)[0] + biasCorrectionFactor;
            predictedLongTermPrice = longTermNetwork.Forward(input)[0] + biasCorrectionFactor;

            // Store the predictions
            shortTermPrediction[bar] = (decimal)predictedShortTermPrice;
            mediumTermPrediction[bar] = (decimal)predictedMediumTermPrice;
            longTermPrediction[bar] = (decimal)predictedLongTermPrice;

            // Adjust colors based on predictions
            shortTermPrediction.Colors[bar] = AssignPredictionColor(predictedShortTermPrice, (double)currentPrice);
            mediumTermPrediction.Colors[bar] = AssignPredictionColor(predictedMediumTermPrice, (double)currentPrice);
            longTermPrediction.Colors[bar] = AssignPredictionColor(predictedLongTermPrice, (double)currentPrice);

            // Handle live candle updates
            if (bar == BarsCount - 1) // Live candle
            {
                // Reset weights and biases before processing live data
                ResetWeightsAndBiases(shortTermNetwork, trainedWeightsShortTerm, trainedBiasesShortTerm);
                ResetWeightsAndBiases(mediumTermNetwork, trainedWeightsMediumTerm, trainedBiasesMediumTerm);
                ResetWeightsAndBiases(longTermNetwork, trainedWeightsLongTerm, trainedBiasesLongTerm);

                // Apply live candle update
                ReverseLearn(bar, currentPrice);

                // Ensure only one live update occurs per tick
                isCandleClosed = false;  // Keep this flag false during live updates
                lastProcessedLiveBar = bar;  // Track the latest live bar
            }
            else if (bar < BarsCount - 1 && bar == lastProcessedLiveBar)
            {
                // If the bar closes, finalize the prediction with the complete data
                isCandleClosed = true;  // Mark the candle as closed
                lastProcessedLiveBar = -1;  // Reset live bar tracking to avoid stacking
            }


            // Handle buffer enqueue for all candles
            if (lastEnqueuedBarShortTerm < bar)
            {
                shortTermBuffer.Enqueue((bar, input, double.NaN, 0.5));
                shortTermBuffer.Enqueue((bar, input, double.NaN, 0.5));
                lastEnqueuedBarShortTerm = bar;
            }

            if (lastEnqueuedBarMediumTerm < bar)
            {
                mediumTermBuffer.Enqueue((bar, input, double.NaN, 0.5));
                mediumTermBuffer.Enqueue((bar, input, double.NaN, 0.5));
                lastEnqueuedBarMediumTerm = bar;
            }

            if (lastEnqueuedBarLongTerm < bar)
            {
                longTermBuffer.Enqueue((bar, input, double.NaN, 0.5));
                longTermBuffer.Enqueue((bar, input, double.NaN, 0.5));
                lastEnqueuedBarLongTerm = bar;
            }

            if (bar <= BarsCount - 1)
            {
                if (lastProcessedBarBuffers < bar)
                {
                    ProcessBuffer(shortTermBuffer, shortTermHorizon, shortTermNetwork);
                    ProcessBuffer(mediumTermBuffer, mediumTermHorizon, mediumTermNetwork);
                    ProcessBuffer(longTermBuffer, longTermHorizon, longTermNetwork);
                    lastProcessedBarBuffers = bar;
                }

                gradientShortTerm[bar] = (decimal)shortTermNetwork.LastGradient;
                gradientMediumTerm[bar] = (decimal)mediumTermNetwork.LastGradient;
                gradientLongTerm[bar] = (decimal)longTermNetwork.LastGradient;
            }

            // Handle network evolution after buffer processing
            if ((shortTermBuffer.Count > 0 || mediumTermBuffer.Count > 0 || longTermBuffer.Count > 0) && !isTrainingStopped)
            {
                EvolveNetworks(bar);
            }

            if (bar == BarsCount - 2) // Last historical candle
            {
                ApplyPredictionColorsAndTransparency(bar, predictedShortTermPrice, predictedMediumTermPrice, predictedLongTermPrice, currentPrice, false);
            }
            else if (bar == BarsCount - 1) // Live candle
            {
                ApplyPredictionColorsAndTransparency(bar, predictedShortTermPrice, predictedMediumTermPrice, predictedLongTermPrice, currentPrice, true);
            }
            else
            {
                ApplyPredictionColorsAndTransparency(bar, predictedShortTermPrice, predictedMediumTermPrice, predictedLongTermPrice, currentPrice, false);
            }

            double VWAPThreshold = CalculateAdaptiveVWAPThreshold(bar);
        }
    }

    private void ApplyPredictionColorsAndTransparency(int bar, double predictedShortTermPrice, double predictedMediumTermPrice, double predictedLongTermPrice, decimal currentPrice, bool isLive)
    {
        int opacityShortTerm = 255;  // Full opacity for short-term
        int opacityMediumTerm = 150;  // Medium opacity for medium-term
        int opacityLongTerm = 80;     // Lowest opacity for long-term

        if (!isLive)
        {
            // Apply dynamic transparency based on the network length
            opacityShortTerm = (int)(255 * (shortTermHorizon / (double)numberOfNetworks));
            opacityMediumTerm = (int)(150 * (mediumTermHorizon / (double)numberOfNetworks));
            opacityLongTerm = (int)(80 * (longTermHorizon / (double)numberOfNetworks));
        }

        // Short-term predictions
        if (predictedShortTermPrice > (double)currentPrice)
            shortTermPrediction.Colors[bar] = System.Drawing.Color.FromArgb(opacityShortTerm, 0, 255, 255);  // Cyan for higher predictions
        else
            shortTermPrediction.Colors[bar] = System.Drawing.Color.FromArgb(opacityShortTerm, 233, 30, 99);  // Red-pink for lower predictions
        shortTermPrediction.Width = 3;

        // Medium-term predictions
        if (predictedMediumTermPrice > (double)currentPrice)
            mediumTermPrediction.Colors[bar] = System.Drawing.Color.FromArgb(opacityMediumTerm, 0, 255, 255);
        else
            mediumTermPrediction.Colors[bar] = System.Drawing.Color.FromArgb(opacityMediumTerm, 233, 30, 99);
        mediumTermPrediction.Width = 2;

        // Long-term predictions
        if (predictedLongTermPrice > (double)currentPrice)
            longTermPrediction.Colors[bar] = System.Drawing.Color.FromArgb(opacityLongTerm, 0, 255, 255);
        else
            longTermPrediction.Colors[bar] = System.Drawing.Color.FromArgb(opacityLongTerm, 233, 30, 99);
        longTermPrediction.Width = 1;
    }

    private double CalculateAdaptiveVWAPThreshold(int bar)
    {
        double baseThreshold = 0.05; // Base threshold, similar to the fixed value you're using
        double volatility = CalculateRecentVolatility(bar); // Use the existing volatility calculation method

        // Adjust threshold dynamically based on volatility (scaling factor of 0.01 is arbitrary, you can tweak it)
        return baseThreshold + (volatility * 0.01);
    }

    // Adapted method to dynamically adjust the VWAP lookback period
    private int GetAdaptiveLookback(int bar)
    {
        int minLookback = 5;
        int maxLookback = 50;

        double volatility = CalculateRecentVolatility(bar);

        // Higher volatility reduces the lookback period, lower volatility increases it
        int adaptiveLookback = (int)(maxLookback - (volatility * (maxLookback - minLookback)));

        // Ensure the lookback remains within the defined range
        return Math.Max(minLookback, Math.Min(maxLookback, adaptiveLookback));
    }

    private double CalculateRecentVolatility(int bar)
    {
        int volatilityLookback = 10; // Use a fixed short period to calculate volatility
        double totalVolatility = 0;

        for (int i = bar - volatilityLookback; i <= bar; i++)
        {
            var candle = GetCandle(i);
            totalVolatility += (double)(candle.High - candle.Low);
        }

        // Average volatility over the lookback period
        return totalVolatility / volatilityLookback;
    }

    // Updated VWAP calculation using adaptive lookback
    private double CalculateVWAP(int bar)
    {
        int adaptiveLookback = GetAdaptiveLookback(bar); // Dynamically adjust the lookback period
        double cumulativePriceVolume = 0;
        double cumulativeVolume = 0;

        for (int i = bar - adaptiveLookback; i <= bar; i++)
        {
            var candle = GetCandle(i);
            double typicalPrice = (double)(candle.High + candle.Low + candle.Close) / 3;
            cumulativePriceVolume += typicalPrice * (double)candle.Volume;
            cumulativeVolume += (double)candle.Volume;
        }
        return cumulativePriceVolume / cumulativeVolume;
    }
    private double DetectTrendReversal(double VWAPValue, double currentPrice, int bar)
    {
        double VWAPThreshold = CalculateAdaptiveVWAPThreshold(bar); // Use the adaptive threshold
        double biasCorrection = 0;

        if (Math.Abs(currentPrice - VWAPValue) > VWAPThreshold)
        {
            biasCorrection = currentPrice > VWAPValue ? -0.1 : 0.1; // Reverse bias correction
        }

        lastVWAPValue = VWAPValue;
        return biasCorrection;
    }

    // Centralized function for assigning colors based on prediction and current price
    private System.Drawing.Color AssignPredictionColor(double predictedPrice, double currentPrice)
    {
        // Blueish color if prediction is higher than the current price
        if (predictedPrice > currentPrice)
            return System.Drawing.Color.FromArgb(255, 0, 255, 255); // High-opacity cyan

        // Reddish color if prediction is lower than the current price
        return System.Drawing.Color.FromArgb(255, 233, 30, 99); // High-opacity red-pink
    }

    // Apply gradient clipping during backpropagation (both for historical and live updates)
    private double ClipGradient(double gradient, double clipValue = 0.5)
    {
        return Math.Max(Math.Min(gradient, clipValue), -clipValue);
    }

    // Reset weights and biases before each live update
    private void ReverseLearn(int liveBar, decimal currentPrice)
    {
        // Reset short-term, medium-term, and long-term weights and biases before updating
        ResetWeightsAndBiases(shortTermNetwork, trainedWeightsShortTerm, trainedBiasesShortTerm);
        ResetWeightsAndBiases(mediumTermNetwork, trainedWeightsMediumTerm, trainedBiasesMediumTerm);
        ResetWeightsAndBiases(longTermNetwork, trainedWeightsLongTerm, trainedBiasesLongTerm);

        // Apply reverse learning to adjust networks with live candle data
        if (liveBar >= 0)
        {
            double[] shortTermInput = ExtractFeatures(liveBar - shortTermHorizon, shortTermHorizon);
            shortTermNetwork.BackPropagate(shortTermInput, new double[] { (double)currentPrice }, adaptiveLearningRate, l2Regularization);

            double[] mediumTermInput = ExtractFeatures(liveBar - mediumTermHorizon, mediumTermHorizon);
            mediumTermNetwork.BackPropagate(mediumTermInput, new double[] { (double)currentPrice }, adaptiveLearningRate, l2Regularization);

            double[] longTermInput = ExtractFeatures(liveBar - longTermHorizon, longTermHorizon);
            longTermNetwork.BackPropagate(longTermInput, new double[] { (double)currentPrice }, adaptiveLearningRate, l2Regularization);
        }
    }

    // Reset the weights and biases to their original values
    private void ResetWeightsAndBiases(NeuralNetwork network, double[][][] originalWeights, double[][] originalBiases)
    {
        for (int layer = 0; layer < network.weights.Length; layer++)
        {
            for (int i = 0; i < network.weights[layer].Length; i++)
            {
                for (int j = 0; j < network.weights[layer][i].Length; j++)
                {
                    network.weights[layer][i][j] = originalWeights[layer][i][j];
                }
            }
        }

        for (int layer = 0; layer < network.biases.Length; layer++)
        {
            for (int i = 0; i < network.biases[layer].Length; i++)
            {
                network.biases[layer][i] = originalBiases[layer][i];
            }
        }
    }
    private double[][][] CloneWeights(double[][][] weights)
    {
        double[][][] clonedWeights = new double[weights.Length][][];
        for (int layer = 0; layer < weights.Length; layer++)
        {
            clonedWeights[layer] = new double[weights[layer].Length][];
            for (int i = 0; i < weights[layer].Length; i++)
            {
                clonedWeights[layer][i] = (double[])weights[layer][i].Clone();
            }
        }
        return clonedWeights;
    }

    private double[][] CloneBiases(double[][] biases)
    {
        double[][] clonedBiases = new double[biases.Length][];
        for (int layer = 0; layer < biases.Length; layer++)
        {
            clonedBiases[layer] = (double[])biases[layer].Clone();
        }
        return clonedBiases;
    }

    bool IsValidLiveCandle(int bar)
    {
        var liveCandle = GetCandle(bar);
        return liveCandle != null && liveCandle.Volume > 0;
    }

    void ProcessBuffer(ConcurrentQueue<(int BarIndex, double[] Input, double ActualPrice, double Weight)> buffer, int horizon, NeuralNetwork network)
    {
        while (buffer.TryPeek(out var prediction))
        {
            int targetBar = prediction.BarIndex + horizon;

            // Ensure that we are not updating weights for candles without future data (invalid predictions)
            if (targetBar < ChartInfo.PriceChartContainer.TotalBars - 1)
            {
                var futureCandle = GetCandle(targetBar);
                double actualPrice = (double)futureCandle.Close;

                // Only update if we have valid future data (actualPrice is not NaN)
                if (!double.IsNaN(actualPrice))
                {
                    network.BackPropagate(prediction.Input, new double[] { actualPrice }, adaptiveLearningRate * prediction.Weight, l2Regularization);
                    buffer.TryDequeue(out _);  // Remove the processed prediction from the buffer
                }
                else
                {
                    // Skip any further processing for invalid predictions
                    buffer.TryDequeue(out _);
                }
            }
            else
            {
                // Skip any predictions where future data is unavailable
                break;
            }
        }
    }
    double[] ExtractFeatures(int bar, int lookback)
    {
        double[] features = new double[inputSize];

        double totalVolume = 0;
        double totalDelta = 0;
        double totalMaxVolumePriceInfoVolume = 0;
        double buyOrders = 0;
        double sellOrders = 0;

        double maxOI = double.MinValue;
        double minOI = double.MaxValue;
        double maxDelta = double.MinValue;
        double minDelta = double.MaxValue;

        double totalVolatility = 0;

        for (int j = bar - lookback + 1; j <= bar; j++)
        {
            if (j < 0) continue;

            var candle_j = GetCandle(j);

            totalVolume += (double)candle_j.Volume;
            totalDelta += (double)candle_j.Delta;
            totalMaxVolumePriceInfoVolume += (double)candle_j.MaxVolumePriceInfo.Volume;

            buyOrders += (double)(MarketDepthInfo?.CumulativeDomAsks ?? 0);
            sellOrders += (double)(MarketDepthInfo?.CumulativeDomBids ?? 0);

            if ((double)candle_j.MaxOI > maxOI) maxOI = (double)candle_j.MaxOI;
            if ((double)candle_j.MinOI < minOI) minOI = (double)candle_j.MinOI;

            if ((double)candle_j.MaxDelta > maxDelta) maxDelta = (double)candle_j.MaxDelta;
            if ((double)candle_j.MinDelta < minDelta) minDelta = (double)candle_j.MinDelta;

            totalVolatility += (double)(candle_j.High - candle_j.Low);
        }

        var currentCandle = GetCandle(bar);
        var previousCandle = GetCandle(bar - 1);

        features[0] = ScaleToRange(totalVolume / lookback, 0, 1e6, -1, 1);
        features[1] = ScaleToRange(maxOI, 0, 1e5, -1, 1);
        features[2] = ScaleToRange(maxOI - minOI, 0, 1e5, -1, 1);
        features[3] = ScaleToRange(totalDelta / lookback, -1e4, 1e4, -1, 1);
        features[4] = ScaleToRange(maxDelta - minDelta, -1e4, 1e4, -1, 1);
        features[5] = ScaleToRange(totalMaxVolumePriceInfoVolume / lookback, 0, 1e6, -1, 1);
        features[6] = ScaleToRange(Math.Log((double)currentCandle.Close / (double)previousCandle.Close), -0.1, 0.1, -1, 1);
        features[7] = ScaleToRange((double)(currentCandle.MaxVolumePriceInfo?.Volume ?? 0), 0, 1e6, -1, 1);
        features[8] = ScaleToRange((double)currentCandle.Delta, -1e4, 1e4, -1, 1);
        features[9] = ScaleToRange((buyOrders - sellOrders) / Math.Max(buyOrders + sellOrders, 1e-8), -1, 1, -1, 1);
        features[10] = ScaleToRange(buyOrders, 0, 1e6, -1, 1);
        features[11] = ScaleToRange(sellOrders, 0, 1e6, -1, 1);
        features[12] = ScaleToRange(totalVolatility / lookback, 0, 100, -1, 1);
        features[13] = ScaleToRange((double)currentCandle.High, 0, 1e6, -1, 1);
        features[14] = ScaleToRange((double)currentCandle.Low, 0, 1e6, -1, 1);
        features[15] = ScaleToRange((double)currentCandle.Open, 0, 1e6, -1, 1);
        features[16] = ScaleToRange((double)currentCandle.Close, 0, 1e6, -1, 1);

        return features;
    }
    double ScaleToRange(double value, double min, double max, double newMin, double newMax)
    {
        return ((value - min) / (max - min)) * (newMax - newMin) + newMin;
    }

    double GetFuturePrice(int bar, int horizon)
    {
        if (bar + horizon < ChartInfo.PriceChartContainer.TotalBars)
        {
            var futureCandle = GetCandle(bar + horizon);
            return (double)futureCandle.Close;
        }
        else
        {
            return double.NaN;
        }
    }
    void EvolveNetworks(int bar)
    {
        if (isTrainingStopped)
            return;

        adaptiveMutationRate = AdjustMutationRateBasedOnPerformance();

        // Adjust mutation rate if close to the end of available bars
        if (ChartInfo.PriceChartContainer.TotalBars - lastProcessedBarBuffers <= longTermHorizon)
        {
            adaptiveMutationRate *= 1.5;
        }

        // Apply mutations to all networks
        ApplyMutation(shortTermNetwork.weights, adaptiveMutationRate);
        ApplyMutation(mediumTermNetwork.weights, adaptiveMutationRate);
        ApplyMutation(longTermNetwork.weights, adaptiveMutationRate);

        // Validation logic, assuming the data is valid and not pseudo-learned
        if (validationData.Count >= validationSize)
        {
            double currentValidationLoss = EvaluateValidationSet();
            Console.WriteLine($"Current Validation Loss: {currentValidationLoss}");

            // Update the best validation loss and manage early stopping
            if (currentValidationLoss < bestValidationLoss)
            {
                bestValidationLoss = currentValidationLoss;
                epochsWithoutImprovement = 0;
            }
            else
            {
                epochsWithoutImprovement++;
                if (epochsWithoutImprovement >= patience)
                {
                    StopTraining();
                }
            }
        }
    }

    double EvaluateValidationSet()
    {
        if (validationData.Count == 0)
            return double.MaxValue;

        double mseShort = 0;
        double mseMedium = 0;
        double mseLong = 0;
        int count = validationData.Count;

        foreach (var data in validationData)
        {
            double[] input = data.Input;
            double actual = data.ActualPrice;

            double predictedShort = shortTermNetwork.Forward(input)[0];
            double predictedMedium = mediumTermNetwork.Forward(input)[0];
            double predictedLong = longTermNetwork.Forward(input)[0];

            mseShort += Math.Pow(predictedShort - actual, 2);
            mseMedium += Math.Pow(predictedMedium - actual, 2);
            mseLong += Math.Pow(predictedLong - actual, 2);
        }

        double averageMSE = (mseShort + mseMedium + mseLong) / (3 * count);
        return averageMSE;
    }
    void StopTraining()
    {
        isTrainingStopped = true;
        Console.WriteLine("Early stopping has been activated. No further training will occur.");
    }
    double AdjustMutationRateBasedOnPerformance()
    {
        double averagePerformance = (shortTermNetwork.Performance + mediumTermNetwork.Performance + longTermNetwork.Performance) / 3.0;

        if (averagePerformance > proliferationThreshold)
        {
            double newRate = mutationRate * 0.5;
            return Math.Max(newRate, 0.001);
        }
        else if (averagePerformance < -proliferationThreshold)
        {
            double newRate = mutationRate * 2;
            return Math.Min(newRate, 0.1);
        }

        return mutationRate;
    }
    void ApplyMutation(double[][][] weights, double currentMutationRate)
    {
        for (int layer = 0; layer < weights.Length; layer++)
        {
            for (int i = 0; i < weights[layer].Length; i++)
            {
                for (int j = 0; j < weights[layer][i].Length; j++)
                {
                    if (!double.IsNaN(weights[layer][i][j]))
                    {
                        if (rand.NextDouble() < currentMutationRate)
                        {
                            weights[layer][i][j] += (rand.NextDouble() - 0.5) * 0.2;
                        }
                    }
                }
            }
        }
    }
    public class NeuralNetwork
    {
        private int inputSize;
        private int[] hiddenSizes;
        private int outputSize;
        public double[][][] weights { get; private set; }
        public double[][][] InitialWeights { get; private set; }  // Store initial weights
        public double[][] biases { get; private set; }            // Make biases public
        public double[][] InitialBiases { get; private set; }     // Store initial biases
        private Random rand;
        public double Performance { get; private set; }
        public double LastGradient { get; set; }  // Allow public access to set the gradient
        public double[] LastInput { get; private set; }        // Add this field to store the last input
        public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, Random rand, double currentPrice)
        {
            this.inputSize = inputSize;
            this.hiddenSizes = hiddenSizes;
            this.outputSize = outputSize;
            this.rand = rand;

            weights = InitializeWeights(inputSize, hiddenSizes, outputSize);
            biases = InitializeBiases(hiddenSizes, outputSize, currentPrice);

            // Store the initial weights and biases
            InitialWeights = CloneWeights(weights);
            InitialBiases = CloneBiases(biases);
        }

        // Helper function to deep copy the weights
        private double[][][] CloneWeights(double[][][] weights)
        {
            double[][][] clonedWeights = new double[weights.Length][][];
            for (int layer = 0; layer < weights.Length; layer++)
            {
                clonedWeights[layer] = new double[weights[layer].Length][];
                for (int i = 0; i < weights[layer].Length; i++)
                {
                    clonedWeights[layer][i] = (double[])weights[layer][i].Clone();
                }
            }
            return clonedWeights;
        }

        // Helper function to deep copy the biases
        private double[][] CloneBiases(double[][] biases)
        {
            double[][] clonedBiases = new double[biases.Length][];
            for (int layer = 0; layer < biases.Length; layer++)
            {
                clonedBiases[layer] = (double[])biases[layer].Clone();
            }
            return clonedBiases;
        }
        private double[][][] InitializeWeights(int inputSize, int[] hiddenSizes, int outputSize)
        {
            double[][][] weights = new double[hiddenSizes.Length + 1][][];

            weights[0] = new double[inputSize][];
            double heFactorInput = Math.Sqrt(2.0 / inputSize);
            for (int i = 0; i < inputSize; i++)
            {
                weights[0][i] = new double[hiddenSizes[0]];
                for (int j = 0; j < hiddenSizes[0]; j++)
                {
                    weights[0][i][j] = (rand.NextDouble() - 0.5) * 2 * heFactorInput;
                }
            }

            for (int k = 1; k < hiddenSizes.Length; k++)
            {
                weights[k] = new double[hiddenSizes[k - 1]][];
                double heFactor = Math.Sqrt(2.0 / hiddenSizes[k - 1]);
                for (int i = 0; i < hiddenSizes[k - 1]; i++)
                {
                    weights[k][i] = new double[hiddenSizes[k]];
                    for (int j = 0; j < hiddenSizes[k]; j++)
                    {
                        weights[k][i][j] = (rand.NextDouble() - 0.5) * 2 * heFactor;
                    }
                }
            }

            weights[hiddenSizes.Length] = new double[hiddenSizes[hiddenSizes.Length - 1]][];
            double heFactorOutput = Math.Sqrt(2.0 / hiddenSizes[hiddenSizes.Length - 1]);
            for (int i = 0; i < hiddenSizes[hiddenSizes.Length - 1]; i++)
            {
                weights[hiddenSizes.Length][i] = new double[outputSize];
                for (int j = 0; j < outputSize; j++)
                {
                    weights[hiddenSizes.Length][i][j] = (rand.NextDouble() - 0.5) * 2 * heFactorOutput;
                }
            }

            return weights;
        }
        private double[][] InitializeBiases(int[] hiddenSizes, int outputSize, double currentPrice)
        {
            double[][] biases = new double[hiddenSizes.Length + 1][];

            for (int i = 0; i < hiddenSizes.Length; i++)
            {
                biases[i] = new double[hiddenSizes[i]];
                for (int j = 0; j < hiddenSizes[i]; j++)
                {
                    biases[i][j] = (rand.NextDouble() - 0.5) * 0.1;
                }
            }

            biases[hiddenSizes.Length] = new double[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                biases[hiddenSizes.Length][j] = currentPrice;
            }

            return biases;
        }

        public double[] Forward(double[] input)
        {
            // Store the input in LastInput
            LastInput = input;

            double[] hiddenLayer = ActivationFunction(MatrixMultiply(input, weights[0], biases[0]));
            for (int k = 1; k < hiddenSizes.Length; k++)
            {
                hiddenLayer = ActivationFunction(MatrixMultiply(hiddenLayer, weights[k], biases[k]));
            }

            return MatrixMultiply(hiddenLayer, weights[hiddenSizes.Length], biases[hiddenSizes.Length]);
        }

        private double[] ActivationFunction(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] > 0 ? input[i] : 0.01 * input[i];
            }
            return output;
        }
        private double[] MatrixMultiply(double[] input, double[][] weights, double[] biases)
        {
            double[] output = new double[weights[0].Length];
            for (int j = 0; j < weights[0].Length; j++)
            {
                output[j] = biases[j];
                for (int i = 0; i < input.Length; i++)
                {
                    if (!double.IsNaN(input[i]))
                    {
                        output[j] += input[i] * weights[i][j];
                    }
                }
            }
            return output;
        }
        public void BackPropagate(double[] input, double[] target, double learningRate, double l2Regularization, bool isPseudoTarget = false)
        {
            double effectiveLearningRate = isPseudoTarget ? learningRate * 0.5 : learningRate;  // Slightly reduce learning rate for pseudo-updates

            double[] hidden1 = ActivationFunction(MatrixMultiply(input, weights[0], biases[0]));
            double[] hidden2 = ActivationFunction(MatrixMultiply(hidden1, weights[1], biases[1]));
            double[] hidden3 = ActivationFunction(MatrixMultiply(hidden2, weights[2], biases[2]));
            double[] output = MatrixMultiply(hidden3, weights[3], biases[3]);

            double[] outputError = new double[target.Length];
            for (int i = 0; i < target.Length; i++)
            {
                outputError[i] = output[i] - target[i];
            }

            double[][] weightsHidden3OutputT = Transpose(weights[3]);
            double[] hidden3Error = MatrixMultiply(outputError, weightsHidden3OutputT, new double[hiddenSizes[2]]);
            double[] hidden3Delta = ComputeLayerDelta(hidden3, hidden3Error);

            double[][] weightsHidden2Hidden3T = Transpose(weights[2]);
            double[] hidden2Error = MatrixMultiply(hidden3Delta, weightsHidden2Hidden3T, new double[hiddenSizes[1]]);
            double[] hidden2Delta = ComputeLayerDelta(hidden2, hidden2Error);

            double[][] weightsHidden1Hidden2T = Transpose(weights[1]);
            double[] hidden1Error = MatrixMultiply(hidden2Delta, weightsHidden1Hidden2T, new double[hiddenSizes[0]]);
            double[] hidden1Delta = ComputeLayerDelta(hidden1, hidden1Error);

            double gradientSum = 0;
            int gradientCount = 0;

            UpdateWeightsAndBiases(input, hidden1, hidden2, hidden3, hidden1Delta, hidden2Delta, hidden3Delta, outputError, effectiveLearningRate, ref gradientSum, ref gradientCount);

            double mse = 0;
            for (int i = 0; i < target.Length; i++)
            {
                mse += Math.Pow(Forward(input)[i] - target[i], 2);
            }
            Performance = mse / target.Length;

            Console.WriteLine($"Performance (MSE): {Performance}");
        }

        private double[] ComputeLayerDelta(double[] layerOutput, double[] layerError)
        {
            double[] delta = new double[layerOutput.Length];
            for (int i = 0; i < layerOutput.Length; i++)
            {
                delta[i] = layerError[i] * (layerOutput[i] > 0 ? 1 : 0.01);
            }
            return delta;
        }

        private void UpdateWeightsAndBiases(double[] input, double[] hidden1, double[] hidden2, double[] hidden3, double[] hidden1Delta, double[] hidden2Delta, double[] hidden3Delta, double[] outputDelta, double learningRate, ref double gradientSum, ref int gradientCount)
        {
            double clipValue = 1.0;

            for (int i = 0; i < input.Length; i++)
            {
                for (int j = 0; j < hiddenSizes[0]; j++)
                {
                    double gradient = hidden1Delta[j] * input[i];
                    gradient = ClipGradient(gradient, clipValue);
                    weights[0][i][j] -= learningRate * gradient;

                    gradientSum += Math.Abs(gradient);
                    gradientCount++;
                }
            }

            for (int j = 0; j < hiddenSizes[0]; j++)
            {
                double biasGradient = hidden1Delta[j];
                biases[0][j] -= learningRate * biasGradient;

                gradientSum += Math.Abs(biasGradient);
                gradientCount++;
            }

            for (int i = 0; i < hidden1.Length; i++)
            {
                for (int j = 0; j < hiddenSizes[1]; j++)
                {
                    double gradient = hidden2Delta[j] * hidden1[i];
                    gradient = ClipGradient(gradient, clipValue);
                    weights[1][i][j] -= learningRate * gradient;

                    gradientSum += Math.Abs(gradient);
                    gradientCount++;
                }
            }

            for (int j = 0; j < hiddenSizes[1]; j++)
            {
                biases[1][j] -= learningRate * hidden2Delta[j];
            }

            for (int i = 0; i < hidden2.Length; i++)
            {
                for (int j = 0; j < hiddenSizes[2]; j++)
                {
                    double gradient = hidden3Delta[j] * hidden2[i];
                    gradient = ClipGradient(gradient, clipValue);
                    weights[2][i][j] -= learningRate * gradient;
                }
            }

            for (int j = 0; j < hiddenSizes[2]; j++)
            {
                biases[2][j] -= learningRate * hidden3Delta[j];
            }

            for (int i = 0; i < hidden3.Length; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    double gradient = outputDelta[j] * hidden3[i];
                    gradient = ClipGradient(gradient, clipValue);
                    weights[3][i][j] -= learningRate * gradient;
                }
            }

            for (int j = 0; j < outputSize; j++)
            {
                biases[3][j] -= learningRate * outputDelta[j];
            }
        }
        private double ClipGradient(double gradient, double clipValue = 0.5)
        {
            return Math.Max(Math.Min(gradient, clipValue), -clipValue);
        }

        private double[][] Transpose(double[][] matrix)
        {
            double[][] transposed = new double[matrix[0].Length][];
            for (int i = 0; i < matrix[0].Length; i++)
            {
                transposed[i] = new double[matrix.Length];
                for (int j = 0; j < matrix.Length; j++)
                {
                    transposed[i][j] = matrix[j][i];
                }
            }
            return transposed;
        }
    }
}