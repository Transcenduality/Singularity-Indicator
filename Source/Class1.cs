using System;
using System.Windows.Media;
using ATAS.Indicators;
using System.Drawing;

public class PredictiveNeuralNetwork : Indicator
{
    private int inputSize = 6; // Updated from 7 to 6 features per timeframe
    private int[] hiddenSizes = { 128, 256, 512 };
    private int outputSize = 1; // One output per prediction timeframe

    private double[][][] weightsInputHidden1;
    private double[][][] weightsHidden1Hidden2;
    private double[][][] weightsHidden2Hidden3;
    private double[][][] weightsHidden3Output;

    private double[][] biasesHidden1;
    private double[][] biasesHidden2;
    private double[][] biasesHidden3;
    private double[][] biasesOutput;

    private double learningRate = 0.005;
    private double l2Regularization = 0.001;

    private ValueDataSeries[] predictiveSeries;
    private System.Windows.Media.Color[][] lineColors;
    private int[] predictionTimeframes = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };

    private Random rand = new Random();

    private double[][] inputMA;
    private double[][] inputStd;
    private int maPeriod = 100;

    private double[] minErrors;

    private int lastProcessedBar = -1;

    private ValueDataSeries[] errorSeries;

    private double[][][] optimalWeightsInputHidden1;
    private double[][][] optimalWeightsHidden1Hidden2;
    private double[][][] optimalWeightsHidden2Hidden3;
    private double[][][] optimalWeightsHidden3Output;

    private double[][] optimalBiasesHidden1;
    private double[][] optimalBiasesHidden2;
    private double[][] optimalBiasesHidden3;
    private double[][] optimalBiasesOutput;

    private bool[] isOptimalSetSaved;

    private double[] previousPredictionDifferences;
    private double consistencyPenaltyWeight = 2;
    private double minPenalty = 1;
    private double maxPenalty = 4;
    private double penaltyAdjustmentRate = 0.1; // Rate of penalty adjustment based on performance

    // Define the specific short-term timeframes
    private int[] shortTermTimeframes = { 1, 2, 4, 8, 16 };

    private System.Drawing.Color ConvertToDrawingColor(System.Windows.Media.Color mediaColor)
    {
        return System.Drawing.Color.FromArgb(mediaColor.A, mediaColor.R, mediaColor.G, mediaColor.B);
    }

    private bool DidCrossPrice(double previousClosePrice, double currentClosePrice, double predictedPrice)
    {
        // Check if the predicted price crosses over the previous close price
        return (predictedPrice < previousClosePrice && predictedPrice > currentClosePrice) ||
               (predictedPrice > previousClosePrice && predictedPrice < currentClosePrice);
    }

    public PredictiveNeuralNetwork() : base(true)
    {
        InitializeWeightsAndBiases();
        InitializeOptimalWeightsAndBiases();
        InitializeDataSeries();
        InitializeErrorSeries();
        InitializeInputNormalization();
        InitializeLineColors();
        InitializeMinErrors();
        InitializeOptimalSetFlags();
        InitializePreviousPredictionDifferences(); // Initialize previous differences
    }

    private void InitializePreviousPredictionDifferences()
    {
        previousPredictionDifferences = new double[predictionTimeframes.Length];
        for (int i = 0; i < predictionTimeframes.Length; i++)
        {
            previousPredictionDifferences[i] = 0.0; // Initialize to zero or any suitable default
        }
    }

    private void InitializeOptimalWeightsAndBiases()
    {
        int timeframes = predictionTimeframes.Length;

        optimalWeightsInputHidden1 = new double[timeframes][][];
        optimalWeightsHidden1Hidden2 = new double[timeframes][][];
        optimalWeightsHidden2Hidden3 = new double[timeframes][][];
        optimalWeightsHidden3Output = new double[timeframes][][];

        optimalBiasesHidden1 = new double[timeframes][];
        optimalBiasesHidden2 = new double[timeframes][];
        optimalBiasesHidden3 = new double[timeframes][];
        optimalBiasesOutput = new double[timeframes][]; // Corrected to 2D array

        for (int i = 0; i < timeframes; i++)
        {
            // Initialize weights without dividing inputSize by predictionTimeframes.Lengthonc
            optimalWeightsInputHidden1[i] = new double[inputSize][];
            for (int j = 0; j < inputSize; j++)
                optimalWeightsInputHidden1[i][j] = new double[hiddenSizes[0]];

            optimalWeightsHidden1Hidden2[i] = new double[hiddenSizes[0]][];
            for (int j = 0; j < hiddenSizes[0]; j++)
                optimalWeightsHidden1Hidden2[i][j] = new double[hiddenSizes[1]];

            optimalWeightsHidden2Hidden3[i] = new double[hiddenSizes[1]][];
            for (int j = 0; j < hiddenSizes[1]; j++)
                optimalWeightsHidden2Hidden3[i][j] = new double[hiddenSizes[2]];

            optimalWeightsHidden3Output[i] = new double[hiddenSizes[2]][];
            for (int j = 0; j < hiddenSizes[2]; j++)
                optimalWeightsHidden3Output[i][j] = new double[outputSize];

            // Initialize biases as 1D arrays
            optimalBiasesHidden1[i] = new double[hiddenSizes[0]];
            optimalBiasesHidden2[i] = new double[hiddenSizes[1]];
            optimalBiasesHidden3[i] = new double[hiddenSizes[2]];
            optimalBiasesOutput[i] = new double[outputSize]; // Corrected to 1D array
        }
    }

    private void InitializeOptimalSetFlags()
    {
        isOptimalSetSaved = new bool[predictionTimeframes.Length];
        for (int i = 0; i < isOptimalSetSaved.Length; i++)
        {
            isOptimalSetSaved[i] = false;
        }
    }

    private void InitializeErrorSeries()
    {
        errorSeries = new ValueDataSeries[predictionTimeframes.Length];
        for (int i = 0; i < predictionTimeframes.Length; i++)
        {
            errorSeries[i] = new ValueDataSeries($"Error {predictionTimeframes[i]}")
            {
                Color = Colors.Red,
                Width = 1,
                ScaleIt = true,
                VisualType = VisualMode.Line
            };
            DataSeries.Add(errorSeries[i]);
        }
    }

    private void InitializeMinErrors()
    {
        minErrors = new double[predictionTimeframes.Length];
        for (int i = 0; i < minErrors.Length; i++)
        {
            minErrors[i] = double.MaxValue;
        }
    }

    private void InitializeLineColors()
    {
        int maxBars = 10000;
        lineColors = new System.Windows.Media.Color[predictionTimeframes.Length][];
        for (int i = 0; i < predictionTimeframes.Length; i++)
        {
            lineColors[i] = new System.Windows.Media.Color[maxBars];
        }
    }

    private void InitializeWeightsAndBiases()
    {
        weightsInputHidden1 = new double[predictionTimeframes.Length][][];
        weightsHidden1Hidden2 = new double[predictionTimeframes.Length][][];
        weightsHidden2Hidden3 = new double[predictionTimeframes.Length][][];
        weightsHidden3Output = new double[predictionTimeframes.Length][][];

        biasesHidden1 = new double[predictionTimeframes.Length][];
        biasesHidden2 = new double[predictionTimeframes.Length][];
        biasesHidden3 = new double[predictionTimeframes.Length][];
        biasesOutput = new double[predictionTimeframes.Length][]; // Corrected to 2D array

        for (int i = 0; i < predictionTimeframes.Length; i++)
        {
            weightsInputHidden1[i] = InitializeWeights(inputSize, hiddenSizes[0]);

            weightsHidden1Hidden2[i] = InitializeWeights(hiddenSizes[0], hiddenSizes[1]);
            weightsHidden2Hidden3[i] = InitializeWeights(hiddenSizes[1], hiddenSizes[2]);
            weightsHidden3Output[i] = InitializeWeights(hiddenSizes[2], outputSize);

            biasesHidden1[i] = InitializeBiases(hiddenSizes[0]);
            biasesHidden2[i] = InitializeBiases(hiddenSizes[1]);
            biasesHidden3[i] = InitializeBiases(hiddenSizes[2]);
            biasesOutput[i] = InitializeBiases(outputSize); // Corrected to 1D array
        }
    }

    private void InitializeDataSeries()
    {
        predictiveSeries = new ValueDataSeries[predictionTimeframes.Length];

        for (int i = 0; i < predictionTimeframes.Length; i++)
        {
            predictiveSeries[i] = new ValueDataSeries($"Prediction {predictionTimeframes[i]}")
            {
                Width = 2,
                ScaleIt = false,
                VisualType = VisualMode.Line,
                Color = Colors.Blue
            };

            DataSeries.Add(predictiveSeries[i]);
        }
    }

    private void InitializeInputNormalization()
    {
        int timeframes = predictionTimeframes.Length;
        inputMA = new double[timeframes][];  // Ensure separate mean and std arrays for each timeframe
        inputStd = new double[timeframes][];

        for (int i = 0; i < timeframes; i++)
        {
            inputMA[i] = new double[6];  // Updated from 7 to 6
            inputStd[i] = new double[6];
            for (int j = 0; j < 6; j++) // Updated loop condition
            {
                inputMA[i][j] = 0;  // Initialize mean and std
                inputStd[i][j] = 1;
            }
        }
    }

    private double[][] InitializeWeights(int inputSizePerTimeframe, int outputSize)
    {
        double[][] weights = new double[inputSizePerTimeframe][];
        for (int i = 0; i < inputSizePerTimeframe; i++)
        {
            weights[i] = new double[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                weights[i][j] = (rand.NextDouble() - 0.5) * 2;
            }
        }
        return weights;
    }

    private double[] InitializeBiases(int size)
    {
        double[] biases = new double[size];
        for (int i = 0; i < size; i++)
        {
            biases[i] = (rand.NextDouble() - 0.5) * 2;
        }
        return biases;
    }

    private double[] NormalizeInput(double[] input, int timeframeIndex, int bar)
    {
        double[] normalizedInput = new double[6]; // Updated from 7 to 6
        for (int i = 0; i < 6; i++) // Updated loop condition
        {
            // Adjusted input normalization to emphasize volatility
            inputMA[timeframeIndex][i] = (inputMA[timeframeIndex][i] * (maPeriod - 1) + input[i]) / maPeriod;
            double volatilityFactor = CalculateVolatilityFactor(timeframeIndex, bar);
            double denom = (Array.Exists(shortTermTimeframes, element => element == predictionTimeframes[timeframeIndex]))
                ? Math.Max(inputStd[timeframeIndex][i], 1e-8)
                : Math.Max(inputStd[timeframeIndex][i] * volatilityFactor, 1e-8);  // Do not amplify std for short-term timeframes

            double normalizedValue = (input[i] - inputMA[timeframeIndex][i]) / denom;
            normalizedInput[i] = Math.Max(Math.Min(normalizedValue, 10), -10);

            // Function to calculate volatility factor
            double CalculateVolatilityFactor(int timeframeIndex, int bar)
            {
                double volatility = CalculateVolatility(timeframeIndex, bar);  // Use the volatility calculation method
                return Math.Max(1.0, volatility * 2);  // Increase normalization scaling during volatile periods
            }
        }
        return normalizedInput;
    }

    private double[] ForwardPropagateWithDynamicBiases(double[] input, int timeframeIndex, double[] dynamicBiasesHidden1, double[] dynamicBiasesHidden2, double[] dynamicBiasesHidden3)
    {
        // Apply dropout between layers to prevent overfitting
        double dropoutRate = 0.2;  // Example dropout rate
        double[] hidden1 = ApplyDropout(ActivationFunction(DotProduct(input, weightsInputHidden1[timeframeIndex], biasesHidden1[timeframeIndex])), dropoutRate);
        double[] hidden2 = ApplyDropout(ActivationFunction(DotProduct(hidden1, weightsHidden1Hidden2[timeframeIndex], biasesHidden2[timeframeIndex])), dropoutRate);
        double[] hidden3 = ApplyDropout(ActivationFunction(DotProduct(hidden2, weightsHidden2Hidden3[timeframeIndex], biasesHidden3[timeframeIndex])), dropoutRate);

        // Dropout function
        double[] ApplyDropout(double[] input, double dropoutRate)
        {
            Random rand = new Random();
            for (int i = 0; i < input.Length; i++)
            {
                if (rand.NextDouble() < dropoutRate)
                    input[i] = 0;  // Disable neurons based on dropout rate
            }
            return input;
        }

        double[] output = DotProduct(hidden3, weightsHidden3Output[timeframeIndex], biasesOutput[timeframeIndex]);

        return output;
    }

    private double[] DotProduct(double[] input, double[][] weights, double[] biases)
    {
        double[] output = new double[weights[0].Length];
        for (int j = 0; j < weights[0].Length; j++)
        {
            output[j] = biases[j];
            for (int i = 0; i < input.Length; i++)
            {
                output[j] += input[i] * weights[i][j];
            }
        }
        return output;
    }

    private double[] ActivationFunction(double[] input)
    {
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Math.Tanh(input[i]);
        }
        return output;
    }

    private bool isDataFromIncompleteCandle = false;

    protected override void OnCalculate(int bar, decimal value)
    {
        int BarsCount = ChartInfo.PriceChartContainer.TotalBars;
        bool isLiveCandle = bar == BarsCount - 1;

        if (isLiveCandle || bar > lastProcessedBar)
        {
            lastProcessedBar = bar;

            for (int i = 0; i < predictionTimeframes.Length; i++)
            {
                int lookback = predictionTimeframes[i];

                if (bar - lookback < 0)
                    continue;

                // Calculate volatility for the current timeframe and bar
                double volatility = CalculateVolatility(i, bar);  // i is the timeframeIndex here

                // Calculate features from candle data
                double[] features = ExtractCandleFeatures(bar, lookback);

                // Normalize input features
                double[] normalizedInput = NormalizeInput(features, i, bar);

                // Apply dynamic biases based on volatility for better live data handling
                double[] adjustedBiasesHidden1 = ApplyDynamicBiases(biasesHidden1[i], volatility, i);
                double[] adjustedBiasesHidden2 = ApplyDynamicBiases(biasesHidden2[i], volatility, i);
                double[] adjustedBiasesHidden3 = ApplyDynamicBiases(biasesHidden3[i], volatility, i);

                // Replace ForwardPropagate with ForwardPropagateWithDynamicBiases
                double[] output = ForwardPropagateWithDynamicBiases(normalizedInput, i, adjustedBiasesHidden1, adjustedBiasesHidden2, adjustedBiasesHidden3);

                double currentPrice = (double)GetCandle(bar).Close;
                double previousClosePrice = (double)GetCandle(bar - 1).Close; // Get the close price of the previous candle
                double predictedChange = output[0];
                double predictedPrice = currentPrice + predictedChange;

                predictiveSeries[i][bar] = (decimal)predictedPrice;

                // Update the color dynamically based on the prediction vs. current price
                byte alpha = (byte)(255 * (1.0 - i * 0.8 / (predictionTimeframes.Length - 1)));
                System.Windows.Media.Color color;

                if (predictedPrice > currentPrice)
                {
                    color = System.Windows.Media.Color.FromArgb(alpha, 0, 255, 255);  // Cyan for upward prediction
                }
                else
                {
                    color = System.Windows.Media.Color.FromArgb(alpha, 233, 30, 99);  // Pink for downward prediction
                }

                lineColors[i][bar] = color;
                predictiveSeries[i].Colors[bar] = ConvertToDrawingColor(lineColors[i][bar]);

                // Check if the predicted price crosses over the previous close price
                bool crossedPrice = DidCrossPrice(previousClosePrice, currentPrice, predictedPrice);

                // Apply reward or penalty for price crossing
                if (crossedPrice)
                {
                    // Reward for crossing the price
                    consistencyPenaltyWeight -= penaltyAdjustmentRate; // Reduce penalty (reward effect)
                    if (consistencyPenaltyWeight < minPenalty) consistencyPenaltyWeight = minPenalty;
                }
                else
                {
                    // Penalty for failing to adapt to trend change
                    consistencyPenaltyWeight += penaltyAdjustmentRate; // Increase penalty
                    if (consistencyPenaltyWeight > maxPenalty) consistencyPenaltyWeight = maxPenalty;
                }

                // Learning only for finalized candles
                if (!isLiveCandle && bar + predictionTimeframes[i] < BarsCount)
                {
                    double futureClosePrice = (double)GetCandle(bar + predictionTimeframes[i]).Close;
                    double[] target = { futureClosePrice - currentPrice };

                    // Backpropagate and update weights with finalized data
                    BackPropagate(normalizedInput, target, i, bar, predictedChange);
                }
                else
                {
                    // Handle cases where no future data is available
                    predictiveSeries[i][bar] = (decimal)(currentPrice + predictedChange);  // Use the current prediction
                }
            }
        }
    }


    // Helper function to extract features from candle data
    private double[] ExtractCandleFeatures(int bar, int lookback)
    {
        double[] features = new double[6]; // Updated from 7 to 6
        double totalVolume = 0;
        double maxOI = double.MinValue;
        double minOI = double.MaxValue;
        double totalDelta = 0;
        double maxDelta = double.MinValue;
        double minDelta = double.MaxValue;
        double totalMaxVolumePriceInfoVolume = 0;

        for (int j = bar - lookback + 1; j <= bar; j++)
        {
            if (j < 0)
                continue;

            var candle_j = GetCandle(j);

            totalVolume += (double)candle_j.Volume;

            if ((double)candle_j.MaxOI > maxOI)
                maxOI = (double)candle_j.MaxOI;
            if ((double)candle_j.MinOI < minOI)
                minOI = (double)candle_j.MinOI;

            totalDelta += (double)candle_j.Delta;

            if ((double)candle_j.MaxDelta > maxDelta)
                maxDelta = (double)candle_j.MaxDelta;
            if ((double)candle_j.MinDelta < minDelta)
                minDelta = (double)candle_j.MinDelta;

            totalMaxVolumePriceInfoVolume += (double)candle_j.MaxVolumePriceInfo.Volume;
        }

        features[0] = totalVolume / lookback;
        features[1] = maxOI;
        features[2] = maxOI - minOI;
        features[3] = totalDelta / lookback;
        features[4] = maxDelta - minDelta;
        features[5] = totalMaxVolumePriceInfoVolume / lookback;

        return features;
    }

    void SaveOptimalWeightsAndBiases(int timeframeIndex)
    {
        for (int i = 0; i < weightsInputHidden1[timeframeIndex].Length; i++)
            for (int j = 0; j < weightsInputHidden1[timeframeIndex][i].Length; j++)
                optimalWeightsInputHidden1[timeframeIndex][i][j] = weightsInputHidden1[timeframeIndex][i][j];

        for (int i = 0; i < weightsHidden1Hidden2[timeframeIndex].Length; i++)
            for (int j = 0; j < weightsHidden1Hidden2[timeframeIndex][i].Length; j++)
                optimalWeightsHidden1Hidden2[timeframeIndex][i][j] = weightsHidden1Hidden2[timeframeIndex][i][j];

        for (int i = 0; i < weightsHidden2Hidden3[timeframeIndex].Length; i++)
            for (int j = 0; j < weightsHidden2Hidden3[timeframeIndex][i].Length; j++)
                optimalWeightsHidden2Hidden3[timeframeIndex][i][j] = weightsHidden2Hidden3[timeframeIndex][i][j];

        for (int i = 0; i < weightsHidden3Output[timeframeIndex].Length; i++)
            for (int j = 0; j < weightsHidden3Output[timeframeIndex][i].Length; j++)
                optimalWeightsHidden3Output[timeframeIndex][i][j] = weightsHidden3Output[timeframeIndex][i][j];

        for (int i = 0; i < biasesHidden1[timeframeIndex].Length; i++)
            optimalBiasesHidden1[timeframeIndex][i] = biasesHidden1[timeframeIndex][i];

        for (int i = 0; i < biasesHidden2[timeframeIndex].Length; i++)
            optimalBiasesHidden2[timeframeIndex][i] = biasesHidden2[timeframeIndex][i];

        for (int i = 0; i < biasesHidden3[timeframeIndex].Length; i++)
            optimalBiasesHidden3[timeframeIndex][i] = biasesHidden3[timeframeIndex][i];

        for (int i = 0; i < biasesOutput[timeframeIndex].Length; i++)
            optimalBiasesOutput[timeframeIndex][i] = biasesOutput[timeframeIndex][i];
    }

    private void AdjustPenaltyBasedOnPerformance()
    {
        double avgError = 0;
        for (int i = 0; i < minErrors.Length; i++)
        {
            avgError += minErrors[i];
        }
        avgError /= minErrors.Length;

        if (avgError > 0.05)
        {
            consistencyPenaltyWeight += penaltyAdjustmentRate;
        }
        else
        {
            consistencyPenaltyWeight -= penaltyAdjustmentRate;
        }
        consistencyPenaltyWeight = Math.Max(minPenalty, Math.Min(maxPenalty, consistencyPenaltyWeight));
    }

    void BackPropagate(double[] input, double[] target, int timeframeIndex, int bar, double currentPredictionDifference)
    {
        // Step 1: Forward pass to compute activations for each layer
        double[] hidden1 = ActivationFunction(DotProduct(input, weightsInputHidden1[timeframeIndex], biasesHidden1[timeframeIndex]));
        double[] hidden2 = ActivationFunction(DotProduct(hidden1, weightsHidden1Hidden2[timeframeIndex], biasesHidden2[timeframeIndex]));
        double[] hidden3 = ActivationFunction(DotProduct(hidden2, weightsHidden2Hidden3[timeframeIndex], biasesHidden3[timeframeIndex]));
        double[] output = DotProduct(hidden3, weightsHidden3Output[timeframeIndex], biasesOutput[timeframeIndex]);

        // Step 2: Compute standard prediction error
        double[] outputError = new double[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            outputError[i] = output[i] - target[i];
        }

        // Step 3: Compute consistency penalty based on sign comparison
        double consistencyPenalty = ComputeConsistencyPenalty(timeframeIndex, bar, output[0]);
        AdjustPenaltyBasedOnPerformance();

        // Incorporate consistency penalty into outputError
        outputError[0] += consistencyPenalty;

        // Step 4: Compute volatility-based adaptive learning rate
        double volatility = CalculateVolatility(timeframeIndex, bar);
        double adaptiveLearningRate = learningRate / (1 + volatility); // Lower the learning rate during volatile periods

        // Step 5: Compute the deltas for output layer
        double[] outputDelta = new double[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            outputDelta[i] = outputError[i];  // No activation function on the output layer (linear activation)
        }

        // Step 6: Backpropagate through hidden3 to output layer
        double[][] weightsHidden3OutputT = Transpose(weightsHidden3Output[timeframeIndex]);
        double[] hidden3Error = DotProduct(outputDelta, weightsHidden3OutputT, new double[hiddenSizes[2]]);
        double[] hidden3Delta = new double[hiddenSizes[2]];
        for (int i = 0; i < hiddenSizes[2]; i++)
        {
            hidden3Delta[i] = hidden3Error[i] * (1 - hidden3[i] * hidden3[i]);  // Derivative of tanh
        }

        // Step 7: Backpropagate through hidden2 to hidden3 layer
        double[][] weightsHidden2Hidden3T = Transpose(weightsHidden2Hidden3[timeframeIndex]);
        double[] hidden2Error = DotProduct(hidden3Delta, weightsHidden2Hidden3T, new double[hiddenSizes[1]]);
        double[] hidden2Delta = new double[hiddenSizes[1]];
        for (int i = 0; i < hiddenSizes[1]; i++)
        {
            hidden2Delta[i] = hidden2Error[i] * (1 - hidden2[i] * hidden2[i]);  // Derivative of tanh
        }

        // Step 8: Backpropagate through hidden1 to hidden2 layer
        double[][] weightsHidden1Hidden2T = Transpose(weightsHidden1Hidden2[timeframeIndex]);
        double[] hidden1Error = DotProduct(hidden2Delta, weightsHidden1Hidden2T, new double[hiddenSizes[0]]);
        double[] hidden1Delta = new double[hiddenSizes[0]];
        for (int i = 0; i < hiddenSizes[0]; i++)
        {
            hidden1Delta[i] = hidden1Error[i] * (1 - hidden1[i] * hidden1[i]);  // Derivative of tanh
        }

        // Step 9: Update the weights and biases (with adaptive learning rate)
        UpdateWeightsAndBiases(input, hidden1, hidden2, hidden3, hidden1Delta, hidden2Delta, hidden3Delta, outputDelta, timeframeIndex, adaptiveLearningRate);
    }


    private double ComputeConsistencyPenalty(int currentTimeframeIndex, int bar, double currentOutput)
    {
        double penalty = 0.0;

        // Determine if the current timeframe is short-term
        if (Array.Exists(shortTermTimeframes, element => element == predictionTimeframes[currentTimeframeIndex]))
        {
            // Compute the sign of the current prediction relative to the current price
            double currentPrice = (double)GetCandle(bar).Close;
            double shortTermChange = currentOutput; // Assuming output[0] is the predicted change
            int signShort = Math.Sign(shortTermChange);

            // Iterate over all long-term timeframes
            for (int i = 0; i < predictionTimeframes.Length; i++)
            {
                if (Array.Exists(shortTermTimeframes, element => element == predictionTimeframes[i]))
                    continue; // Skip short-term timeframes in this loop

                double longTermPredictedPrice = (double)predictiveSeries[i][bar];
                double longTermChange = longTermPredictedPrice - currentPrice;
                int signLong = Math.Sign(longTermChange);

                if (signShort == signLong)
                {
                    // Penalize if signs are the same
                    penalty += consistencyPenaltyWeight;
                    if (penalty > maxPenalty) penalty = maxPenalty;
                }
                else
                {
                    // Reward (negative penalty) if signs are different
                    penalty -= consistencyPenaltyWeight;
                    if (penalty < minPenalty) penalty = minPenalty;
                }
            }
        }
        return penalty;
    }

    // New method to calculate volatility (simple version using price change)
    private double CalculateVolatility(int timeframeIndex, int bar)
    {
        double sumPriceChange = 0;
        int lookback = predictionTimeframes[timeframeIndex];

        for (int i = bar - lookback + 1; i <= bar; i++)
        {
            if (i < 0) continue;
            var candle = GetCandle(i);
            sumPriceChange += Math.Abs((double)(candle.Close - candle.Open));
        }

        double avgPriceChange = sumPriceChange / lookback;
        return avgPriceChange;
    }

    double[] ApplyDynamicBiases(double[] biases, double volatility, int timeframeIndex)
    {
        // Adjust biases dynamically based on volatility
        double adjustmentFactor = 1 + (volatility * 0.1); // Increase the factor as per volatility sensitivity
        double[] adjustedBiases = new double[biases.Length];

        for (int i = 0; i < biases.Length; i++)
        {
            adjustedBiases[i] = biases[i] * adjustmentFactor;
        }

        return adjustedBiases;
    }

    double[][] Transpose(double[][] matrix)
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

    void UpdateWeightsAndBiases(double[] input, double[] hidden1, double[] hidden2, double[] hidden3, double[] hidden1Delta, double[] hidden2Delta, double[] hidden3Delta, double[] outputDelta, int timeframeIndex, double adaptiveLearningRate)
    {
        double clipValue = 1.0;  // Gradient clipping value to prevent exploding gradients

        // Update weights and biases between input and hidden1
        for (int i = 0; i < input.Length; i++)
        {
            for (int j = 0; j < hiddenSizes[0]; j++)
            {
                double gradient = hidden1Delta[j] * input[i] + l2Regularization * weightsInputHidden1[timeframeIndex][i][j];
                gradient = Math.Max(Math.Min(gradient, clipValue), -clipValue);  // Clip gradient
                weightsInputHidden1[timeframeIndex][i][j] -= learningRate * gradient;
            }
        }

        for (int i = 0; i < hiddenSizes[0]; i++)
        {
            double gradient = hidden1Delta[i];
            gradient = Math.Max(Math.Min(gradient, clipValue), -clipValue);  // Clip gradient
            biasesHidden1[timeframeIndex][i] -= learningRate * gradient;
        }

        // Update weights and biases between hidden1 and hidden2
        for (int i = 0; i < hidden1.Length; i++)
        {
            for (int j = 0; j < hiddenSizes[1]; j++)
            {
                double gradient = hidden2Delta[j] * hidden1[i] + l2Regularization * weightsHidden1Hidden2[timeframeIndex][i][j];
                gradient = Math.Max(Math.Min(gradient, clipValue), -clipValue);  // Clip gradient
                weightsHidden1Hidden2[timeframeIndex][i][j] -= learningRate * gradient;
            }
        }

        for (int i = 0; i < hiddenSizes[1]; i++)
        {
            double gradient = hidden2Delta[i];
            gradient = Math.Max(Math.Min(gradient, clipValue), -clipValue);  // Clip gradient
            biasesHidden2[timeframeIndex][i] -= learningRate * gradient;
        }

        // Update weights and biases between hidden2 and hidden3
        for (int i = 0; i < hidden2.Length; i++)
        {
            for (int j = 0; j < hiddenSizes[2]; j++)
            {
                double gradient = hidden3Delta[j] * hidden2[i] + l2Regularization * weightsHidden2Hidden3[timeframeIndex][i][j];
                gradient = Math.Max(Math.Min(gradient, clipValue), -clipValue);  // Clip gradient
                weightsHidden2Hidden3[timeframeIndex][i][j] -= learningRate * gradient;
            }
        }

        for (int i = 0; i < hiddenSizes[2]; i++)
        {
            double gradient = hidden3Delta[i];
            gradient = Math.Max(Math.Min(gradient, clipValue), -clipValue);  // Clip gradient
            biasesHidden3[timeframeIndex][i] -= learningRate * gradient;
        }

        // Update weights and biases between hidden3 and output
        for (int i = 0; i < hidden3.Length; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                double gradient = outputDelta[j] * hidden3[i] + l2Regularization * weightsHidden3Output[timeframeIndex][i][j];
                gradient = Math.Max(Math.Min(gradient, clipValue), -clipValue);  // Clip gradient
                weightsHidden3Output[timeframeIndex][i][j] -= learningRate * gradient;
            }
        }

        for (int i = 0; i < outputSize; i++)
        {
            double gradient = outputDelta[i];
            gradient = Math.Max(Math.Min(gradient, clipValue), -clipValue);  // Clip gradient
            biasesOutput[timeframeIndex][i] -= learningRate * gradient;
        }
    }
}
