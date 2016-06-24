using System;

// For 2014 Microsoft Build Conference attendees
// April 2-4, 2014
// San Francisco, CA
//
// This is source for a C# console application.
// To compile you can 1.) create a new Visual Studio
// C# console app project named BuildNeuralNetworkDemo
// then zap away the template code and replace with this code,
// or 2.) copy this code into notepad, save as NeuralNetworkProgram.cs
// on your local machine, launch the special VS command shell
// (it knows where the csc.exe compiler is), cd-navigate to
// the directory containing the .cs file, type 'csc.exe
// NeuralNetworkProgram.cs' and hit enter, and then after 
// the compiler creates NeuralNetworkProgram.exe, you can
// run from the command line.
//
// This is an enhanced neural network. It is fully-connected
// and feed-forward. The training algorithm is back-propagation
// with momentum and weight decay. The input data is normalized
// so training is quite fast.
namespace Quaetrix.NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin Build 2014 neural network demo");
            Console.WriteLine("\nData is the famous Iris flower set.");
            Console.WriteLine("Data is sepal length, sepal width, petal length, petal width -> iris species");
            Console.WriteLine("Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0 ");
            Console.WriteLine("The goal is to predict species from sepal length, width, petal length, width\n");

            Console.WriteLine("Raw data resembles:\n");
            Console.WriteLine(" 5.1, 3.5, 1.4, 0.2, Iris setosa");
            Console.WriteLine(" 7.0, 3.2, 4.7, 1.4, Iris versicolor");
            Console.WriteLine(" 6.3, 3.3, 6.0, 2.5, Iris virginica");
            Console.WriteLine(" ......\n");

            double[][] allData = Data.GetData();

            Console.WriteLine("\nFirst 6 rows of entire 150-item data set:");
            Maths.ShowMatrix(allData, 6, 1, true);

            Console.WriteLine("Creating 80% training and 20% test data matrices");
            double[][] trainData = null;
            double[][] testData = null;
            Test.MakeTrainTest(allData, out trainData, out testData);

            Console.WriteLine("\nFirst 5 rows of training data:");
            Maths.ShowMatrix(trainData, 5, 1, true);
            Console.WriteLine("First 3 rows of test data:");
            Maths.ShowMatrix(testData, 3, 1, true);

            Maths.Normalize(trainData, new int[] { 0, 1, 2, 3 });
            Maths.Normalize(testData, new int[] { 0, 1, 2, 3 });

            Console.WriteLine("\nFirst 5 rows of normalized training data:");
            Maths.ShowMatrix(trainData, 5, 1, true);
            Console.WriteLine("First 3 rows of normalized test data:");
            Maths.ShowMatrix(testData, 3, 1, true);

            Console.WriteLine("\nCreating a 4-input, 7-hidden, 3-output neural network");
            Console.Write("Hard-coded tanh function for input-to-hidden and softmax for ");
            Console.WriteLine("hidden-to-output activations");
            const int numInput = 4;
            const int numHidden = 7;
            const int numOutput = 3;
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            Console.WriteLine("\nInitializing weights and bias to small random values");
            nn.InitializeWeights();

            int maxEpochs = 2000;
            double learnRate = 0.05;
            double momentum = 0.01;
            double weightDecay = 0.0001;
            Console.WriteLine("Setting maxEpochs = 2000, learnRate = 0.05, momentum = 0.01, weightDecay = 0.0001");
            Console.WriteLine("Training has hard-coded mean squared error < 0.020 stopping condition");

            Console.WriteLine("\nBeginning training using incremental back-propagation\n");
            nn.Train(trainData, maxEpochs, learnRate, momentum, weightDecay);
            Console.WriteLine("Training complete");

            double[] weights = nn.GetWeights();
            Console.WriteLine("Final neural network weights and bias values:");
            Maths.ShowVector(weights, 10, 3, true);

            double trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData);
            Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));

            Console.WriteLine("\nEnd neural network program, press any key to continue...\n");
            Console.ReadLine();

        }        
    }  
}
