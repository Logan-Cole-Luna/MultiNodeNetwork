import java.util.Arrays;
import java.util.Scanner;

public class multiNodeNetwork {

    private static double[][] inputArray = {{0, 0},
            {0, 1},
            {1, 0},
            {1, 1}};
    private static int[] y = {0, 0, 0, 1};
    private static double[] weights;
    //Weights for the second node
    private static double[] weights2;

    private static double learningRate = 0.1;
    private static int activationFunction;
    // the max iterations of the process
    private static int maxIterations;
    private static int iterations;

    private static double correctPredictions1 = 0.0;
    private static double correctPredictions2 = 0.0;

    public static void main(String[] args) {
        // User selects max # of iterations
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the max number of training iterations: ");
        maxIterations = sc.nextInt();

        // User selects function to use
        System.out.println("Select the activation function (1 = ReLU, 2 = Tanh, 3 = LeakyReLU, 4 = Sigmoid): ");
        System.out.println("Note: The sigmoid function never reaches 100% accuracy, and is present to display this issue ");
        activationFunction = sc.nextInt();

        // Creating weights & assigning low random number to weights
        weights = new double[inputArray[0].length];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() * 2 - 1.0;
            System.out.println(weights[i]);
        }

        weights2 = new double[inputArray[0].length];
        for (int i = 0; i < weights2.length; i++) {
            weights2[i] = Math.random() * 2 - 1.0;
            System.out.println(weights2[i]);
        }

        int i = 0;
        // While loop that goes until accuracy reaches 100% OR user selected iteration is reached
        while (true){
            correctPredictions1 = 0;
            correctPredictions2 = 0;
            double[] predictedOutput = new double[0];
            double error1 = 0;
            double error2 = 0;
            for (int j = 0; j < inputArray.length; j++) {
                // Calculating accuracy of predictions
                predictedOutput = predictOutput(inputArray[j]);
                //the threshold is 0.5 (as the predicted output is rounded to either 0 or 1).
                if (Math.round(predictedOutput[0]) == y[j]) {
                    correctPredictions1++;
                }
                if (Math.round(predictedOutput[1]) == y[j]) {
                    correctPredictions2++;
                }
                // Calculating weights
                error1 = y[j] - predictedOutput[0];
                error2 = y[j] - predictedOutput[1];

                // Updating weights
                for (int k = 0; k < weights.length; k++) {
                    weights[k] = weights[k] + learningRate * error1 * inputArray[j][k];
                }
                for (int k = 0; k < weights2.length; k++) {
                    weights2[k] = weights2[k] + learningRate * error2 * inputArray[j][k];
                }
            }
            // Printing out calculated results
            System.out.println("|Iteration of Node 1: " + (i + 1) + "| y = " + predictedOutput[0] + "| error = " + error1 +"| Current accuracy = "+ correctPredictions1 / inputArray.length * 100 + "%|");
            System.out.println("|Iteration of Node 2: " + (i + 1) + "| y = " + predictedOutput[1] + "| error = " + error2 +"| Current accuracy = "+ correctPredictions2 / inputArray.length * 100 + "%|");
            System.out.println("Node 1 Current Weights " + weights[0] + ", " + weights[1]+ "\t\tNode 2 Current Weights " + weights2[0] + ", " + weights2[1]);
            // Incrementing iteration for loop
            i++;
            // Ending loop if accuracing == 100%
            if ((correctPredictions1 / inputArray.length) == 1) {
                break;
            }
            else if ((correctPredictions2 / inputArray.length) == 1) {
                break;
            }
            // Ending loop if iterations == Max Iterations
            else if ( i == maxIterations){
                System.out.println("\nMax Iteration of " + maxIterations + " reached\n");
                break;
            }
        }
        // Printing final results
        System.out.println("\nFinal Iteration: " + (i));
        System.out.println("Final Weights: ");
        for (double weight : weights) {
            System.out.print(weight + " \n");
        }
        for (double weight : weights2) {
            System.out.print(weight + " \n");
        }
        System.out.println();

        // Calculating final accuracy
        correctPredictions1 = 0;
        correctPredictions2 = 0;
        for (int l = 0; l < inputArray.length; l++) {
            double[] predictedOutput = new double[0];
            predictedOutput = predictOutput(inputArray[l]);

            //the threshold is 0.5 (as the predicted output is rounded to either 0 or 1).
            if (Math.round(predictedOutput[0]) == y[l]) {
                correctPredictions1++;
            }
            if (Math.round(predictedOutput[1]) == y[l]) {
                correctPredictions2++;
            }
            System.out.print("|PredictedOutput for Node 1 = " + Math.round(predictedOutput[0]) + " |\t\t |PredictedOutput for Node 2 = " + Math.round(predictedOutput[1]) + " |\t\t | y =  " + y[l]+"|\n");

        }
        System.out.println("Accuracy of Node 1: " + correctPredictions1 / inputArray.length * 100 + "%");
        System.out.println("Accuracy of Node 2: " + correctPredictions2 / inputArray.length * 100 + "%");
    }

    // Function to calculate weighted sum and put this sum through the activation functions
    private static double[] predictOutput(double[] input) {
        double weightedSum1 = 0;
        double weightedSum2 = 0;

        for (int i = 0; i < input.length; i++) {
            weightedSum1 += input[i] * weights[i];
            weightedSum2 += input[i] * weights2[i];
        }
        double output1 = 0.0;
        double output2 = 0.0;
        switch (activationFunction) {
            case 1:
                output1 = ReLU(weightedSum1);
                output2 = ReLU(weightedSum2);
            case 2:
                output1 = tanh(weightedSum1);
                output2 = tanh(weightedSum2);
            case 3:
                output1 = leakyReLU(weightedSum1);
                output2 = leakyReLU(weightedSum2);
            case 4:
                output1 = sigmoid(weightedSum1);
                output2 = sigmoid(weightedSum2);
            default:
                return new double[]{output1, output2};
        }

    }
    private static double ReLU(double x) {
        return Math.max(0, x);
    }
    // Tanh (Hyperbolic Tangent): This activation function is commonly used in recurrent neural networks. It maps input values between -1 and 1.
    private static double tanh(double x) {
        return Math.tanh(x);
    }
    // The leaky ReLU adds a small negative slope for negative input values.
    private static double leakyReLU(double x) {
        return Math.max(0.01 * x, x);
    }
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
