package util;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import exception.NeuralNetworkError;

public class TestParser {
	public static double[][] parse(String path, int neuronNumber)
			throws FileNotFoundException {
		Scanner sc = new Scanner(new File(path));

		int number = sc.nextInt();

		if (number % neuronNumber != 0)
			throw new NeuralNetworkError(
					"Incorrect number of doubles in test file.");

		double result[][] = new double[number / neuronNumber][neuronNumber];

		for (int i = 0; i < number / neuronNumber; i++)
			for (int j = 0; j < neuronNumber; j++)
				result[i][j] = sc.nextDouble();

		return result;
	}
}
