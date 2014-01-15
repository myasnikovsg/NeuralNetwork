package util;

import feedforward.FFNLayer;
import feedforward.FFNNetwork;

public class MatrixUtils {

	public static void arrayToNetwork(double array[], FFNNetwork network) {
		int index = 0;

		for (FFNLayer layer : network.getLayers())
			if (layer.getNext() != null)
				index = layer.getMatrix().fromPackedArray(array, index);
	}

	public static double[] networkToArray(FFNNetwork network) {
		int size = 0;

		for (final FFNLayer layer : network.getLayers())
			if (layer.hasMatrix())
				size += layer.getMatrixSize();

		final double result[] = new double[size];
				
		int index = 0;

		for (final FFNLayer layer : network.getLayers()) {
			if (layer.getNext() != null) {
				double matrix[] = layer.getMatrix().toPackedArray();
				for (int i = 0; i < matrix.length; i++)
					result[index++] = matrix[i];
			}
		}

		return result;
	}

}