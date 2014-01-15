package util;

public class ErrorCalculation {

	private double globalError;
	private int size;

	public double calculateRMS() {
		double err = Math.sqrt(globalError / (size));
		return err;
	}

	public void reset() {
		globalError = 0;
		size = 0;
	}

	public void updateError(double actual[], double ideal[]) {
		for (int i = 0; i < actual.length; i++) {
			double delta = ideal[i] - actual[i];
			globalError += delta * delta;
			size += ideal.length;
		}
	}
}