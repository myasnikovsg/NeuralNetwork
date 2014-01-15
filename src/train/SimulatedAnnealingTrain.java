package train;

import feedforward.FFNNetwork;
import util.MatrixUtils;

public class SimulatedAnnealingTrain implements Train {
	private int cycles;

	private double error;
	private double ideal[][];

	private double input[][];
	private FFNNetwork network;
	private double startTemperature;
	private double stopTemperature;
	private double temperature;

	public SimulatedAnnealingTrain(FFNNetwork network, double startTemp,
			double stopTemp, int cycles, double input[][], double ideal[][]) {
		this.network = network;
		this.input = input;
		this.ideal = ideal;
		this.temperature = startTemp;
		this.startTemperature = startTemp;
		this.stopTemperature = stopTemp;
		this.cycles = cycles;
	}

	public double determineError() {
		return network.calculateError(input, ideal);
	}

	public double[] getArray() {
		return MatrixUtils.networkToArray(network);
	}

	public double[] getArrayCopy() {
		return getArray();
	}

	public int getCycles() {
		return cycles;
	}

	public double getError() {
		return error;
	}

	public FFNNetwork getNetwork() {
		return network;
	}

	public double getStartTemperature() {
		return startTemperature;
	}

	public double getStopTemperature() {
		return stopTemperature;
	}

	public double getTemperature() {
		return temperature;
	}

	public void iteration() {
		double bestArray[];

		setError(determineError());
		bestArray = getArrayCopy();

		temperature = getStartTemperature();

		for (int i = 0; i < cycles; i++) {
			double curError;
			randomize();
			curError = determineError();
			if (curError < getError()) {
				bestArray = getArrayCopy();
				setError(curError);
			}

			putArray(bestArray);
			double ratio = (getStopTemperature() - getStartTemperature())
					/ (getCycles() - 1);
			temperature -= ratio;
		}
	}

	public void putArray(double[] array) {
		MatrixUtils.arrayToNetwork(array, network);
	}

	public void randomize() {
		double array[] = MatrixUtils.networkToArray(network);

		for (int i = 0; i < array.length; i++) {
			double add = 0.5 - (Math.random());
			add /= getStartTemperature();
			add *= this.temperature;
			array[i] = array[i] + add;
		}

		MatrixUtils.arrayToNetwork(array, network);
	}

	public void setCycles(int cycles) {
		this.cycles = cycles;
	}

	public void setError(double error) {
		this.error = error;
	}

	public void setStartTemperature(double startTemperature) {
		this.startTemperature = startTemperature;
	}

	public void setStopTemperature(double stopTemperature) {
		this.stopTemperature = stopTemperature;
	}

	public void setTemperature(double temperature) {
		this.temperature = temperature;
	}

}