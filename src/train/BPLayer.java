package train;

import util.Constants;
import feedforward.FFNLayer;
import matrix.Matrix;
import matrix.MatrixMath;

public class BPLayer {
	private Matrix accMatrixDelta;

	private int biasRow;

	private BPTrain bp;
	private double error[];

	private double errorDelta[];

	private FFNLayer layer;

	private Matrix matrixDelta;

	public BPLayer(BPTrain bp, FFNLayer layer) {
		this.bp = bp;
		this.layer = layer;

		int neuronCount = layer.getNeuronCount();

		this.error = new double[neuronCount];
		this.errorDelta = new double[neuronCount];

		if (layer.getNext() != null) {
			this.accMatrixDelta = new Matrix(layer.getNeuronCount() + 1, layer
					.getNext().getNeuronCount());
			this.matrixDelta = new Matrix(layer.getNeuronCount() + 1, layer
					.getNext().getNeuronCount());
			this.biasRow = layer.getNeuronCount();
		}
	}

	public void accumulateMatrixDelta(int i1, int i2, double value) {
		accMatrixDelta.add(i1, i2, value);
	}

	public void accumulateThresholdDelta(int index, double value) {
		accMatrixDelta.add(biasRow, index, value);
	}

	public void calcError() {

		BPLayer next = bp.getBPLayer(layer.getNext());

		for (int i = 0; i < layer.getNext().getNeuronCount(); i++) {
			for (int j = 0; j < layer.getNeuronCount(); j++) {
				accumulateMatrixDelta(j, i,
						next.getErrorDelta(i) * layer.getFire(j));
				setError(
						j,
						getError(j) + layer.getMatrix().get(j, i)
								* next.getErrorDelta(i));
			}
			accumulateThresholdDelta(i, next.getErrorDelta(i));
		}

		if (layer.isHidden())
			for (int i = 0; i < layer.getNeuronCount(); i++)
				setErrorDelta(i, Constants.epsilonize(calculateDelta(i)));
	}

	public void calcError(double ideal[]) {

		for (int i = 0; i < layer.getNeuronCount(); i++) {
			setError(i, ideal[i] - layer.getFire(i));
			setErrorDelta(i, Constants.epsilonize(calculateDelta(i)));
		}
	}

	private double calculateDelta(int i) {
		return getError(i)
				* layer.getActivationFunction().derivativeFunction(
						layer.getFire(i));
	}

	public void clearError() {
		for (int i = 0; i < layer.getNeuronCount(); i++) {
			error[i] = 0;
		}
	}

	public double getError(int index) {
		return error[index];
	}

	public double getErrorDelta(int index) {
		return errorDelta[index];
	}

	public void learn(double learnRate, double momentum) {
		if (layer.hasMatrix()) {
			Matrix m1 = MatrixMath.multiply(accMatrixDelta, learnRate);
			Matrix m2 = MatrixMath.multiply(matrixDelta, momentum);
			matrixDelta = MatrixMath.add(m1, m2);
			layer.setMatrix(MatrixMath.add(layer.getMatrix(), matrixDelta));
			accMatrixDelta.clear();
		}
	}

	public void setError(int index, double d) {
		error[index] = Constants.epsilonize(d);
	}

	public void setErrorDelta(int index, double d) {
		errorDelta[index] = d;
	}
}