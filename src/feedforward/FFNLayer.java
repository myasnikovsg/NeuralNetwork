package feedforward;

import java.io.Serializable;

import matrix.Matrix;
import matrix.MatrixMath;
import activation.ActivationFunction;

public class FFNLayer implements Serializable {

	private static final long serialVersionUID = -8039114834799422228L;

	private double fire[];

	private Matrix matrix;

	private FFNLayer next;
	private FFNLayer previous;

	private ActivationFunction activationFunction;

	public FFNLayer(ActivationFunction thresholdFunction, int neuronCount) {
		fire = new double[neuronCount];
		activationFunction = thresholdFunction;
	}

	public FFNLayer cloneStructure() {
		return new FFNLayer(activationFunction, getNeuronCount());
	}

	public double[] computeOutputs(double pattern[]) {
		if (pattern != null) 
			for (int i = 0; i < getNeuronCount(); i++) 
				setFire(i, pattern[i]);

		Matrix inputMatrix = createInputMatrix(fire);

		for (int i = 0; i < next.getNeuronCount(); i++) {
			Matrix col = matrix.getCol(i);
			double sum = MatrixMath.dotProduct(col, inputMatrix);
			next.setFire(i, activationFunction.activationFunction(sum));
		}

		return fire;
	}

	private Matrix createInputMatrix(double pattern[]) {
		Matrix result = new Matrix(1, pattern.length + 1);
		for (int i = 0; i < pattern.length; i++) 
			result.set(0, i, pattern[i]);

		result.set(0, pattern.length, 1);

		return result;
	}

	public double[] getFire() {
		return fire;
	}

	public double getFire(int index) {
		return fire[index];
	}

	public Matrix getMatrix() {
		return matrix;
	}

	public int getMatrixSize() {
		if (matrix == null) 
			return 0;
		 else 
			return matrix.size();
	}

	public int getNeuronCount() {
		return fire.length;
	}
	
	public FFNLayer getNext() {
		return next;
	}

	public FFNLayer getPrevious() {
		return previous;
	}

	public boolean hasMatrix() {
		return matrix != null;
	}

	public boolean isHidden() {
		return (next != null) && (previous != null);
	}

	public boolean isInput() {
		return previous == null;
	}

	public boolean isOutput() {
		return next == null;
	}

	public void prune(int neuron) {
		if (matrix != null) {
			setMatrix(MatrixMath.deleteRow(matrix, neuron));
		}

		FFNLayer previous = getPrevious();
		
		if (!isInput() && previous.hasMatrix()) 
			previous.setMatrix(MatrixMath.deleteCol(previous.getMatrix(), neuron));
	}

	public void reset(double min, double max) {
		if (hasMatrix()) {
			matrix.ramdomize(min, max);
		}
	}

	public void setFire(int index, double d) {
		fire[index] = d;
	}

	public void setMatrix(Matrix matrix) {
		if (hasMatrix()) {
			fire = new double[matrix.getRows() - 1];
		}
		this.matrix = matrix;
	}

	public void setNext(FFNLayer next) {
		this.next = next;
		matrix = new Matrix(getNeuronCount() + 1, next.getNeuronCount());
	}

	public void setPrevious(FFNLayer previous) {
		this.previous = previous;
	}

	public ActivationFunction getActivationFunction() {
		return this.activationFunction;
	}
}