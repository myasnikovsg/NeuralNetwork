package matrix;

import java.io.Serializable;

public class Matrix implements Cloneable, Serializable {

	private static final long serialVersionUID = 1057422033939779358L;

	public static Matrix createColumnMatrix(double input[]) {
		double d[][] = new double[input.length][1];
		for (int row = 0; row < d.length; row++) {
			d[row][0] = input[row];
		}
		return new Matrix(d);
	}

	public static Matrix createRowMatrix(double input[]) {
		double d[][] = new double[1][input.length];
		System.arraycopy(input, 0, d[0], 0, input.length);
		return new Matrix(d);
	}

	double matrix[][];

	public Matrix(boolean sourceMatrix[][]) {
		matrix = new double[sourceMatrix.length][sourceMatrix[0].length];
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getCols(); c++) {
				if (sourceMatrix[r][c]) {
					this.set(r, c, 1);
				} else {
					this.set(r, c, -1);
				}
			}
		}
	}

	public Matrix(double sourceMatrix[][]) {
		matrix = new double[sourceMatrix.length][sourceMatrix[0].length];
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getCols(); c++) {
				this.set(r, c, sourceMatrix[r][c]);
			}
		}
	}

	public Matrix(int rows, int cols) {
		matrix = new double[rows][cols];
	}

	public void add(int row, int col, double value) {
		set(row, col, get(row, col) + value);
	}

	public void clear() {
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getCols(); c++) {
				set(r, c, 0);
			}
		}
	}

	@Override
	public Matrix clone() {
		return new Matrix(matrix);
	}

	public int fromPackedArray(double[] array, int index) {
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getCols(); c++) {
				this.matrix[r][c] = array[index++];
			}
		}
		return index;
	}

	public double get(int row, int col) {
		return matrix[row][col];
	}

	public Matrix getCol(int col) {
		double newMatrix[][] = new double[getRows()][1];

		for (int row = 0; row < getRows(); row++) {
			newMatrix[row][0] = matrix[row][col];
		}

		return new Matrix(newMatrix);
	}

	public int getCols() {
		return matrix[0].length;
	}

	public Matrix getRow(int row) {
		double newMatrix[][] = new double[1][getCols()];

		for (int col = 0; col < getCols(); col++) {
			newMatrix[0][col] = matrix[row][col];
		}

		return new Matrix(newMatrix);
	}

	public int getRows() {
		return matrix.length;
	}

	public boolean isVector() {
		if (getRows() == 1) {
			return true;
		} else {
			return getCols() == 1;
		}
	}

	public void ramdomize(double min, double max) {
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getCols(); c++) {
				this.matrix[r][c] = (Math.random() * (max - min)) + min;
			}
		}
	}

	public void set(int row, int col, double value) {
		matrix[row][col] = value;
	}

	public int size() {
		return matrix[0].length * matrix.length;
	}

	public double sum() {
		double result = 0;
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getCols(); c++) {
				result += matrix[r][c];
			}
		}
		return result;
	}

	public double[] toPackedArray() {
		double result[] = new double[getRows() * getCols()];

		int index = 0;
		for (int r = 0; r < getRows(); r++) {
			for (int c = 0; c < getCols(); c++) {
				result[index++] = this.matrix[r][c];
			}
		}

		return result;
	}

}