package matrix;

public class MatrixMath {

	public static Matrix add(Matrix a, Matrix b) {
		double result[][] = new double[a.getRows()][a.getCols()];

		for (int r = 0; r < a.getRows(); r++)
			for (int c = 0; c < a.getCols(); c++)
				result[r][c] = a.get(r, c) + b.get(r, c);

		return new Matrix(result);
	}

	public static void copy(Matrix source, Matrix target) {
		for (int r = 0; r < source.getRows(); r++)
			for (int c = 0; c < source.getCols(); c++)
				target.set(r, c, source.get(r, c));
	}

	public static Matrix deleteCol(Matrix matrix, int deleted) {
		double newMatrix[][] = new double[matrix.getRows()][matrix.getCols() - 1];

		for (int r = 0; r < matrix.getRows(); r++) {
			int targetCol = 0;
			for (int c = 0; c < matrix.getCols(); c++)
				if (c != deleted) {
					newMatrix[r][targetCol] = matrix.get(r, c);
					targetCol++;
				}
		}

		return new Matrix(newMatrix);
	}

	public static Matrix deleteRow(Matrix matrix, int deleted) {
		double newMatrix[][] = new double[matrix.getRows() - 1][matrix
				.getCols()];

		int targetRow = 0;

		for (int r = 0; r < matrix.getRows(); r++)
			if (r != deleted) {
				for (int c = 0; c < matrix.getCols(); c++)
					newMatrix[targetRow][c] = matrix.get(r, c);
				targetRow++;
			}

		return new Matrix(newMatrix);
	}

	public static Matrix divide(Matrix a, double b) {
		double result[][] = new double[a.getRows()][a.getCols()];

		for (int r = 0; r < a.getRows(); r++)
			for (int c = 0; c < a.getCols(); c++)
				result[r][c] = a.get(r, c) / b;

		return new Matrix(result);
	}

	public static double dotProduct(Matrix a, Matrix b) {

		double aArray[] = a.toPackedArray();
		double bArray[] = b.toPackedArray();

		double result = 0;
		final int length = aArray.length;

		for (int i = 0; i < length; i++)
			result += aArray[i] * bArray[i];

		return result;
	}

	public static Matrix multiply(Matrix a, double b) {
		double result[][] = new double[a.getRows()][a.getCols()];

		for (int r = 0; r < a.getRows(); r++)
			for (int c = 0; c < a.getCols(); c++)
				result[r][c] = a.get(r, c) * b;

		return new Matrix(result);
	}

	public static Matrix multiply(Matrix a, Matrix b) {

		double result[][] = new double[a.getRows()][b.getCols()];

		for (int r = 0; r < a.getRows(); r++)
			for (int ñ = 0; ñ < b.getCols(); ñ++)
				for (int i = 0; i < a.getCols(); i++)
					result[r][ñ] += a.get(r, i) * b.get(i, ñ);

		return new Matrix(result);
	}

	public static Matrix subtract(Matrix a, Matrix b) {

		double result[][] = new double[a.getRows()][a.getCols()];

		for (int r = 0; r < a.getRows(); r++)
			for (int c = 0; c < a.getCols(); c++)
				result[r][c] = a.get(r, c) - b.get(r, c);

		return new Matrix(result);
	}

	public static double vectorLength(Matrix a) {
		double v[] = a.toPackedArray();

		double rtn = 0.0;
		for (int i = 0; i < v.length; i++)
			rtn += Math.pow(v[i], 2);

		return Math.sqrt(rtn);
	}

}