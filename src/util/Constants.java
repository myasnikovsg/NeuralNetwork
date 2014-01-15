package util;

public class Constants {

	public static final double BIGGEST_NUMBER = 1.0E20;
	public static final double SMALLEST_NUMBER = -1.0E20;

	public static double epsilonize(double d) {
		if (d < SMALLEST_NUMBER)
			return SMALLEST_NUMBER;
		else if (d > BIGGEST_NUMBER)
			return BIGGEST_NUMBER;
		else
			return d;
	}
}