package activation;

public class ActivationTANH implements ActivationFunction {

	private static final long serialVersionUID = -6199807504275057899L;
	
	public double activationFunction(double d) {
		final double result = (Math.exp(d * 2.0) - 1.0) / (Math.exp(d * 2.0) + 1.0);
		return result;
	}
	
	public double derivativeFunction(double d) {
		return( 1.0 - Math.pow(activationFunction(d), 2.0));
	}
}