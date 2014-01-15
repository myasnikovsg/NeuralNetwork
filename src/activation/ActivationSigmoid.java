package activation;

public class ActivationSigmoid implements ActivationFunction {
	
	private static final long serialVersionUID = -5871206465890723959L;
	
	public double activationFunction(double d) {
		return 1.0 / (1 + Math.exp(-1.0 * d));
	}
	
	public double derivativeFunction(double d) {
		return d * (1.0 - d);
	}

}