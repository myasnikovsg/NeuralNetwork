package activation;

import exception.NeuralNetworkError;

public class ActivationLinear implements ActivationFunction {
	
	private static final long serialVersionUID = 7311576294611586978L;
	
	public double activationFunction(double d) {
		return d;
	}

	public double derivativeFunction(double d) {
		throw new NeuralNetworkError("Can't use the linear activation function where a derivative is required.");
	}

}