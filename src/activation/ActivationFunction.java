package activation;


import java.io.Serializable;

public interface ActivationFunction extends Serializable {

	public double activationFunction(double d);
	public double derivativeFunction(double d);
	
}