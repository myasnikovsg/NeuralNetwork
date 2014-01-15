package exception;

public class NeuralNetworkError extends RuntimeException {

	private static final long serialVersionUID = 5141202608946216111L;

	public NeuralNetworkError(String msg) {
		super(msg);
	}
}