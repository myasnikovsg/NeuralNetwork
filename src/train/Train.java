package train;

import feedforward.FFNNetwork;

public interface Train {

	public double getError();

	public FFNNetwork getNetwork();

	public void iteration();

}