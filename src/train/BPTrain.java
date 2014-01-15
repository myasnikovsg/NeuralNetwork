package train;

import java.util.HashMap;

import feedforward.FFNLayer;
import feedforward.FFNNetwork;

public class BPTrain implements Train {

	private double error;
	private double ideal[][];
	private double input[][];

	private HashMap<FFNLayer, BPLayer> layerMap = new HashMap<FFNLayer, BPLayer>();

	private double learnRate;

	private double momentum;
	private FFNNetwork network;

	public BPTrain(FFNNetwork network, double input[][], double ideal[][],
			double learnRate, double momentum) {
		this.network = network;
		this.learnRate = learnRate;
		this.momentum = momentum;
		this.input = input;
		this.ideal = ideal;

		for (FFNLayer layer : network.getLayers()) {
			BPLayer bpl = new BPLayer(this, layer);
			layerMap.put(layer, bpl);
		}
	}

	public void calcError(double ideal[]) {

		for (FFNLayer layer : network.getLayers()) {
			getBPLayer(layer).clearError();
		}

		for (int i = network.getLayers().size() - 1; i >= 0; i--) {
			FFNLayer layer = network.getLayers().get(i);
			if (layer.isOutput())
				getBPLayer(layer).calcError(ideal);
			else
				getBPLayer(layer).calcError();
		}
	}

	public BPLayer getBPLayer(FFNLayer layer) {
		return layerMap.get(layer);
	}

	@Override
	public double getError() {
		return error;
	}

	@Override
	public FFNNetwork getNetwork() {
		return network;
	}

	@Override
	public void iteration() {
		for (int i = 0; i < input.length; i++) {
			network.computeOutputs(input[i]);
			calcError(ideal[i]);
		}
		learn();

		error = network.calculateError(input, ideal);
	}

	public void learn() {
		for (FFNLayer layer : network.getLayers())
			getBPLayer(layer).learn(learnRate, momentum);
	}
}