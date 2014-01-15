package prune;

import java.util.ArrayList;

import train.BPTrain;
import activation.ActivationFunction;
import activation.ActivationLinear;
import activation.ActivationSigmoid;
import activation.ActivationTANH;
import feedforward.FFNLayer;
import feedforward.FFNNetwork;

public class Prune {
	private FFNNetwork network;
	private ActivationFunction activationFunction;

	private double input[][];
	private double ideal[][];

	private double learnRate;
	private double momentum;
	private double maxError;
	private double error;
	private double markErrorRate;

	private int sinceMark;
	private int cycles;

	private int hiddenNeuronCount;

	private boolean done;

	BPTrain bp;

	public Prune(int activationFunctionCode, double learnRate, double momentum, double input[][],
			double ideal[][], double maxError) {
		switch (activationFunctionCode) {
		case 0: 
			this.activationFunction = new ActivationSigmoid();
			break;
		case 1 : 
			this.activationFunction = new ActivationLinear();
			break;
		case 2 : 
			this.activationFunction = new ActivationTANH();
			break;
		default :
			this.activationFunction = new ActivationSigmoid();
		}
		this.learnRate = learnRate;
		this.momentum = momentum;
		this.input = input;
		this.ideal = ideal;
		this.maxError = maxError;
	}

	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	public void setActivationFunction(ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	public Prune(FFNNetwork network, double train[][], double ideal[][],
			double maxError) {
		this.network = network;
		this.input = train;
		this.ideal = ideal;
		this.maxError = maxError;
	}

	private FFNNetwork clipHiddenNeuron(int neuron) {
		FFNNetwork result = (FFNNetwork) network.clone();
		ArrayList<FFNLayer> c = result.getHiddenLayers();
		Object layers[] = c.toArray();
		((FFNLayer) layers[0]).prune(neuron);
		return result;
	}

	private double calcError(FFNNetwork network) {
		return network.calculateError(input, ideal);

	}

	private boolean findNeuron() {
		for (int i = 0; i < network.getHiddenLayerCount(); i++) {
			FFNNetwork trial = clipHiddenNeuron(i);
			double error = calcError(trial);
			if (error < maxError) {
				network = trial;
				return true;
			}
		}
		return false;
	}

	public FFNNetwork getCurrentNetwork() {
		return network;
	}

	public int getCycles() {
		return cycles;
	}

	public boolean getDone() {
		return done;
	}

	public double getError() {
		return error;
	}

	private int getHiddenCount() {
		ArrayList<FFNLayer> c = network.getHiddenLayers();
		Object layers[] = c.toArray();
		return ((FFNLayer) layers[0]).getNeuronCount();
	}

	public int getHiddenNeuronCount() {
		return hiddenNeuronCount;
	}

	private void increment() {
		boolean f = false;

		if (markErrorRate == 0) {
			markErrorRate = error;
			sinceMark = 0;
		} else {
			sinceMark++;
			if (sinceMark > 2000 * getHiddenNeuronCount()) {
				if ((markErrorRate - error) < 0.01)
					f = true;
				markErrorRate = error;
				sinceMark = 0;
			}
		}

		if (error < maxError)
			done = true;

		if (f) {
			cycles = 0;
			hiddenNeuronCount++;

			network = new FFNNetwork();
			network.addLayer(new FFNLayer(activationFunction, input[0].length));
			network.addLayer(new FFNLayer(activationFunction, hiddenNeuronCount));
			network.addLayer(new FFNLayer(activationFunction, ideal[0].length));
			network.reset(-1, 1);

			bp = new BPTrain(network, input, ideal, learnRate, momentum);
		}
	}

	public void pruneIncremental() {
		if (done) {
			return;
		}
		
		int hiddenNeuronCountMem = getHiddenNeuronCount(); 
		while (hiddenNeuronCountMem == getHiddenCount()) {
			bp.iteration();

			error = bp.getError();
			cycles++;

			increment();
		}
	}

	public int pruneSelective() {
		int i = getHiddenCount();
		while (findNeuron()) {
			;
		}
		return (i - getHiddenCount());
	}

	public void startIncremental() {
		hiddenNeuronCount = 1;
		cycles = 0;
		done = false;

		network = new FFNNetwork();
		network.addLayer(new FFNLayer(activationFunction, input[0].length));
		network.addLayer(new FFNLayer(activationFunction, hiddenNeuronCount));
		network.addLayer(new FFNLayer(activationFunction, ideal[0].length));
		network.reset(-1, 1);

		bp = new BPTrain(network, input, ideal, learnRate, momentum);
	}

}