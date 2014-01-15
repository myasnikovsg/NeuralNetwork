package feedforward;

import java.io.Serializable;
import java.util.ArrayList;

import util.ErrorCalculation;
import util.MatrixUtils;
import activation.ActivationLinear;
import activation.ActivationSigmoid;
import activation.ActivationTANH;


public class FFNNetwork implements Serializable {

	private static final long serialVersionUID = 1805517284957838695L;
	
	private FFNLayer inputLayer;
	private FFNLayer outputLayer;
	private ArrayList<FFNLayer> layers = new ArrayList<FFNLayer>();
	private int activationFunctionCode;
	
	public static FFNNetwork createFFN(int layerCount, int activationFunctionCode, int layerNeuronCount[]) {
		FFNNetwork result = new FFNNetwork();
		result.setActivationFunctionCode(new Integer(activationFunctionCode));
		for (int i = 0; i < layerCount; i++)
			switch (activationFunctionCode) {
			case 0: 
				result.addLayer(new FFNLayer(new ActivationSigmoid(), layerNeuronCount[i]));
				break;
			case 1 : 
				result.addLayer(new FFNLayer(new ActivationLinear(), layerNeuronCount[i]));
				break;
			case 2 : 
				result.addLayer(new FFNLayer(new ActivationTANH(), layerNeuronCount[i]));
				break;
			default :
				result.addLayer(new FFNLayer(new ActivationSigmoid(), layerNeuronCount[i]));
			}
			
		return result;
	}

	public FFNNetwork() {}

	public void addLayer(final FFNLayer layer) {
		if (outputLayer != null) {
			layer.setPrevious(this.outputLayer);
			outputLayer.setNext(layer);
		}

		if (layers.size() == 0) 
			inputLayer = outputLayer = layer;
		 else 
			outputLayer = layer;
		
		layers.add(layer);
	}

	public double calculateError(final double input[][], final double ideal[][]) {
		final ErrorCalculation errorCalculation = new ErrorCalculation();

		for (int i = 0; i < ideal.length; i++) {
			computeOutputs(input[i]);
			errorCalculation.updateError(outputLayer.getFire(), 
					ideal[i]);
		}
		
		return (errorCalculation.calculateRMS());
	}

	public int calculateNeuronCount() {
		int result = 0;
		
		for (FFNLayer layer : layers) 
			result += layer.getNeuronCount();
		
		return result;
	}

	@Override
	public Object clone() {
		FFNNetwork result = cloneStructure();
		
		double copy[] = MatrixUtils.networkToArray(this);
		MatrixUtils.arrayToNetwork(copy, result);
		
		return result;
	}

	public FFNNetwork cloneStructure() {
		FFNNetwork result = new FFNNetwork();

		for (FFNLayer layer : layers) {
			FFNLayer clonedLayer = new FFNLayer(layer.getActivationFunction(), layer.getNeuronCount());
			result.addLayer(clonedLayer);
		}

		return result;
	}

	public double[] computeOutputs(double input[]) {

		for (FFNLayer layer : layers) 
			if (layer.isInput()) 
				layer.computeOutputs(input);
			else if (layer.isHidden()) 
				layer.computeOutputs(null);
			
		return outputLayer.getFire();
	}

	public int getHiddenLayerCount() {
		return layers.size() - 2;
	}
	
	public int getActivationFunctionCode() {
		return activationFunctionCode;
	}

	public void setActivationFunctionCode(int activationFunctionCode) {
		this.activationFunctionCode = activationFunctionCode;
	}

	public ArrayList<FFNLayer> getHiddenLayers() {
		ArrayList<FFNLayer> result = new ArrayList<FFNLayer>();
		
		for (final FFNLayer layer : layers) 
			if (layer.isHidden()) 
				result.add(layer);
				
		return result;
	}

	public FFNLayer getInputLayer() {
		return inputLayer;
	}

	public ArrayList<FFNLayer> getLayers() {
		return layers;
	}

	public FFNLayer getOutputLayer() {
		return outputLayer;
	}
	
	public int getWeightMatrixSize() {
		int result = 0;
		
		for (FFNLayer layer : layers) {
			result += layer.getMatrixSize();
		}
		
		return result;
	}

	public void reset(double min, double max) {
		for (FFNLayer layer : layers) 
			layer.reset(min, max);
	}
}