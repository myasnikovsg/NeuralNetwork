package train;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import util.MatrixUtils;

import feedforward.FFNNetwork;

public class Chromosome implements Comparable<Chromosome> {

	private static final double RANGE = 2.0;
	private double cost;
	private double[] genes;
	private GeneticAlgorithmTrain geneticAlgorithm;
	private FFNNetwork network;

	public Chromosome(GeneticAlgorithmTrain genetic, FFNNetwork network) {
		setGeneticAlgorithm(genetic);
		setNetwork(network);

		initGenes(network.getWeightMatrixSize());
		updateGenes();
	}

	public void calculateCost() {
		updateNetwork();

		double input[][] = getGeneticAlgorithm().getInput();
		double ideal[][] = getGeneticAlgorithm().getIdeal();

		setCost(getNetwork().calculateError(input, ideal));
	}

	@Override
	public int compareTo(Chromosome other) {
		if (getCost() > other.getCost())
			return 1;
		else if (getCost() < other.getCost())
			return -1;
		else
			return 0;
	}

	public double getCost() {
		return cost;
	}

	public double getGene(int gene) {
		return genes[gene];
	}

	public double[] getGenes() {
		return genes;
	}

	public GeneticAlgorithmTrain getGeneticAlgorithm() {
		return geneticAlgorithm;
	}

	public FFNNetwork getNetwork() {
		return this.network;
	}

	private double getNotTaken(Chromosome source, Set<Double> taken) {
		int geneLength = source.size();

		for (int i = 0; i < geneLength; i++) {
			double trial = source.getGene(i);
			if (!taken.contains(trial)) {
				taken.add(trial);
				return trial;
			}
		}
		return Double.NaN;
	}

	public void initGenes(int length) {
		double result[] = new double[length];
		Arrays.fill(result, 0);
		setGenes(result);
	}

	public void mate(Chromosome father, Chromosome offspring1,
			Chromosome offspring2) {
		int geneLength = getGenes().length;

		int cutpoint1 = (int) (Math.random() * (geneLength - getGeneticAlgorithm()
				.getCutLength()));
		int cutpoint2 = cutpoint1 + getGeneticAlgorithm().getCutLength();

		Set<Double> taken1 = new HashSet<Double>();
		Set<Double> taken2 = new HashSet<Double>();

		for (int i = 0; i < geneLength; i++) {
			if ((i < cutpoint1) || (i > cutpoint2)) {
			} else {
				offspring1.setGene(i, father.getGene(i));
				offspring2.setGene(i, getGene(i));
				taken1.add(offspring1.getGene(i));
				taken2.add(offspring2.getGene(i));
			}
		}

		for (int i = 0; i < geneLength; i++) {
			if ((i < cutpoint1) || (i > cutpoint2)) {
				if (getGeneticAlgorithm().isPreventRepeat()) {
					offspring1.setGene(i, getNotTaken(this, taken1));
					offspring2.setGene(i, getNotTaken(father, taken2));
				} else {
					offspring1.setGene(i, getGene(i));
					offspring2.setGene(i, father.getGene(i));
				}
			}
		}

		if (Math.random() < getGeneticAlgorithm().getMutationPercent()) {
			offspring1.mutate();
		}
		if (Math.random() < getGeneticAlgorithm().getMutationPercent()) {
			offspring2.mutate();
		}

		offspring1.calculateCost();
		offspring2.calculateCost();
	}

	public void mutate() {
		int length = getGenes().length;
		for (int i = 0; i < length; i++) {
			double d = getGene(i);
			d *= ((RANGE * Math.random() * 2) - RANGE);
			setGene(i, d);
		}
	}

	public void setCost(double cost) {
		this.cost = cost;
	}

	public void setGene(int gene, double value) {
		genes[gene] = value;
	}

	public void setGenes(double[] genes) {
		this.genes = genes;
		calculateCost();
	}

	public void setGeneticAlgorithm(GeneticAlgorithmTrain geneticAlgorithm) {
		this.geneticAlgorithm = geneticAlgorithm;
	}

	public void setNetwork(FFNNetwork network) {
		this.network = network;
	}

	private int size() {
		return genes.length;
	}

	public void updateGenes() {
		setGenes(MatrixUtils.networkToArray(network));
	}

	public void updateNetwork() {
		MatrixUtils.arrayToNetwork(getGenes(), network);
	}
}