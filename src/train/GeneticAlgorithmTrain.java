package train;

import java.util.Arrays;

import feedforward.FFNNetwork;

public class GeneticAlgorithmTrain implements Train {

	private Chromosome[] chromosomes;
	private int cutLength;
	protected double ideal[][];
	protected double input[][];
	private double matingPopulation;
	private double mutationPercent;
	private double percentToMate;
	private int populationSize;

	private boolean preventRepeat;

	public GeneticAlgorithmTrain(FFNNetwork network, int populationSize,
			double mutationPercent, double percentToMate, double input[][],
			double ideal[][]) {

		this.setMutationPercent(mutationPercent);
		this.setMatingPopulation(percentToMate * 2);
		this.setPopulationSize(populationSize);
		this.setPercentToMate(percentToMate);
		this.setCutLength(network.getWeightMatrixSize() / 3);

		this.input = input;
		this.ideal = ideal;

		setChromosomes(new Chromosome[getPopulationSize()]);
		for (int i = 0; i < getChromosomes().length; i++) {
			FFNNetwork chromosomeNetwork = (FFNNetwork) network.clone();

			Chromosome c = new Chromosome(this, chromosomeNetwork);
			c.updateGenes();
			setChromosome(i, c);
		}
		sortChromosomes();
	}
	public Chromosome getChromosome(int i) {
		return chromosomes[i];
	}

	public Chromosome[] getChromosomes() {
		return chromosomes;
	}

	public int getCutLength() {
		return cutLength;
	}

	public double getError() {
		FFNNetwork network = getNetwork();
		return network.calculateError(input, ideal);
	}

	public double[][] getIdeal() {
		return ideal;
	}

	public double[][] getInput() {
		return input;
	}

	public double getMatingPopulation() {
		return matingPopulation;
	}

	public double getMutationPercent() {
		return mutationPercent;
	}

	public FFNNetwork getNetwork() {
		Chromosome c = getChromosome(0);
		c.updateNetwork();
		return c.getNetwork();
	}

	public double getPercentToMate() {
		return percentToMate;
	}

	public int getPopulationSize() {
		return populationSize;
	}

	public boolean isPreventRepeat() {
		return preventRepeat;
	}

	public void iteration() {

		int countToMate = (int) (getPopulationSize() * getPercentToMate());
		int offspringCount = countToMate * 2;
		int offspringIndex = getPopulationSize() - offspringCount;
		int matingPopulationSize = (int) (getPopulationSize() * getMatingPopulation());

		for (int i = 0; i < countToMate; i++) {
			Chromosome mother = this.chromosomes[i];
			int fatherInt = (int) (Math.random() * matingPopulationSize);
			Chromosome father = this.chromosomes[fatherInt];
			Chromosome offspring1 = this.chromosomes[offspringIndex];
			Chromosome offspring2 = this.chromosomes[offspringIndex + 1];

			mother.mate(father, offspring1, offspring2);

			offspringIndex += 2;
		}

		sortChromosomes();
	}

	public void setChromosome(int i, Chromosome value) {
		chromosomes[i] = value;
	}

	public void setChromosomes(Chromosome[] chromosomes) {
		this.chromosomes = chromosomes;
	}

	public void setCutLength(int cutLength) {
		this.cutLength = cutLength;
	}

	public void setMatingPopulation(double matingPopulation) {
		this.matingPopulation = matingPopulation;
	}

	public void setMutationPercent(double mutationPercent) {
		this.mutationPercent = mutationPercent;
	}

	public void setPercentToMate(double percentToMate) {
		this.percentToMate = percentToMate;
	}

	public void setPopulationSize(int populationSize) {
		this.populationSize = populationSize;
	}

	public void setPreventRepeat(boolean preventRepeat) {
		this.preventRepeat = preventRepeat;
	}

	public void sortChromosomes() {
		Arrays.sort(chromosomes);
	}

}