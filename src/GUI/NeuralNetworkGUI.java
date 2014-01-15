package GUI;

import java.awt.Color;
import java.awt.Component;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.StringTokenizer;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

import prune.Prune;

import train.BPTrain;
import train.GeneticAlgorithmTrain;
import train.SimulatedAnnealingTrain;
import train.Train;
import util.MatrixUtils;
import util.TestParser;
import exception.GUIException;
import exception.NeuralNetworkError;
import feedforward.FFNLayer;
import feedforward.FFNNetwork;

public class NeuralNetworkGUI extends JFrame implements ActionListener,
		KeyListener {

	private static final long serialVersionUID = 9211713612721942173L;

	private static int neuronDiameter = 50;

	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				NeuralNetworkGUI gui = new NeuralNetworkGUI();
				gui.setVisible(true);
			}
		});
	}

	private static Font thresholdFont = new Font(Font.SERIF, Font.BOLD, 15);
	private static Font weightFont = new Font(Font.SERIF, Font.PLAIN, 12);
	private static Font commandLineLogFont = new Font(Font.SERIF, Font.PLAIN,
			14);

	private static String invalidCommandString = "Invalid comand. Try again.";

	private ArrayList<String> commandLineHistory = new ArrayList<String>();

	private int commandLineHistoryPointer = 0;
	private int shower;
	private int curEpoch;
	private int precision = 2;

	private JTextArea logTextArea;
	private JScrollPane logScrollPane;
	private JTextField commandLineField;
	private JButton commandLineButton;
	private JLabel backgroundLabel;

	private BufferedImage background;

	private FFNNetwork ffnn;

	private Train train;
	
	private Prune prune;

	public NeuralNetworkGUI() {
		setTitle("Neural Networks");
		setSize(1200, 800);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(EXIT_ON_CLOSE);

		background = new BufferedImage(1200, 600, BufferedImage.TYPE_INT_RGB);
		Graphics g = background.getGraphics();
		g.setColor(Color.white);
		g.fillRect(0, 0, background.getWidth(), background.getHeight());
		g.dispose();

		backgroundLabel = new JLabel(new ImageIcon(background));
		backgroundLabel.setBounds(0, 0, background.getWidth(),
				background.getHeight());
		backgroundLabel.addMouseListener(new PopupListener());

		logTextArea = new JTextArea();
		logTextArea.setFont(commandLineLogFont);
		logTextArea.setEditable(false);
		logScrollPane = new JScrollPane(logTextArea,
				JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
				JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		logScrollPane.setBounds(20, 610, 1150, 80);

		commandLineField = new JTextField();
		commandLineField.setBounds(20, 695, 1080, 20);
		commandLineField.setFont(commandLineLogFont);
		commandLineField.setName("command line");
		commandLineField.addKeyListener(this);

		commandLineButton = new JButton("Go");
		commandLineButton.setBounds(1105, 695, 65, 20);
		commandLineButton.setName("command line");
		commandLineButton.addActionListener(this);

		this.setLayout(null);
		add(backgroundLabel);
		add(logScrollPane);
		add(commandLineField);
		add(commandLineButton);
	}

	public void appendLog(String s) {
		String time = new Timestamp(System.currentTimeMillis()).toString();
		logTextArea.append("[" + time.substring(time.indexOf(' ') + 1) + "] "
				+ s + "\n");
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		String source = ((Component) e.getSource()).getName();

		ChangeDialog dialog;

		if (source.startsWith("change threshold")) {
			int layer = Integer.parseInt(source.substring(
					source.indexOf('$') + 1, source.indexOf(':')));
			int number = Integer
					.parseInt(source.substring(source.indexOf(':') + 1));

			dialog = new ChangeDialog(ffnn
					.getLayers()
					.get(layer - 1)
					.getMatrix()
					.get(ffnn.getLayers().get(layer - 1).getNeuronCount(),
							number), layer, number, 0, true);

			dialog.setOwner(this);
			dialog.pack();
			dialog.setVisible(true);
		}

		if (source.startsWith("change weight")) {
			int layer = Integer.parseInt(source.substring(
					source.indexOf('$') + 1, source.indexOf(':')));
			int numberFrom = Integer.parseInt(source.substring(
					source.indexOf(':') + 1, source.indexOf('^')));
			int numberTo = Integer.parseInt(source.substring(source
					.indexOf('^') + 1));

			dialog = new ChangeDialog(ffnn.getLayers().get(layer).getMatrix()
					.get(numberFrom, numberTo), layer, numberFrom, numberTo,
					false);

			dialog.setOwner(this);
			dialog.pack();
			dialog.setVisible(true);
		}
		if (source.startsWith("command line")) {
			String command = commandLineField.getText().trim();
			commandLineHistory.add(command);
			commandLineHistoryPointer = commandLineHistory.size() - 1;
			commandLineField.setText("");
			dispatchCommand(command);
		}
	}

	public void dispatchCommand(String command) {
		StringTokenizer st = new StringTokenizer(command, ",{}[]() ");
		if (st.countTokens() == 0)
			return;
		String token = st.nextToken();
		if (token.equalsIgnoreCase("create")) { // +
			try {
				int layerCount = Integer.parseInt(st.nextToken());
				
				if (layerCount < 2)
					throw new GUIException("Can't create feedforward neural network with less than 2 layers.");
				
				int layerNeuronCount[] = new int[layerCount];
				
				for (int i = 0; i < layerCount; i++)
					layerNeuronCount[i] = Integer.parseInt(st.nextToken());
				
				token = st.nextToken();
				int activ = 1;
				
				if (token.equalsIgnoreCase("sigmoid")) 
					activ = 0;
				else
					if (token.equalsIgnoreCase("linear"))
						activ = 1;
					else
						if (token.equalsIgnoreCase("tanh")) 
							activ = 2;
						else
							throw new GUIException("No such activation function.");
				
				ffnn = FFNNetwork.createFFN(layerCount, activ, layerNeuronCount);
				
				drawFFNN();
				
				appendLog("New feedforward neural network created.");
				return;
			} catch (GUIException e) {
				appendLog(e.getMessage());
				return;
			} catch (Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		}
		if (token.equalsIgnoreCase("change")) { // +
			token = st.nextToken();
			if (token.equalsIgnoreCase("t")) { // +
				try {
					int layer = Integer.parseInt(st.nextToken());
					int number = Integer.parseInt(st.nextToken());
					double threshold = Double.parseDouble(st.nextToken());
					
					if (layer < 0 || layer > ffnn.getHiddenLayerCount() + 1)
						throw new GUIException("Invalid layer number.");
					if (layer == 0)
						throw new GUIException("No thresholds for neurons on first layer.");
					if (number < 0 || number > ffnn.getLayers().get(layer).getNeuronCount())
						throw new GUIException("Invaild neuron number.");
					
					changeThreshold(layer, number, threshold);
					
					appendLog("Threshold for neuron in layer " + layer + " number " + number + " changed to " + threshold);
					return;
				} catch (GUIException e) {
					appendLog(e.getMessage());
					return;
				} catch (Exception e) {
					appendLog(invalidCommandString);
					return;
				}
			}
			if (token.equalsIgnoreCase("w")) { // +
				try {
					int layer = Integer.parseInt(st.nextToken());
					int numberFrom = Integer.parseInt(st.nextToken());
					int numberTo = Integer.parseInt(st.nextToken());
					double weight = Double.parseDouble(st.nextToken());
					
					if (layer < 0 || layer > ffnn.getHiddenLayerCount() + 1)
						throw new GUIException("Invalid layer number.");
					if (layer == ffnn.getHiddenLayerCount() + 1)
						throw new GUIException("No weights for neurons on last layer.");
					if (numberFrom < 0 || numberFrom > ffnn.getLayers().get(layer).getNeuronCount() ||
							numberTo < 0 || numberTo > ffnn.getLayers().get(layer + 1).getNeuronCount())
						throw new GUIException("Invaild neuron number.");
					
					changeWeight(layer, numberFrom, numberTo, weight);
					
					appendLog("Weight of axon from neuron in layer " + layer + " number " + 
							numberFrom + " to neuron in layer " + layer + 1 + " number " + numberTo + 
							"changed to " + weight);
					return;
				} catch (GUIException e) {
					appendLog(e.getMessage());
					return;
				} catch (Exception e) {
					appendLog(invalidCommandString);
					return;
				}
			}
		}
		
		if (token.equalsIgnoreCase("reset")) // +
			try {
				double min = Double.parseDouble(st.nextToken());
				double max = Double.parseDouble(st.nextToken());
			
				if (max < min)
					throw new GUIException("Min > max? Huh.");
				
				ffnn.reset(min, max);
			
				drawFFNN();
			
				appendLog("Network reseted.");
				return;
			} catch (GUIException e) {
				appendLog(e.getMessage());
			} catch (Exception e) {
				appendLog(invalidCommandString);
			}
			
		if (token.equalsIgnoreCase("backpropagation")) // +
			try {
				double learnRate = Double.parseDouble(st.nextToken());
				double momentum = Double.parseDouble(st.nextToken());
				String inputPath = st.nextToken();
				String idealPath = st.nextToken();	
				
				if (learnRate < 0 || learnRate > 1)
					throw new GUIException("Learn rate must be in [0, 1]");
				if (momentum < 0 || momentum > 1) 
					throw new GUIException("Momentum must be in [0, 1]");
				
				double input[][];
				double ideal[][];
				
				try {
					input = TestParser.parse(inputPath, ffnn.getInputLayer().getNeuronCount());
				} catch (FileNotFoundException e) {
					appendLog("Input file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				try {
					ideal = TestParser.parse(idealPath, ffnn.getOutputLayer().getNeuronCount());
				} catch (FileNotFoundException e) {
					appendLog("Ideal file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				train = new BPTrain(ffnn, input, ideal, learnRate, momentum);
				curEpoch = 0;
				
				appendLog("Backpropagation training in progress. Use iterate command to view next iteration. Use stop command to end training.");
				return;
			} catch (GUIException e) {
				appendLog(e.getMessage());
			} catch (Exception e) {
				appendLog(invalidCommandString);
			}
		
		if (token.equalsIgnoreCase("genetic")) // +
			try {
				int populationSize = Integer.parseInt(st.nextToken());
				double mutationPercent = Double.parseDouble(st.nextToken());
				double percentToMate = Double.parseDouble(st.nextToken());
				
				if (populationSize < 1) 
					throw new GUIException("Invalid population size.");
				if (mutationPercent < 0 || mutationPercent > 1)
					throw new GUIException("Mutation percent must be in [0, 1]");
				if (percentToMate < 0 || percentToMate > 1) 
					throw new GUIException("Percent to mate must be in [0, 1]");
				
				String inputPath = st.nextToken();
				String idealPath = st.nextToken();
				
				double input[][];
				double ideal[][];
				
				try {
					input = TestParser.parse(inputPath, ffnn.getInputLayer().getNeuronCount());
				} catch (FileNotFoundException e) {
					appendLog("Input file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				try {
					ideal = TestParser.parse(idealPath, ffnn.getOutputLayer().getNeuronCount());
				} catch (FileNotFoundException e) {
					appendLog("Ideal file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				train = new GeneticAlgorithmTrain(ffnn, populationSize, mutationPercent, percentToMate, input, ideal);
				curEpoch = 0;
				
				appendLog("Genetic algorithm training in progress. Use iterate command to view next iteration. Use stop command to end training.");
				return;
			} catch (GUIException e) {
				appendLog(e.getMessage());
				return;
			} catch (Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		
		if (token.equalsIgnoreCase("annealing")) // +
			try {
				double startTemp = Double.parseDouble(st.nextToken());
				double stopTemp = Double.parseDouble(st.nextToken());
				int cycles = Integer.parseInt(st.nextToken());
				
				
				if (startTemp < stopTemp) 
					throw new GUIException("Start temperature < stop temperature? Huh.");
				if (cycles < 1)
					throw new GUIException("Invalid cycles.");
				
				String inputPath = st.nextToken();
				String idealPath = st.nextToken();
				
				double input[][];
				double ideal[][];
				
				try {
					input = TestParser.parse(inputPath, ffnn.getInputLayer().getNeuronCount());
				} catch (FileNotFoundException e) {
					appendLog("Input file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				try {
					ideal = TestParser.parse(idealPath, ffnn.getOutputLayer().getNeuronCount());
				} catch (FileNotFoundException e) {
					appendLog("Ideal file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				train = new SimulatedAnnealingTrain(ffnn, startTemp, stopTemp, cycles, input, ideal);
				curEpoch = 0;
				
				appendLog("Simulated annealing algorithm training in progress. Use iterate command to view next iteration. Use stop command to end training.");
				return;
			} catch (GUIException e) {
				appendLog(e.getMessage());
				return;
			} catch (Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		
		if (token.equalsIgnoreCase("iterate")) {
			if (train == null) {
				appendLog("No active training.");
				return;
			}
			
			for (int i = 0; i < shower; i++)
				train.iteration();
			
			curEpoch += shower;
			
			ffnn = train.getNetwork();
			
			drawFFNN();
			
			appendLog("Showing epoch number " + curEpoch + ". Current error = " + train.getError());
			return;
		}
		
		if (token.equalsIgnoreCase("stop")) {
			if (train == null) {
				appendLog("No train to stop.");
				return;
			}
			
			appendLog("Training ended.");
			train = null;
			return;
		}
		
		if (token.equalsIgnoreCase("offset")) 
			try {
				shower = Integer.parseInt(st.nextToken());
				if (shower < 0)
					throw new GUIException("Invalid show offset.");
				appendLog("Show offset set to " + shower);
				return;
			} catch (GUIException e) {
				appendLog(e.getMessage());
				return;
			} catch (Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		
		if (token.equalsIgnoreCase("error")) // + 
			try {
				String inputPath = st.nextToken();
				String idealPath = st.nextToken();
				
				double input[][];
				double ideal[][];
				
				try {
					input = TestParser.parse(inputPath, ffnn.getInputLayer().getNeuronCount());
				} catch (FileNotFoundException e) {
					appendLog("Input file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				try {
					ideal = TestParser.parse(idealPath, ffnn.getOutputLayer().getNeuronCount());
				} catch (FileNotFoundException e) {
					appendLog("Ideal file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}	
				
				appendLog("Current error = " + ffnn.calculateError(input, ideal));
				return;
			} catch (Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		
		if (token.equalsIgnoreCase("save")) // +
			try {
				String saveString = st.nextToken();
				try {
					PrintWriter pw = new PrintWriter(new File(saveString));
					double networkToArray[] = MatrixUtils.networkToArray(ffnn);
					pw.println(networkToArray.length);
					pw.println(ffnn.getActivationFunctionCode());
					pw.println(ffnn.getLayers().size());
					for (FFNLayer layer : ffnn.getLayers()) 
						pw.println(layer.getNeuronCount());
					for (int i = 0; i < networkToArray.length; i++)
						pw.println(networkToArray[i]);
					appendLog("Feedforward tree saved to " + saveString + ".");
					pw.close();
					return;
				} catch (IOException e) {
					appendLog("Invalid file path to save.");
					return;
				}
			} catch(Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		
		if (token.equalsIgnoreCase("precision")) // +
			try {
				precision = Integer.parseInt(st.nextToken());
				
				if (precision < 1)
					throw new GUIException("Invalid precision value.");
				
				drawFFNN();
				
				appendLog("Precision changed to " + precision);
				return;
			} catch (GUIException e) {
				appendLog(e.getMessage());
				return;
			} catch (Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		
		if (token.equalsIgnoreCase("load")) // +
			try {
				String loadString = st.nextToken();
				try {
					Scanner sc = new Scanner(new File(loadString));
					double arrayToNetwork[] = new double[sc.nextInt()];
					int activationFunctionCode = sc.nextInt();
					int layerCount = sc.nextInt();
					int layerNeuronCount[] = new int[layerCount];
					for (int i = 0; i < layerCount; i++)
						layerNeuronCount[i] = sc.nextInt();
					ffnn = FFNNetwork.createFFN(layerCount, activationFunctionCode, layerNeuronCount);
					for (int i = 0; i < arrayToNetwork.length; i++)
						arrayToNetwork[i] = sc.nextDouble();
					MatrixUtils.arrayToNetwork(arrayToNetwork, ffnn);
					
					drawFFNN();
					appendLog("Feedforward tree loaded from " + loadString + ".");
					sc.close();
					return;
				} catch (IOException e) {
					appendLog("Invalid file path to load.");
					return;
				}
			} catch(Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		
		if (token.equalsIgnoreCase("pruneInc")) // +
			try {
				double learnRate = Double.parseDouble(st.nextToken());
				double momentum = Double.parseDouble(st.nextToken());
				double maxError = Double.parseDouble(st.nextToken());
				int inputNeuronCount = Integer.parseInt(st.nextToken());
				int outputNeuronCount = Integer.parseInt(st.nextToken());
				String activationFunctionString = st.nextToken();
				String inputPath = st.nextToken();
				String idealPath = st.nextToken();	
				
				if (learnRate < 0 || learnRate > 1)
					throw new GUIException("Learn rate must be in [0, 1]");
				if (momentum < 0 || momentum > 1) 
					throw new GUIException("Momentum must be in [0, 1]");
				if (inputNeuronCount < 1)
					throw new GUIException("Input level must have at least 1 neuron.");
				if (outputNeuronCount < 1)
					throw new GUIException("Output level must have at least 1 neuron.");
				double input[][];
				double ideal[][];
				
				int activationFunctionCode = 0;
				if (activationFunctionString.equalsIgnoreCase("sigmoid")) 
					activationFunctionCode = 0;
				else
					if (activationFunctionString.equalsIgnoreCase("linear"))
						activationFunctionCode = 1;
					else
						if (activationFunctionString.equalsIgnoreCase("tanh")) 
							activationFunctionCode = 2;
						else
							throw new GUIException("No such activation function.");
				
				try {
					input = TestParser.parse(inputPath, inputNeuronCount);
				} catch (FileNotFoundException e) {
					appendLog("Input file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				try {
					ideal = TestParser.parse(idealPath, outputNeuronCount);
				} catch (FileNotFoundException e) {
					appendLog("Ideal file not found.");
					return;
				} catch (NeuralNetworkError e) {
					appendLog(e.getMessage());
					return;
				}
				
				prune = new Prune(activationFunctionCode, learnRate, momentum, input, ideal, maxError);
				prune.startIncremental();
				ffnn = prune.getCurrentNetwork();
				
				drawFFNN();
				
				appendLog("We will now construct new feedforward neural net with acceptable error. Use prune command to watch next step");
				return;
			} catch (GUIException e) {
				appendLog(e.getMessage());
				return;
			} catch (Exception e) {
				appendLog(invalidCommandString);
				return;
			}
		
//		if (token.equalsIgnoreCase("prunesel")) // something odd, matrix arrayoutofbound
//			try {
//				double maxError = Double.parseDouble(st.nextToken());
//				
//				String inputPath = st.nextToken();
//				String idealPath = st.nextToken();
//				
//				if (maxError < 0)
//					throw new GUIException("Max error must be positive.");
//				
//				double input[][];
//				double ideal[][];
//				
//				try {
//					input = TestParser.parse(inputPath, ffnn.getInputLayer().getNeuronCount());
//				} catch (FileNotFoundException e) {
//					appendLog("Input file not found.");
//					return;
//				} catch (NeuralNetworkError e) {
//					appendLog(e.getMessage());
//					return;
//				}
//				
//				try {
//					ideal = TestParser.parse(idealPath, ffnn.getOutputLayer().getNeuronCount());
//				} catch (FileNotFoundException e) {
//					appendLog("Ideal file not found.");
//					return;
//				} catch (NeuralNetworkError e) {
//					appendLog(e.getMessage());
//					return;
//				}
//				
//				prune = new Prune(ffnn, input, ideal, maxError);
//				int deletedNeurons =  prune.pruneSelective();
//				ffnn = prune.getCurrentNetwork();
//				
//				drawFFNN();
//				
//				appendLog("Network was pruned. " + deletedNeurons + " neurons were deleted in process.");
//				return;
//			} catch (GUIException e) {
//				appendLog(e.getMessage());
//				return;
//			} catch (Exception e) {
//				appendLog(invalidCommandString);
//				return;
//			}
		
		if (token.equalsIgnoreCase("prune")) {
			if (prune == null) {
				appendLog("No prune to act.");
				return;
			}
			
			prune.pruneIncremental();
			ffnn = prune.getCurrentNetwork();
			
			drawFFNN();
			
			appendLog("Incremental pruning. Hidden neuron count = " + prune.getHiddenNeuronCount() + ". Current error = " + 
					prune.getError());
			return;
		} 
		
		appendLog(invalidCommandString);
	}

	public void changeWeight(int layer, int numberFrom, int numberTo,
			double weight) {
		ffnn.getLayers().get(layer).getMatrix()
				.set(numberFrom, numberTo, weight);
		drawFFNN();
	}
	
	public void changeThreshold(int layer, int number, double threshold) {
		ffnn.getLayers()
				.get(layer - 1)
				.getMatrix()
				.set(ffnn.getLayers().get(layer - 1).getNeuronCount(), number,
						threshold);
		drawFFNN();
	}

	private class PopupListener extends MouseAdapter {
		public void mousePressed(MouseEvent e) {
			maybeShowPopup(e);
		}

		public void mouseReleased(MouseEvent e) {
			maybeShowPopup(e);
		}

		private void maybeShowPopup(MouseEvent e) {
			if (e.isPopupTrigger()) {
				Point eventP = e.getPoint();
				int layer = -1;
				int number = 0;
				for (int i = 0; i < ffnn.getLayers().size(); i++) {
					if (layer != -1)
						break;
					for (int j = 0; j < ffnn.getLayers().get(i)
							.getNeuronCount(); j++) {
						Point neuronP = getNeuronPoint(i, j);
						if (neuronP.distance(eventP) < neuronDiameter / 2) {
							layer = i;
							number = j;
							break;
						}
					}
				}

				if (layer == -1)
					return;

				JPopupMenu popup = new JPopupMenu();

				if (layer != 0) {
					JMenuItem thresholdChangeItem = new JMenuItem(
							"Change thershold");
					thresholdChangeItem.setName("change threshold$" + layer
							+ ":" + number);
					thresholdChangeItem
							.addActionListener(NeuralNetworkGUI.this);
					popup.add(thresholdChangeItem);
				}

				JMenu changeWeightSubMenu = new JMenu("Change weight to ...");
				changeWeightSubMenu.addActionListener(NeuralNetworkGUI.this);

				if (layer != ffnn.getLayers().size() - 1) {
					JMenuItem changeWeightItems[] = new JMenuItem[ffnn
							.getLayers().get(layer + 1).getNeuronCount()];
					for (int i = 0; i < changeWeightItems.length; i++) {
						changeWeightItems[i] = new JMenuItem("neuron " + i);
						changeWeightItems[i].setName("change weight$" + layer
								+ ":" + number + "^" + i);
						changeWeightItems[i]
								.addActionListener(NeuralNetworkGUI.this);
						changeWeightSubMenu.add(changeWeightItems[i]);
					}

					popup.add(changeWeightSubMenu);
				}

				popup.show((Component) e.getSource(), e.getX(), e.getY());
			}
		}
	}

	private Point getNeuronPoint(int layer, int number) {
		int layerCount = ffnn.getLayers().size();

		int x_start = Math.max(70, background.getWidth() / (layerCount + 1));
		int x_step = (background.getWidth() - (x_start * 2)) / (layerCount - 1);

		int neuronCount = ffnn.getLayers().get(layer).getNeuronCount();

		int y_start = Math.max(70, background.getHeight() / (neuronCount + 1));
		int y_step = (background.getHeight() - (y_start * 2))
				/ (Math.max(neuronCount - 1, 1));

		return new Point(x_start + layer * x_step, y_start + number * y_step);
	}

	public void drawFFNN() {

		Graphics g = background.getGraphics();
		g.setColor(Color.white);
		g.fillRect(0, 0, background.getWidth(), background.getHeight());

		g.setColor(Color.black);
		g.setFont(thresholdFont);

		for (int i = 0; i < ffnn.getLayers().size(); i++)
			for (int j = 0; j < ffnn.getLayers().get(i).getNeuronCount(); j++) {
				Point p = getNeuronPoint(i, j);
				String threshold = "";
				try {
					threshold = String.format(
							"%." + precision + "f",
							ffnn.getLayers()
									.get(i - 1)
									.getMatrix()
									.get(ffnn.getLayers().get(i - 1)
											.getNeuronCount(), j));
				} catch (Exception e) {
				}
				if (!threshold.equals("")) {
					g.drawString(threshold, (int) (p.x - Math.round(g
							.getFontMetrics().getStringBounds(threshold, g)
							.getWidth() / 2)), (int) (p.y + Math.round(g
							.getFontMetrics().getStringBounds(threshold, g)
							.getHeight() / 4)));
				}
				g.drawOval(p.x - neuronDiameter / 2, p.y - neuronDiameter / 2,
						neuronDiameter, neuronDiameter);
			}

		g.setFont(weightFont);

		for (int i = 0; i < ffnn.getLayers().size() - 1; i++) {
			for (int from = 0; from < ffnn.getLayers().get(i).getNeuronCount(); from++)
				for (int to = 0; to < ffnn.getLayers().get(i + 1)
						.getNeuronCount(); to++) {
					Point fromP = getNeuronPoint(i, from);
					Point toP = getNeuronPoint(i + 1, to);
					double vecX = toP.x - fromP.x;
					double vecY = toP.y - fromP.y;
					double vecL = Math.sqrt(vecX * vecX + vecY * vecY);
					g.drawLine(
							(int) (fromP.x + Math.round((neuronDiameter * (vecX
									/ vecL / 2)))),
							(int) (fromP.y + Math.round(neuronDiameter
									* (vecY / vecL / 2))),
							(int) (toP.x - Math.round(neuronDiameter
									* (vecX / vecL / 2))),
							(int) (toP.y - Math.round(neuronDiameter
									* (vecY / vecL / 2))));
					String weight = String.format("%." + precision + "f", ffnn
							.getLayers().get(i).getMatrix().get(from, to));
					g.drawString(weight,
							(int) (fromP.x + Math
									.round(((neuronDiameter + weightFont
											.getSize()) * (vecX / vecL / 2)))),
							(int) (fromP.y + Math
									.round(((neuronDiameter + weightFont
											.getSize()) * (vecY / vecL / 2)))));
				}
		}

		g.dispose();
		repaint();
	}

	@Override
	public void keyPressed(KeyEvent e) {
	}

	@Override
	public void keyReleased(KeyEvent e) {
		if (e.getKeyCode() == KeyEvent.VK_ENTER)
			actionPerformed(new ActionEvent(commandLineButton, 0, ""));
		if (e.getKeyCode() == KeyEvent.VK_UP && commandLineHistoryPointer > -1
				&& commandLineHistory.size() != 0) {
			commandLineField
					.setText(commandLineHistory
							.get((commandLineHistoryPointer == 0 ? commandLineHistoryPointer
									: commandLineHistoryPointer--)));
		}
		if (e.getKeyCode() == KeyEvent.VK_DOWN
				&& commandLineHistoryPointer > -1
				&& commandLineHistoryPointer < commandLineHistory.size()) {
			commandLineField.setText(commandLineHistory
					.get((commandLineHistoryPointer == commandLineHistory
							.size() - 1 ? commandLineHistoryPointer
							: commandLineHistoryPointer++)));
		}
	}

	@Override
	public void keyTyped(KeyEvent e) {
	}

}
