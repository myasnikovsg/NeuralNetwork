package GUI;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.NumberFormat;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFormattedTextField;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class ChangeDialog extends JDialog {

	private static final long serialVersionUID = -206282908807917683L;

	private JFormattedTextField fieldNewValue;
	private NeuralNetworkGUI owner;

	public ChangeDialog(double curValue, final int layer, final int numberFrom,
			final int numberTo, final boolean isChangeThresholdDialog) {
		setLocationRelativeTo(null);
		setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));
		setTitle("Change threshold");

		JPanel newValuePanel = new JPanel();
		newValuePanel.setLayout(new BoxLayout(newValuePanel,
				BoxLayout.LINE_AXIS));
		JLabel textNewValue;
		if (isChangeThresholdDialog)
			textNewValue = new JLabel("New threshold for neuron number "
					+ numberFrom + " in layer " + layer);
		else
			textNewValue = new JLabel("New weight from neuron " + numberFrom
					+ " in layer " + layer + " to neuron number" + numberTo
					+ " in layer " + layer + 1);

		NumberFormat numberFormat = NumberFormat.getInstance();
		numberFormat.setMinimumIntegerDigits(1);
		numberFormat.setMaximumIntegerDigits(2);
		numberFormat.setMinimumFractionDigits(0);
		numberFormat.setMaximumFractionDigits(2);
		fieldNewValue = new JFormattedTextField(numberFormat);
		fieldNewValue.setText(Double.toString(curValue));
		fieldNewValue.setPreferredSize(new Dimension(30, 15));

		newValuePanel.add(Box.createRigidArea(new Dimension(20, 20)));
		newValuePanel.add(textNewValue);
		newValuePanel.add(Box.createRigidArea(new Dimension(50, 20)));
		newValuePanel.add(fieldNewValue);
		newValuePanel.add(Box.createRigidArea(new Dimension(20, 20)));

		JButton okButton = new JButton("OK");
		okButton.addActionListener(new ActionListener() {

			@Override
			public void actionPerformed(ActionEvent arg0) {
				dispose();
				if (isChangeThresholdDialog)
					owner.changeThreshold(layer, numberFrom,
							Double.parseDouble(fieldNewValue.getText()));
				else
					owner.changeWeight(layer, numberFrom, numberTo,
							Double.parseDouble(fieldNewValue.getText()));
			}
		});

		add(newValuePanel);
		add(okButton);
	}

	public void setOwner(NeuralNetworkGUI owner) {
		this.owner = owner;
	}
}
