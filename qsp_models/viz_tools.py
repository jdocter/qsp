# visualization tools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from . import QSPCircuit 

def plot_qsp_response(f, model):
	"""Plot the QSP response againts the desired function response.
	
	Params
	------
	f : function float --> float
		the desired function to be implemented by the QSP sequence
	model : Keras `Model` with `QSP` layer
		model trained to approximate f
	"""
	all_th = np.arange(0, np.pi, np.pi / 300)

	# construct circuit
	phis = model.trainable_weights[0].numpy()
	qsp_circuit = QSPCircuit(phis)
	qsp_circuit.svg()
	circuit_px = qsp_circuit.eval_px(all_th)
	circuit_qx = qsp_circuit.eval_qx(all_th)
	qsp_response = qsp_circuit.qsp_response(all_th)

	df = pd.DataFrame({"x": np.cos(all_th), "Imag[p(x)]": np.imag(circuit_px), 
		"Real[p(x)]": np.real(circuit_px), "Real[q(x)]": np.real(circuit_qx), 
		"desired f": f(np.cos(all_th)), "QSP Response": qsp_response})
	df = df.melt("x", var_name="src", value_name="value")
	sns.lineplot(x="x", y="value", hue="src", data=df).set_title("QSP Response")
	plt.show()

def plot_loss(history):
	"""Plot the error of a trained QSP model. 
		
	Params
	------
	history : tensorflow `History` object
	"""
	plt.plot(history.history['loss'])
	plt.title("Learning QSP Angles")
	plt.xlabel("Iterations")
	plt.ylabel("Error")
	plt.show()