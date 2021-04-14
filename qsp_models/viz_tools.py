# visualization tools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from qsp_circuit import QSPCircuit 

def plot_qsp_response(f, model):
	"""Plot the QSP response againts the desired function response.
	
	Params
	------
	f : function float --> float
		the desired function to be implemented by the QSP sequence
	model : Keras `Model` with `QSP` layer
		model trained to approximate f
	"""
	all_th = np.arange(0, np.pi, np.pi / 100)

	# construct circuit
	phis = model.trainable_weights[0].numpy()
	qsp_circuit = QSPCircuit(phis)
	qsp_circuit.svg()
	circuit_px = qsp_circuit.eval_px(all_th)

	df = pd.DataFrame({"x": np.cos(all_th), "Imag[p(x)]": np.imag(circuit_px), "Real[p(x)]": np.real(circuit_px), "desired f": f(np.cos(all_th))})
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