# Learning Quantum Signal Processing Angles 

## Examples
----------------------

`qsp_models` is a package intended to learn the QSP angles for a desired response function. 

### Classes

`qsp_model.QSP` is a arameterized quantum signal processing layer. Parameterized by the $\phi$ angles.  

`qsp_model.QSPCircuit` is a `cirq.Circuit` that implements the QSP sequence given by `phis`. It is a tool to evaluate and visualize the response of a given QSP sequence, by supporting substitution of arbitrary theta into the sequence.

### Functions

`qsp_model.construct_qsp_model` is a helper function that compiles a QSP model with mean squared error and adam optimizer.

`qsp_model.plot_loss` is a helper function to plot the QSP response againts the desired function response.
	
`qsp_model.plot_qsp_response` is a helper function to plot the error of a trained QSP model 
  
