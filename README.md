# Quantum Signal Processing Angle Learning via Gradient Descent

[`qsp_layers.py`](https://github.com/jdocter/qsp/blob/main/qsp_layers.py)

contains class `HybridControlledPQC` (Hybrid Controlled Parameterized Quantum Circuit)

This is a keras layer for a controlled parameterized quantum circuit similar to [this](https://www.tensorflow.org/quantum/api_docs/python/tfq/layers/ControlledPQC). 

This class is termed "hybrid" because it allows for native trainable parameters within the model, in addition to controlled parameters. Controlled parameters are those that are fed into the circuit model as inputs. In the case of QSP, the native trainable parameters are the Z rotation arguments/angles. The controlled parameter is the unknown angle theta or equivalently x = cos(theta).


[`qsp_angle_estimation.ipynb`](https://github.com/jdocter/qsp/blob/main/qsp_angle_estimation.ipynb)

Preliminary testing of QSP angle estimation in settings of noiseless X rotations. Results for both discrete and continuos. 


[`noisy_qsp_angle_estimation.ipynb`](https://github.com/jdocter/qsp/blob/main/noisy_qsp_angle_estimation.ipynb)

Preliminary support for QSP angle learning with noise.
