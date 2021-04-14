from cirq.contrib.svg import SVGCircuit
import cirq
import sympy
import numpy as np

class QSPCircuit(cirq.Circuit):
    """QSP circuit

    A `cirq.Circuit` that implements the QSP sequence given by `phis`  
    
    A tool to evaluate and visualize the response of a given QSP sequence.

    Allows substitution of arbitrary theta into the sequence
    """
    def __init__(self, phis):
        super(QSPCircuit, self).__init__()
        # recall that in the QSP sequence we rotate as exp(i * phi * Z), but rz(theta) := exp(i * theta/2 * Z)
        self.phis = np.array(phis).flatten() * (-2)
        self.theta = sympy.Symbol("theta")
        self.q = cirq.GridQubit(0, 0)
        self._build_qsp_sequence(self.q)

    def _build_qsp_sequence(self, q):
        self.append(cirq.Circuit(cirq.rz(self.phis[0])(q)))
        for phi in self.phis[1:]:
            c = cirq.Circuit(cirq.rx(self.theta)(q), cirq.rz(phi)(q))
            self.append(c)

    def svg(self):
        """Get the SVG circuit (for visualization)"""
        return SVGCircuit(self)

    def eval_px(self, thetas):
        """Evaluate the QSP response for a list of thetas

        params
        -----
        thetas: list of floats
            list of theta input of a QSP sequence

        returns
        -------
        numpy array with shape (len(params),)
            evaluates the qsp response P(x) for each theta in thetas
        """
        real_pxs = []
        for theta in np.array(thetas).flatten():
            resolver = cirq.ParamResolver({"theta" : theta * (-2)})
            u = cirq.resolve_parameters(self, resolver).unitary()
            real_pxs.append(u[0,0])
        return np.array(real_pxs)

    def eval_real_px(self, thetas):
        """Evaluate the QSP response (real part) for a list of thetas"""
        return np.real(self.eval_px(thetas))

    def eval_imag_px(self, thetas):
        """Evaluate the QSP response (imaginary part) for a list of thetas"""
        return np.imag(self.eval_px(thetas))