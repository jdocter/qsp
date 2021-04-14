from __future__ import print_function

import matplotlib.pyplot as plt
import mpmath
import seaborn


# NOTE: Created via: seaborn.husl_palette(6)[4]
HUSL_BLUE = (0.23299120924703914, 0.639586552066035, 0.9260706093977744)
EXPECTED_COEFFS = [
    '0x1.5555555555593p-1',
    '0x1.999999997fa04p-2',
    '0x1.2492494229359p-2',
    '0x1.c71c51d8e78afp-3',
    '0x1.7466496cb03dep-3',
    '0x1.39a09d078c69fp-3',
    '0x1.2f112df3e5244p-3',
]
EXPONENTS = (2, 4, 6, 8, 10, 12, 14)
SIZE_INTERVAL = 512
NUM_POINTS = 8
MAX_STEPS = 20
CTX = mpmath.MPContext()
CTX.prec = 200  # Bits vs. default of 53
# NOTE: Slightly larger than 3 - 2 * mpmath.sqrt(2)
#       so that we can capture an 8th equi-oscillating
#       point.
MAX_X = CTX.mpf('0.1717')
INTERVAL_SAMPLE = CTX.linspace(0, MAX_X, SIZE_INTERVAL)


def exp_by_squaring(val, n):
    """Exponentiate by squaring.

    :type val: float
    :param val: A value to exponentiate.

    :type n: int
    :param n: The (positive) exponent.

    :rtype: float
    :returns: The exponentiated value.
    """
    result = 1
    pow_val = val
    while n != 0:
        n, remainder = divmod(n, 2)
        if remainder == 1:
            result *= pow_val
        pow_val = pow_val * pow_val
    return result


def R_scalar(s):
    """Compute the value of R(s).

    R(s) is function such that

           log(1 + s) - log(1 - s) = s(2 + R(s))

    Uses ``mpmath`` to compute the answer in high precision.

    :type s: float
    :param s: A scalar to evaluate ``R(s)`` at.

    :rtype: float
    :returns: The value of ``R(s)`` at our input value.
    """
    if s == 0.0:
        # We can't divide by 0, but we know that
        # R(0) = 2 * 0^2/3 + 2 * 0^4 / 5 + ... = 0.
        return CTX.mpf(0.0)

    numer = CTX.log(1 + s) - CTX.log(1 - s)
    return numer / s - CTX.mpf(2.0)


def update_remez_poly(x_vals):
    """Updates the Remez polynomial coefficients.

    :type x_vals: list
    :param x_vals: The ``x``-values where equi-oscillating occurs.
                   We assume there are ``NUM_POINTS`` values.

    :rtype: tuple
    :returns: A pair, the first is the coefficients (as a list) and the second
              is a scalar (the equi-oscillating error).
    """
    # Columns correspond to an exponent while rows correspond to
    # an x-value.
    coeff_system = CTX.matrix(NUM_POINTS, NUM_POINTS)
    rhs = CTX.matrix(NUM_POINTS, 1)
    for row in range(NUM_POINTS):
        # Handle the final column first (just a sign).
        coeff_system[row, NUM_POINTS - 1] = (-1)**row
        x_val = x_vals[row]
        rhs[row, 0] = R_scalar(x_val)
        # Handle all columns left (final column already done).
        for col in range(NUM_POINTS - 1):
            pow_ = EXPONENTS[col]
            coeff_system[row, col] = exp_by_squaring(x_val, pow_)
    soln = CTX.lu_solve(coeff_system, rhs, real=True)
    # Turn into a row vector and then turn it into a list.
    soln, = soln.T.tolist()
    return soln[:-1], soln[-1]


def get_chebyshev_points(num_points):
    """Get Chebyshev points for [0, MAX_X].

    :type num_points: int
    :param num_points: The number of points to use.

    :rtype: list
    :returns: The Chebyshev points for our interval.
    """
    result = []
    for index in range(2 * num_points - 1, 0, -2):
        theta = CTX.pi * index / CTX.mpf(2.0 * num_points)
        result.append(0.5 * MAX_X * (1 + CTX.cos(theta)))
    return result


class ErrorFunc(object):
    """Error function for representing R(s) and P(s).

    Takes a given set of ``x``-values, computes the necessary
    coefficients of ``P(s)`` with them and uses them to define
    the function R(s) - P(s).

    :type x_vals: list
    :param x_vals: A 1D array. The ``x``-values where
                   equi-oscillating occurs.
    """

    def __init__(self, x_vals):
        self.x_vals = x_vals
        # Computed values.
        self.poly_coeffs, self.E = update_remez_poly(x_vals)

    def poly_approx_scalar(self, value):
        """Evaluate the polynomial :math:`f(x)`.

        Uses the same method as in ``math/log.go`` to compute

        .. math::

           L_1 x^2 + L_2 x^4 + \\cdots + L_7 x^{14}

        :type value: float
        :param value: The value to compute ``f(x)`` at.

        :rtype: float
        :returns: The value of ``f(x)`` at our input value.
        """
        L1 = self.poly_coeffs[0]
        L2 = self.poly_coeffs[1]
        L3 = self.poly_coeffs[2]
        L4 = self.poly_coeffs[3]
        L5 = self.poly_coeffs[4]
        L6 = self.poly_coeffs[5]
        L7 = self.poly_coeffs[6]
        s2 = value * value
        s4 = s2 * s2
        t1 = s2 * (L1 + s4 * (L3 + s4 * (L5 + s4 * L7)))
        t2 = s4 * (L2 + s4 * (L4 + s4 * L6))
        return t1 + t2

    def signed_error_scalar(self, value):
        """Error function R(s) - P(s).

        :type value: float
        :param value: An ``x``-value.

        :rtype: float
        :returns: The value of the error R(s) - P(s).
        """
        return R_scalar(value) - self.poly_approx_scalar(value)

    def signed_error_diff(self, value):
        """Derivative of error function R'(s) - P'(s).

        :type value: float
        :param value: An ``x``-value.

        :rtype: float
        :returns: The value of the error R'(s) - P'(s).
        """
        return CTX.diff(self.signed_error_scalar, value)

    def signed_error(self, values):
        """Error function R(s) - P(s).

        :type values: list
        :param values: A list of ``x``-values.

        :rtype: list
        :returns: The value of the error R(s) - P(s) at each point.
        """
        result = []
        for val in values:
            result.append(self.signed_error_scalar(val))
        return result


def locate_abs_max(values):
    """Locate the absolute maximum of a list of values.

    :type values: list
    :param values: A list of scalar values.

    :rtype: int
    :returns: The index where the maximum occurs.
    """
    curr_max = -CTX.inf
    curr_max_index = -1
    for index, value in enumerate(values):
        abs_val = abs(value)
        if abs_val > curr_max:
            curr_max = abs_val
            curr_max_index = index
    return curr_max_index


def get_peaks(x_data, y_data, num_peaks):
    """Get the peaks from oscillating output data.

    :type x_data: list
    :param x_data: The ``x``-values where the outputs occur.

    :type y_data: list
    :param y_data: The oscillating output data.

    :type num_peaks: int
    :param num_peaks: The number of peaks to locate.

    :rtype: list
    :returns: The ``x``-locations of all the peaks that were found,
              in the order that they were found.
    """
    local_data = y_data[::]
    size_interval = len(y_data)
    peak_locations = []
    while len(peak_locations) < num_peaks:
        curr_biggest = locate_abs_max(local_data)
        x_now = x_data[curr_biggest]
        if x_now in peak_locations:
            raise ValueError('Repeat value found.')
        peak_locations.append(x_now)
        # Find the sign so we can identify all nearby points on the
        # peak (they will have the same sign).
        sign_x_now = CTX.sign(local_data[curr_biggest])
        local_data[curr_biggest] = 0.0
        # Zero out all the values on the same peak to the
        # right of x_now.
        index = curr_biggest + 1
        while (index < size_interval and
               CTX.sign(local_data[index]) == sign_x_now):
            local_data[index] = 0.0
            index += 1
        # Zero out all the values on the same peak to the
        # left of x_now.
        index = curr_biggest - 1
        while (index >= 0 and
               CTX.sign(local_data[index]) == sign_x_now):
            local_data[index] = 0.0
            index -= 1

    return peak_locations


def get_new_x_vals(x_vals, sample_points=INTERVAL_SAMPLE):
    """Perform single pass of Remez algorithm.

    First computes the coefficients based on ``x_vals``, then locates
    the extrema of ``|P(s) - R(s)|``.

    :type x_vals: list
    :param x_vals: The ``x``-values where equi-oscillating occurs.

    :type sample_points: list
    :param sample_points: (Optional) The points we choose extrema from.

    :rtype: list
    :returns: The new ``x``-values.
    """
    err_func = ErrorFunc(x_vals)
    size_interval = len(sample_points)

    approx_outputs = err_func.signed_error(sample_points)
    max_vals = get_peaks(sample_points, approx_outputs, NUM_POINTS)
    max_vals = sorted(max_vals)
    # Move from the fixed grid (given by ``sample_points``) onto the entire
    # interval by finding critical points nearby. We **don't** do this
    # for the biggest max value since the function has no critical point on
    # the outside.
    new_vals = []
    for val in max_vals[:-1]:
        new_vals.append(CTX.findroot(err_func.signed_error_diff, val))
    new_vals.append(max_vals[-1])
    # NOTE: We could / should check that new_vals == sorted(new_vals)
    #       and that it has no repeats.
    return new_vals


def plot_x_vals(x_vals, sample_points, filename=None):
    """Plot the error R(s) - P(s) for P(s) given by ``x``-values.

    Also plots the locations of each equi-oscillating ``x``-value on
    the curve.

    :type x_vals: list
    :param x_vals: The ``x``-values where equi-oscillating occurs.

    :type sample_points: list
    :param sample_points: The points we choose extrema from.

    :type filename: str
    :param filename: (Optional) The filename to save the plot in. If
                     not specified, just shows the plot.
    """
    err_func = ErrorFunc(x_vals)
    approx_outputs = err_func.signed_error(sample_points)
    plt.plot(sample_points, approx_outputs, color=HUSL_BLUE)
    plt.plot(x_vals, err_func.signed_error(x_vals), marker='o',
             color='black', linestyle='None')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
        print('Saved ' + filename)


def _list_delta_norm(list1, list2):
    return CTX.norm(CTX.matrix(list1) - CTX.matrix(list2), p=2)


def _lead_substr_match(val1, val2):
    min_len = min(len(val1), len(val2))
    for index in range(1, min_len + 1):
        if val1[:index] != val2[:index]:
            return index - 1
    return min_len


def _print_double_hex_compare(actual_mpf, expected):
    # Convert back to double, then to hex.
    actual_as_hex = float(str(actual_mpf)).hex()
    print('- ' + actual_as_hex)
    expected_begin, expected_expon = expected.split('p', 1)
    match_index = _lead_substr_match(actual_as_hex,
                                     expected_begin)
    buffer = '.' * (len(expected_begin) - match_index)
    print('  ' + expected_begin[:match_index] +
          buffer + 'p' + expected_expon)


def main(sample_points=INTERVAL_SAMPLE,
         threshold=CTX.mpf(2)**(-26), plot_all=False):
    """Run the Remez algorithm until termination.

    :type sample_points: list
    :param sample_points: (Optional) The points we choose extrema from.

    :type threshold: float
    :param threshold: (Optional) The minimum value of the norm of the
                      difference between current ``x``-values and the
                      next set of ``x``-values found via the Remez
                      algorithm. Defaults to ``2^{-26}``.

    :type plot_all: bool
    :param plot_all: (Optional) Flag indicating all updates should be
                     plotted. Defaults to :data:`False`.
    """
    prev_x_vals = [CTX.inf] * NUM_POINTS
    x_vals = get_chebyshev_points(NUM_POINTS)
    num_steps = 0
    while _list_delta_norm(x_vals, prev_x_vals) > threshold:
        if num_steps >= MAX_STEPS:
            print('Max. steps encountered. Does not converge.')
            break
        if plot_all:
            plot_x_vals(x_vals, sample_points)
        prev_x_vals = x_vals
        x_vals = get_new_x_vals(x_vals, sample_points=sample_points)
        num_steps += 1

    norm_update = _list_delta_norm(x_vals, prev_x_vals)
    msg = 'Difference between successive x-vectors: %g' % (norm_update,)
    print(msg)
    msg = 'Completed in %d steps' % (num_steps,)
    print(msg)
    err_func = ErrorFunc(x_vals)
    print('Coefficients:')
    for index, coeff in enumerate(err_func.poly_coeffs):
        _print_double_hex_compare(coeff, EXPECTED_COEFFS[index])
    plot_x_vals(x_vals, sample_points, filename='remez_approx.png')


if __name__ == '__main__':
    main()
