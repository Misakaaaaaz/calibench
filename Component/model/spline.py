from .calibrator import Calibrator
from scipy.special import softmax
import numpy as np
import torch


class SplineCalibrator(Calibrator):
    def __init__(self, spline_method: str = "natural", splines: int = 6):
        """
        Initialize the spline calibrator.

        Args:
            spline_method (str):
                The spline boundary/run-out strategy used when fitting the spline recalibration function.
                Options:
                - "natural"
                - "parabolic"
                - "cubic"
            splines (int):
                Number of spline knots (control points) used to fit the recalibration curve.
        """
        super(Calibrator, self).__init__()
        self.spline_method = spline_method
        self.splines = splines
        self.frecal = None

    def fit(self, val_logits, val_labels, **kwargs):
        """
        Fit the spline recalibration function on a validation set.

        Args:
            val_logits (torch.Tensor or numpy.ndarray): Validation logits
            val_labels (torch.Tensor or numpy.ndarray): Validation labels
            **kwargs: Optional overrides for fitting
                - spline_method (str): Overrides self.spline_method for this fit call.
                - splines (int): Overrides self.splines (number of knots) for this fit call.

        Returns:
            A dictionary describing the hyperparameters used during fitting:
                {
                    "spline_method": <str>,
                    "splines": <int>
                }
        """
        spline_method = kwargs.get("spline_method", self.spline_method)
        splines = int(kwargs.get("splines", self.splines))
        val_logits = ensure_numpy(val_logits)
        val_labels = ensure_numpy(val_labels)
        val_probs = softmax(val_logits, 1)
        scores_val, labels_val = get_top_results(val_probs, val_labels, nn=-1)
        self.frecal = get_recalibration_function(scores_val, labels_val, spline_method, splines)
        return {"spline_method": self.spline_method, "splines": self.splines}

    def calibrate(self, test_logits, return_logits=False, **kwargs):
        """
        Apply the fitted spline recalibration function to test logits.

        Args:
            test_logits (torch.Tensor or numpy.ndarray): Test logits
            return_logits (bool): Whether to return logits or probabilities

        Returns:
            Depending on flags:
            - If return_logits is True:
                Return calibrated logits.
            - Else:
                Return calibrated probabilities.

        Raises:
            RuntimeError:
                If fit() has not been called (frecal is None).
        """
        if self.frecal is None:
            raise RuntimeError("SplineCalibrator must be fit() before calibrate().")
        test_logits = ensure_numpy(test_logits)
        test_probs = softmax(test_logits, 1)
        scores_test, top_indices_test = get_top_results_(test_probs, nn=-1, return_topn_classid=True)
        cal_scores = np.array([self.frecal(float(sc)) for sc in scores_test])
        cal_scores = np.clip(cal_scores, 0.0, 1.0)
        probs_test_calibrated = np.copy(test_probs)
        for i, (idx, cal_score) in enumerate(zip(top_indices_test, cal_scores)):
            orig_score = test_probs[i, idx]
            if orig_score > 0:
                scale_factor = (1 - cal_score) / (1 - orig_score) if orig_score < 1 else 0
                probs_test_calibrated[i] *= scale_factor
                probs_test_calibrated[i, idx] = cal_score
        if return_logits:
            return np.log(np.clip(probs_test_calibrated, 1e-8, 1-1e-8))
        else:
            return probs_test_calibrated


class Spline():
    # Initializer
    def __init__(self, x, y, kx, runout='parabolic'):
        # This calculates and initializes the spline
        # Store the values of the knot points
        self.kx = kx
        self.delta = kx[1] - kx[0]
        self.nknots = len(kx)
        self.runout = runout
        # Now, compute the other matrices
        m_from_ky = self.ky_to_M()  # Computes second derivatives from knots
        my_from_ky = np.concatenate([m_from_ky, np.eye(len(kx))], axis=0)
        y_from_my = self.my_to_y(x)
        y_from_ky = y_from_my @ my_from_ky
        # Now find the least squares solution
        ky = np.linalg.lstsq(y_from_ky, y, rcond=-1)[0]
        # Return my
        self.ky = ky
        self.my = my_from_ky @ ky

    def my_to_y(self, vecx):
        # Makes a matrix that computes y from M
        # The matrix will have one row for each value of x
        # Make matrices of the right size
        ndata = len(vecx)
        nknots = self.nknots
        delta = self.delta
        mM = np.zeros((ndata, nknots))
        my = np.zeros((ndata, nknots))
        for i, xx in enumerate(vecx):
            # First work out which knots it falls between
            j = int(np.floor((xx - self.kx[0]) / delta))
            if j >= self.nknots - 1: j = self.nknots - 2
            if j < 0: j = 0
            x = xx - j * delta
            # Fill in the values in the matrices
            mM[i, j] = -x ** 3 / (6.0 * delta) + x ** 2 / 2.0 - 2.0 * delta * x / 6.0
            mM[i, j + 1] = x ** 3 / (6.0 * delta) - delta * x / 6.0
            my[i, j] = -x / delta + 1.0
            my[i, j + 1] = x / delta
        # Now, put them together
        M = np.concatenate([mM, my], axis=1)
        return M

    def my_to_dy(self, vecx):
        # Makes a matrix that computes y from M for a sequence of values x
        # The matrix will have one row for each value of x in vecx
        # Knots are at evenly spaced positions kx
        # Make matrices of the right size
        ndata = len(vecx)
        h = self.delta
        mM = np.zeros((ndata, self.nknots))
        my = np.zeros((ndata, self.nknots))
        for i, xx in enumerate(vecx):
            # First work out which knots it falls between
            j = int(np.floor((xx - self.kx[0]) / h))
            if j >= self.nknots - 1: j = self.nknots - 2
            if j < 0: j = 0
            x = xx - j * h
            mM[i, j] = -x ** 2 / (2.0 * h) + x - 2.0 * h / 6.0
            mM[i, j + 1] = x ** 2 / (2.0 * h) - h / 6.0
            my[i, j] = -1.0 / h
            my[i, j + 1] = 1.0 / h
        # Now, put them together
        M = np.concatenate([mM, my], axis=1)
        return M

    def ky_to_M(self):
        # Make a matrix that computes the
        A = 4.0 * np.eye(self.nknots - 2)
        b = np.zeros(self.nknots - 2)
        for i in range(1, self.nknots - 2):
            A[i - 1, i] = 1.0
            A[i, i - 1] = 1.0
        # For parabolic run-out spline
        if self.runout == 'parabolic':
            A[0, 0] = 5.0
            A[-1, -1] = 5.0
        # For cubic run-out spline
        if self.runout == 'cubic':
            A[0, 0] = 6.0
            A[0, 1] = 0.0
            A[-1, -1] = 6.0
            A[-1, -2] = 0.0
        # The goal
        delta = self.delta
        B = np.zeros((self.nknots - 2, self.nknots))
        for i in range(0, self.nknots - 2):
            B[i, i] = 1.0
            B[i, i + 1] = -2.0
            B[i, i + 2] = 1.0
        B = B * (6 / delta ** 2)
        # Now, solve
        Ainv = np.linalg.inv(A)
        AinvB = Ainv @ B
        # Now, add rows of zeros for M[0] and M[n-1]
        # This depends on the type of spline
        if (self.runout == 'natural'):
            z0 = np.zeros((1, self.nknots))  # for natural spline
            z1 = np.zeros((1, self.nknots))  # for natural spline
        if (self.runout == 'parabolic'):
            # For parabolic runout spline
            z0 = AinvB[0]
            z1 = AinvB[-1]
        if (self.runout == 'cubic'):
            # For cubic runout spline
            # First and last two rows
            z0 = AinvB[0]
            z1 = AinvB[1]
            zm1 = AinvB[-1]
            zm2 = AinvB[-2]
            z0 = 2.0 * z0 - z1
            z1 = 2.0 * zm1 - zm2
        # Reshape to (1, n) matrices
        z0 = z0.reshape((1, -1))
        z1 = z1.reshape((1, -1))
        AinvB = np.concatenate([z0, AinvB, z1], axis=0)
        return AinvB

    def evaluate(self, x):
        # Evaluates the spline at a vector of values
        y = self.my_to_y(x) @ self.my
        return y

    def evaluate_deriv(self, x):
        # Evaluates the spline at a vector (or single) point
        y = self.my_to_dy(x) @ self.my
        return y


class interpolated_function:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lastindex = len0(self.x) - 1
        self.low = self.x[0]
        self.high = self.x[-1]

    def __call__(self, x):
        # Finds the interpolated value of the function at x
        # Easiest thing if value is out of range is to give maximum value
        if x >= self.x[-1]: return self.y[-1]
        if x <= self.x[0]: return self.y[0]
        # Find the first x above.  ind cannot be 0, because of previous test
        # ind cannot be > lastindex, because of last test
        ind = first_above(self.x, x)
        alpha = x - self.x[ind - 1]
        beta = self.x[ind] - x
        # Special case.  This occurs when two values of x are equal
        if alpha + beta == 0:
            return self.y[ind]
        return float((beta * self.y[ind] + alpha * self.y[ind - 1]) / (alpha + beta))


def is_numpy_object(x):
    return type(x).__module__ == np.__name__


def len0(x):
    # Proper len function that REALLY works.
    # It gives the number of indices in first dimension
    # Lists and tuples
    if isinstance(x, list):
        return len(x)
    if isinstance(x, tuple):
        return len(x)
    # Numpy array
    if isinstance(x, np.ndarray):
        return x.shape[0]
    # Other numpy objects have length zero
    if is_numpy_object(x):
        return 0
    # Unindexable objects have length 0
    if x is None:
        return 0
    if isinstance(x, int):
        return 0
    if isinstance(x, float):
        return 0
    # Do not count strings
    if type(x) == type("a"):
        return 0
    return 0


def ensure_numpy(a):
    if torch.is_tensor(a):
        # First move to CPU if tensor is on GPU
        a = a.cpu()
        a = a.numpy()
    elif not isinstance(a, np.ndarray):
        a = np.array(a)
    return a


def first_above(A, val, low=0, high=-1):
    # Find the first time that the array exceeds, or equals val in the range low to high
    # inclusive -- this uses binary search
    # Initialization
    if high == -1: high = len0(A) - 1
    # Stopping point, when interval reduces to one element
    if high == low:
        if val <= A[low]:
            return low
        else:
            # The element does not exist.  This means that there is nowhere
            # in the array where A[k] >= val
            return low + 1  # This will be out-of-bounds if the array never exceeds val
    # Otherwise, we subdivide and continue -- mid must be less then high
    # but can equal low, when high-low = 1
    mid = low + (high - low) // 2
    if A[mid] >= val:
        # In this case, the first time must be in the interval [low, mid]
        return first_above(A, val, low, mid)
    else:
        # In this case, the first time A[k] exceeds val must be to the right
        return first_above(A, val, mid + 1, high)


def get_recalibration_function(scores_in, labels_in, spline_method, splines):
    # Find a function for recalibration
    # Change to numpy
    scores = ensure_numpy(scores_in)
    labels = ensure_numpy(labels_in)
    # Sort the data according to score
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]
    # Accumulate and normalize by dividing by num samples
    nsamples = len0(scores)
    integrated_accuracy = np.cumsum(labels) / nsamples
    integrated_scores = np.cumsum(scores) / nsamples
    percentile = np.linspace(0.0, 1.0, nsamples)
    # Now, try to fit a spline to the accumulated accuracy
    nknots = splines
    kx = np.linspace(0.0, 1.0, nknots)
    spline = Spline(percentile, integrated_accuracy - integrated_scores, kx, runout=spline_method)
    # Evaluate the spline to get the accuracy
    acc = spline.evaluate_deriv(percentile)
    acc += scores
    # Return the interpolating function -- uses full (not decimated) scores and
    # accuracy
    func = interpolated_function(scores, acc)
    return func


def get_top_results(scores, labels, nn, inclusive=False, return_topn_classid=False):
    # Different if we want to take inclusing scores
    if inclusive: return get_top_results_inclusive(scores, labels, nn=nn)
    #  nn should be negative, -1 means top, -2 means second top, etc
    # Get the position of the n-th largest value in each row
    topn = [np.argpartition(score, nn)[nn] for score in scores]
    nthscore = [score[n] for score, n in zip(scores, topn)]
    labs = [1.0 if int(label) == int(n) else 0.0 for label, n in zip(labels, topn)]
    # Change to tensor
    tscores = np.array(nthscore)
    tacc = np.array(labs)
    if return_topn_classid:
        return tscores, tacc, topn
    else:
        return tscores, tacc


def get_top_results_inclusive(scores, labels, nn=-1):
    #  nn should be negative, -1 means top, -2 means second top, etc
    # Order scores in each row, so that nn-th score is in nn-th place
    order = np.argpartition(scores, nn)
    # Reorder the scores accordingly
    top_scores = np.take_along_axis(scores, order, axis=-1)[:, nn:]
    # Get the top nn labels
    top_labels = order[:, nn:]
    # Sum the top scores
    sumscores = np.sum(top_scores, axis=-1)
    # See if label is in the top nn
    labs = np.array([1.0 if int(label) in n else 0.0 for label, n in zip(labels, top_labels)])
    return sumscores, labs


def get_top_results_(scores, nn, inclusive=False, return_topn_classid=False):
    # Different if we want to take inclusing scores
    if inclusive: return get_top_results_inclusive_(scores, nn=nn)
    #  nn should be negative, -1 means top, -2 means second top, etc
    # Get the position of the n-th largest value in each row
    topn = [np.argpartition(score, nn)[nn] for score in scores]
    nthscore = [score[n] for score, n in zip(scores, topn)]
    # Change to tensor
    tscores = np.array(nthscore)
    if return_topn_classid:
        return tscores, topn
    else:
        return tscores


def get_top_results_inclusive_(scores, nn=-1):
    #  nn should be negative, -1 means top, -2 means second top, etc
    # Order scores in each row, so that nn-th score is in nn-th place
    order = np.argpartition(scores, nn)
    # Reorder the scores accordingly
    top_scores = np.take_along_axis(scores, order, axis=-1)[:, nn:]
    # Sum the top scores
    sumscores = np.sum(top_scores, axis=-1)
    return sumscores
