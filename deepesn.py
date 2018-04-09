import numpy as np

from scipy import sparse
from scipy.sparse import linalg as slinalg

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression, BayesianRidge

def NRMSE(y_true, y_pred, scaler):
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)

    #Normalized Root Mean Squared Error
    y_std = np.std(y_true)

    return np.sqrt(mean_squared_error(y_true, y_pred))/y_std

class ESN(object):
    def __init__(self, n_internal_units = 100, spectral_radius = 0.9, connectivity = 0.5, input_scaling = 0.5, input_shift = 0.0,
                 teacher_scaling = 0.5, teacher_shift = 0.0, feedback_scaling = 0.01, noise_level = 0.01):
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._spectral_radius = spectral_radius
        self._connectivity = connectivity

        self._input_scaling = input_scaling
        self._input_shift = input_shift
        self._teacher_scaling = teacher_scaling
        self._teacher_shift = teacher_shift
        self._feedback_scaling = feedback_scaling
        self._noise_level = noise_level
        self._dim_output = None

        # The weights will be set later, when data is provided
        self._input_weights = None
        self._feedback_weights = None

        # Generate internal weights
        self._internal_weights = self._initialize_internal_weights(n_internal_units, connectivity, spectral_radius)

class DeepESN(object):
    def __init__(self, esn_list):
        #Initialize list of esn
        self._esn_list = esn_list

        # Regression method
        # Initialized to None for now. Will be set during 'fit'.
        self._regression_method = None

    def fit(self, Xtr, Ytr, n_drop = 100, regression_method = 'linear', regression_parameters = None, embedding = 'identity', n_dim = 3):
        _,_ = self._fit_transform(Xtr = Xtr, Ytr = Ytr, n_drop = n_drop, regression_method = regression_method,
                                  regression_parameters = regression_parameters)

        return

    def _fit_transform(self, Xtr, Ytr, n_drop = 100, regression_method = 'linear', regression_parameters = None):
        n_data, dim_data = Xtr.shape
        _, dim_output = Ytr.shape

        self._dim_output = dim_output

        # If this is the first time the network is tuned, set the input and feedback weights.
        # The weights are dense and uniformly distributed in [-1.0, 1.0]
        if (self._input_weights is None):
            self._input_weights = 2.0*np.random.rand(self._n_internal_units, dim_data) - 1.0

        if (self._feedback_weights is None):
            self._feedback_weights = 2.0*np.random.rand(self._n_internal_units, dim_output) - 1.0

        # Initialize regression method
        if (regression_method == 'linear'):
            # Use canonical linear regression
            self._regression_method = LinearRegression()

        else:
            # Error: unknown regression method
            print('Unknown regression method',regression_method)

            # Calculate states/embedded states.
            states, embedded_states, _ = self._compute_state_matrix(X=Xtr, Y=Ytr, n_drop=n_drop)

            # Train output
            self._regression_method.fit(np.concatenate(
                (embedded_states, self._scaleshift(Xtr[n_drop:, :], self._input_scaling, self._input_shift)),axis=1),
                self._scaleshift(Ytr[n_drop:, :], self._teacher_scaling,self._teacher_shift).flatten())

            return states, embedded_states

    def _scaleshift(self, x, scale, shift):
        # Scales and shifts x by scale and shift
        return (x*scale + shift)

    def _uscaleshift(self, x, scale, shift):
        # Reverts the scale and shift applied by _scaleshift
        return ((x-shift)/float(scale))

    def _initialize_internal_weights(self, n_internal_units, connectivity, spectral_radius):
        # The eigs function might not converge. Attempt until it does.
        convergence = False
        while (not convergence):
            # Generate sparse, uniformly distributed weights.
            internal_weights = sparse.rand(n_internal_units, n_internal_units, density=connectivity).todense()

            # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
            internal_weights[np.where(internal_weights > 0)] -= 0.5

            try:
                # Get the largest eigenvalue
                w,_ = slinalg.eigs(internal_weights, k=1, which='LM')

                convergence = True

            except:
                continue

        # Adjust the spectral radius.
        internal_weights /= np.abs(w)/spectral_radius

        return internal_weights

if __name__ == "__main__":
    pass