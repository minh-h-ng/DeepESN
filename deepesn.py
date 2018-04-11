import numpy as np

from scipy import sparse
from scipy.sparse import linalg as slinalg

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression, BayesianRidge

def NRMSE(y_true, y_pred, scaler):
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)

    #Normalized Root Mean Squared Error
    y_std = np.std(y_true)

    return np.sqrt(mean_squared_error(y_true, y_pred))/y_std

class ESN(object):
    def __init__(self, n_internal_units = 100, spectral_radius = 0.9, connectivity = 0.5, input_scaling = 0.5, input_shift = 0.0,
                 feedback_scaling = 0.01, noise_level = 0.01):
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._spectral_radius = spectral_radius
        self._connectivity = connectivity

        self._input_scaling = input_scaling
        self._input_shift = input_shift
        self._feedback_scaling = feedback_scaling
        self._noise_level = noise_level

        # The weights will be set later, when data is provided
        self._input_weights = None
        self._feedback_weights = None

        # Generate internal weights
        self._internal_weights = self._initialize_internal_weights(n_internal_units, connectivity, spectral_radius)

        # Initialize Reservoir state
        self._reservoir_state = np.zeros((1, self._n_internal_units), dtype=float)

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

class DeepESN(object):
    def __init__(self, esn_list, teacher_scaling = 0.5, teacher_shift = 0.0):
        # Initialize list of esn
        self._esn_list = esn_list

        # Initialize attributes
        self._teacher_scaling = teacher_scaling
        self._teacher_shift = teacher_shift
        self._dim_output = None

        # Regression method
        # Initialized to None for now. Will be set during 'fit'.
        self._regression_method = None

    def fit(self, Xtr, Ytr, n_drop = 100, regression_method = 'linear', regression_parameters = None):
        _,_ = self._fit_transform(Xtr = Xtr, Ytr = Ytr,
                                  n_drop = n_drop,
                                  regression_method = regression_method,
                                  regression_parameters = regression_parameters)

        return

    def _fit_transform(self, Xtr, Ytr, n_drop = 100, regression_method = 'linear', regression_parameters = None):
        n_data, dim_data = Xtr.shape
        _, dim_output = Ytr.shape

        self._dim_output = dim_output

        # If this is the first time the network is tuned, set the input and feedback weights.
        # The weights are dense and uniformly distributed in [-1.0, 1.0]
        if (self._esn_list[0]._input_weights is None):
            for i in range(len(self._esn_list)):
                curESN = self._esn_list[i]
                if i==0:
                    curESN._input_weights = 2.0*np.random.rand(curESN._n_internal_units, dim_data) - 1.0
                else:
                    prevESN = self._esn_list[i-1]
                    curESN._input_weights = 2.0*np.random.rand(curESN._n_internal_units, prevESN._n_internal_units) - 1.0

        if (self._esn_list[0]._feedback_weights is None):
            for i in range(len(self._esn_list)):
                curESN = self._esn_list[i]
                if i==len(self._esn_list)-1:
                    curESN._feedback_weights = 2.0*np.random.rand(curESN._n_internal_units, dim_output) - 1.0
                else:
                    nextESN = self._esn_list[i+1]
                    curESN._feedback_weights = 2.0*np.random.rand(curESN._n_internal_units, nextESN._n_internal_units) - 1.0

        # Initialize regression method
        if (regression_method == 'linear'):
            # Use canonical linear regression
            self._regression_method = LinearRegression()

        elif (regression_method == 'ridge'):
            # Ridge regression
            self._regression_method = Ridge(alpha=regression_parameters[0])

        else:
            # Error: unknown regression method
            print('Unknown regression method',regression_method)

        # Calculate states/embedded states.
        states, embedded_states, _ = self._compute_state_matrix(X=Xtr, Y=Ytr, n_drop=n_drop)

        # Train output
        firstESN = self._esn_list[0]
        self._regression_method.fit(np.concatenate(
            (embedded_states, self._scaleshift(Xtr[n_drop:, :], firstESN._input_scaling, firstESN._input_shift)), axis=1),
            self._scaleshift(Ytr[n_drop:, :], self._teacher_scaling, self._teacher_shift).flatten())

        return states, embedded_states

    def predict(self, X, Y = None, n_drop = 100, error_function = NRMSE, scaler = None):
        Yhat, error, _, _ = self._predict_transform(X = X, Y = Y, n_drop = n_drop, error_function = error_function, scaler = scaler)

        return Yhat, error

    def _predict_transform(self, X, Y = None, n_drop = 100, error_function = NRMSE, scaler = None):
        # Predict outputs
        states,embedded_states,Yhat = self._compute_state_matrix(X = X, n_drop = n_drop)

        # Revert scale and shift
        Yhat = self._uscaleshift(Yhat, self._teacher_scaling, self._teacher_shift)

        # Compute error if ground truth is provided
        if (Y is not None):
            error = error_function(Y[n_drop:,:], Yhat, scaler)

        return Yhat, error, states, embedded_states

    def _compute_state_matrix(self, X, Y = None, n_drop = 100):
        n_data, _ = X.shape

        # Initial output value
        previous_output = np.zeros((1, self._dim_output), dtype=float)

        # Storage
        lastESN = self._esn_list[-1]
        state_matrix = np.empty((n_data - n_drop, lastESN._n_internal_units), dtype=float)
        outputs = np.empty((n_data - n_drop, self._dim_output), dtype=float)

        # No embedding yet
        embedded_states = np.empty((n_data - n_drop, lastESN._n_internal_units), dtype=float)

        for i in range(n_data):
            for j in range(len(self._esn_list)):
                curESN = self._esn_list[j]
                firstESN = self._esn_list[0]
                data_input = np.atleast_2d(self._scaleshift(X[i, :], firstESN._input_scaling, firstESN._input_shift))

                # Process inputs
                curESN._reservoir_state = np.atleast_2d(curESN._reservoir_state)
                if len(self._esn_list)==1:
                    current_input = np.atleast_2d(self._scaleshift(X[i, :], curESN._input_scaling, curESN._input_shift))
                    feedback = curESN._feedback_scaling * np.atleast_2d(previous_output)
                elif j==len(self._esn_list)-1:
                    prevESN = self._esn_list[j-1]
                    current_input = np.atleast_2d(self._scaleshift(prevESN._reservoir_state, curESN._input_scaling, curESN._input_shift))
                    feedback = curESN._feedback_scaling * np.atleast_2d(previous_output)
                elif j==0:
                    nextESN = self._esn_list[j+1]
                    current_input = np.atleast_2d(self._scaleshift(X[i, :], curESN._input_scaling, curESN._input_shift))
                    feedback = curESN._feedback_scaling * np.atleast_2d(nextESN._reservoir_state)
                else:
                    prevESN = self._esn_list[j-1]
                    nextESN = self._esn_list[j+1]
                    current_input = np.atleast_2d(self._scaleshift(prevESN._reservoir_state, curESN._input_scaling, curESN._input_shift))
                    feedback = curESN._feedback_scaling * np.atleast_2d(nextESN._reservoir_state)

                # Calculate state. Add noise and apply nonlinearity
                """print('internal weights:',curESN._internal_weights.shape)
                print('reservoir:',curESN._reservoir_state.T.shape)
                print('input weights:',curESN._input_weights.shape)
                print('current input:',current_input.T.shape)
                print('feedback weights:',curESN._feedback_weights.shape)
                print('feedback:',feedback.T.shape)"""
                state_before_tanh = curESN._internal_weights.dot(curESN._reservoir_state.T) + curESN._input_weights.dot(current_input.T) + \
                                    curESN._feedback_weights.dot(feedback.T)
                state_before_tanh += np.random.rand(curESN._n_internal_units, 1) * curESN._noise_level
                curESN._reservoir_state = np.tanh(state_before_tanh).T

            # Embed data and perform regression if applicable
            if (Y is not None):
                # If we are training, the previous output should be a scaled and shifted version of the ground truth.
                previous_output = self._scaleshift(Y[i, :], self._teacher_scaling, self._teacher_shift)
            else:
                lastESN = self._esn_list[-1]

                # No embedding yet
                current_embedding = lastESN._reservoir_state

                # Perform regression
                previous_output = self._regression_method.predict(np.concatenate((current_embedding,data_input),axis=1))

            # Store everything after the dropout period
            if (i > n_drop - 1):
                state_matrix[i-n_drop,:] = lastESN._reservoir_state.flatten()

                # Only save embedding for test data.
                # In training, we do it after computing the whole state matrix.
                if (Y is None):
                    embedded_states[i-n_drop,:] = current_embedding.flatten()

                outputs[i-n_drop,:] = previous_output.flatten()

        # Now, embed the data if we are in training (currently no embedding yet)
        if (Y is not None):
            embedded_states = state_matrix

        return state_matrix, embedded_states, outputs

    def _scaleshift(self, x, scale, shift):
        # Scales and shifts x by scale and shift
        return (x*scale + shift)

    def _uscaleshift(self, x, scale, shift):
        # Reverts the scale and shift applied by _scaleshift
        return ((x-shift)/float(scale))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_from_config(Xtr, Ytr, Xte, Yte, config, scaler):
    # Instantiate ESN object
    esn_list = []
    for i in range(len(config['n_internal_units'])):
        esn = ESN(n_internal_units = config['n_internal_units'][i],
                  spectral_radius = config['spectral_radius'][i],
                  connectivity = config['connectivity'][i],
                  input_scaling = config['input_scaling'][i],
                  input_shift = config['input_shift'][i],
                  feedback_scaling = config['feedback_scaling'][i],
                  noise_level = config['noise_level'][i])
        esn_list.append(esn)

    deepesn = DeepESN(esn_list,
                      teacher_scaling = config['teacher_scaling'],
                      teacher_shift = config['teacher_shift'])

    #Get parameters
    n_drop = config['n_drop']
    regression_method = config['regression_method']
    regression_parameters = config['regression_parameters']

    # Fit and predict
    deepesn.fit(Xtr, Ytr, n_drop=n_drop, regression_method=regression_method, regression_parameters=regression_parameters)

    Yhat, error = deepesn.predict(Xte, Yte, scaler=scaler)

    return Yhat, error

def generate_datasets(X, Y, test_percent = 0.5, val_percent = 0.2, scaler = StandardScaler):
    n_data,_ = X.shape

    n_te = np.ceil(test_percent*n_data).astype(int)
    n_val = np.ceil(val_percent*n_data).astype(int)
    #n_te = 491
    #n_val = 400

    n_tr = n_data - n_te - n_val

    # Split dataset
    Xtr = X[:n_tr, :]
    Ytr = Y[:n_tr, :]

    Xval = X[n_tr:-n_te, :]
    Yval = Y[n_tr:-n_te, :]

    Xte = X[-n_te:, :]
    Yte = Y[-n_te:, :]

    # Scale
    Xscaler = scaler()
    Yscaler = scaler()

    # Fit scaler on training set
    Xtr = Xscaler.fit_transform(Xtr)
    Ytr = Yscaler.fit_transform(Ytr)

    # Transform the rest
    Xval = Xscaler.transform(Xval)
    Yval = Yscaler.transform(Yval)

    Xte = Xscaler.transform(Xte)
    Yte = Yscaler.transform(Yte)

    return Xtr, Ytr, Xval, Yval, Xte, Yte, Yscaler

def load_from_text(path):
    data = np.loadtxt(path, delimiter=',')

    return np.atleast_2d(data[:, :-1]), np.atleast_2d(data[:, -1]).T

def reconstruct_input(arrays,reconstructconfig):
    reconstructDim = [reconstructconfig['reconstruct_dim_x'],reconstructconfig['reconstruct_dim_y'],
                      reconstructconfig['reconstruct_dim_z']]
    reconstructDelay = [reconstructconfig['reconstruct_delay_x'], reconstructconfig['reconstruct_delay_y'],
                      reconstructconfig['reconstruct_delay_z']]
    startIndex = 0
    for i in range(len(reconstructDim)):
        if (reconstructDim[i]-1)*reconstructDelay[i]>startIndex:
            startIndex = (reconstructDim[i]-1)*reconstructDelay[i]
    returnVals = []
    for array in arrays:
        reconstructed = None
        dataDim = array.shape
        for i in range(startIndex, dataDim[0]):
            #curIndex = i - startIndex
            construct = None
            for j in range(dataDim[1]):
                subSeries = array[range(i, i-reconstructDelay[j]*(reconstructDim[j]-1)-1,
                                        -reconstructDelay[j]),j]
                if construct is None:
                    construct = subSeries
                else:
                    construct = np.concatenate([construct, subSeries])
            if reconstructed is None:
                reconstructed = construct
            else:
                reconstructed = np.vstack([reconstructed, construct])
        returnVals.append(reconstructed)
    return returnVals

def reconstruct_output(arrays,reconstructconfig):
    reconstructDim = [reconstructconfig['reconstruct_dim_x'], reconstructconfig['reconstruct_dim_y'],
                      reconstructconfig['reconstruct_dim_z']]
    reconstructDelay = [reconstructconfig['reconstruct_delay_x'], reconstructconfig['reconstruct_delay_y'],
                        reconstructconfig['reconstruct_delay_z']]
    startIndex = 0
    for i in range(len(reconstructDim)):
        if (reconstructDim[i] - 1) * reconstructDelay[i] > startIndex:
            startIndex = (reconstructDim[i] - 1) * reconstructDelay[i]
    returnVals = []
    for array in arrays:
        dataDim = array.shape
        reconstructed = None
        for i in range(startIndex, dataDim[0]):
            subSeries = array[i, dataDim[1] - 1]
            if reconstructed is None:
                reconstructed = subSeries
            else:
                reconstructed = np.vstack([reconstructed, subSeries])
        returnVals.append(reconstructed)
    return returnVals

def reconstruct_input_santafe(arrays,reconstructconfig):
    reconstructDim = reconstructconfig['reconstruct_dim_x']
    reconstructDelay = reconstructconfig['reconstruct_delay_x']
    startIndex = 0
    for i in range(reconstructDim):
        if (reconstructDim - 1) * reconstructDelay > startIndex:
            startIndex = (reconstructDim - 1) * reconstructDelay
    returnVals = []
    for array in arrays:
        reconstructed = None
        dataDim = array.shape
        for i in range(startIndex, dataDim[0]):
            subSeries = array[range(i, i - reconstructDelay * (reconstructDim - 1) - 1,
                                    -reconstructDelay), 0]
            if reconstructed is None:
                reconstructed = subSeries
            else:
                reconstructed = np.vstack([reconstructed, subSeries])
        returnVals.append(reconstructed)
    return returnVals

def reconstruct_output_santafe(arrays,reconstructconfig):
    reconstructDim = [reconstructconfig['reconstruct_dim_x']]
    reconstructDelay = [reconstructconfig['reconstruct_delay_x']]
    startIndex = 0
    for i in range(len(reconstructDim)):
        if (reconstructDim[i] - 1) * reconstructDelay[i] > startIndex:
            startIndex = (reconstructDim[i] - 1) * reconstructDelay[i]
    returnVals = []
    for array in arrays:
        dataDim = array.shape
        reconstructed = None
        for i in range(startIndex, dataDim[0]):
            subSeries = array[i, dataDim[1] - 1]
            if reconstructed is None:
                reconstructed = subSeries
            else:
                reconstructed = np.vstack([reconstructed, subSeries])
        returnVals.append(reconstructed)
    return returnVals

if __name__ == "__main__":
    pass