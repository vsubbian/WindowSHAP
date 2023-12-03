import warnings

import numpy as np
import shap

warnings.filterwarnings('ignore')


class StationaryWindowSHAP():
    '''
    A class for computing the shapely values for time sereis data. Only the shap values for the first output
    is reported.
    
    Parameters:
    model: A model object that will be used for prediction. The model object must have a method called predict() which produces the model output for a given input
    window_len: The length of the window for the algorithm
    B_ts: A 3D numpy array of background time series data
    test_ts: A 3D numpy array of test time series data
    B_mask: A 3D numpy array of background masking data. It is only used for specific models such as GRUD where a masking variable is passed to the model alongside the time series data. (default: None)
    B_dem: A 2D numpy array of background demographic data (non-temporal data). It is only used for specific models with both modelities of temporal and non-temporal variables. (default: None)
    test_mask: A 3D numpy array of test mask data (default: None)
    test_dem: A 2D numpy array of test demographic data (default: None)
    model_type: The type of model being used. Set the parameter to 'lstm' when time series data is the only input, pick 'lstm_dem' when input includes both time sereis and demographic (non-termporal) data, and 'grud' when you are using GRUD structure.  (default: 'lstm')
    '''

    def __init__(self, model, window_len, B_ts, test_ts, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='lstm'):
        self.model = model
        self.window_len = window_len
        self.num_window = np.ceil(B_ts.shape[1] / self.window_len).astype('int')
        self.num_background = len(B_ts)
        self.num_test = len(test_ts)
        self.background_ts = B_ts
        self.background_mask = B_mask
        self.background_dem = B_dem
        self.test_ts = test_ts
        self.test_mask = test_mask
        self.test_dem = test_dem
        self.model_type = model_type
        self.ts_phi = None
        self.dem_phi = None
        self.explainer = None

        # Problem sizes
        self.num_ts_ftr = B_ts.shape[2]
        self.num_ts_step = B_ts.shape[1]
        self.num_dem_ftr = 0 if B_dem is None else B_dem.shape[1]

        # Creating all data (background and test together)
        self.all_ts = np.concatenate((self.background_ts, self.test_ts), axis=0)
        self.all_mask = None if test_mask is None else np.concatenate(
            (self.background_mask, self.test_mask), axis=0)
        self.all_dem = None if test_dem is None else np.concatenate(
            (self.background_dem, self.test_dem), axis=0)

        # Creating converted data for SHAP
        self.background_data = self.data_prepare(ts_x=self.background_ts, dem_x=self.background_dem,
                                                 start_idx=0)
        self.test_data = self.data_prepare(ts_x=self.test_ts, dem_x=self.test_dem,
                                           start_idx=self.num_background)

    def data_prepare(self, ts_x, dem_x=None, start_idx=0):
        assert len(ts_x.shape) == 3
        assert dem_x is None or len(dem_x.shape) == 2
        dem_len = 0 if dem_x is None else dem_x.shape[1]

        total_num_features = self.num_dem_ftr + self.num_ts_ftr * self.num_window

        x_ = [[i] * total_num_features for i in range(start_idx, start_idx + ts_x.shape[0])]

        return np.array(x_)

    def wraper_predict(self, x):
        assert len(x.shape) == 2

        dem_x, ts_x = x[:, :self.num_dem_ftr].copy(), x[:, self.num_dem_ftr:].copy()

        # initializing the value of all arrays
        ts_x_ = np.zeros((x.shape[0], self.all_ts.shape[1], self.all_ts.shape[2]))
        mask_x_ = np.zeros_like(ts_x_)
        dem_x_ = np.zeros_like(dem_x, dtype=float)
        tstep = np.ones((x.shape[0], self.all_ts.shape[1], 1)) * \
                np.reshape(np.arange(0, self.all_ts.shape[1]), (1, self.all_ts.shape[1], 1))

        # Reshaping the ts indices based on the num time windows and features
        ts_x = ts_x.reshape((ts_x.shape[0], self.num_window, self.num_ts_ftr))

        for i in range(x.shape[0]):
            # creating time series data
            for t in range(self.num_ts_step):
                for j in range(self.num_ts_ftr):
                    # Finding the corresponding time interval
                    wind_t = np.ceil((t + 1) / self.window_len).astype('int') - 1
                    ind = ts_x[i, wind_t, j]
                    ts_x_[i, t, j] = self.all_ts[ind, t, j]
                    mask_x_[i, t, j] = None if self.all_mask is None else self.all_mask[ind, t, j]
            # creating static data
            for j in range(dem_x.shape[1]):
                ind = dem_x[i, j]
                dem_x_[i, j] = None if self.all_dem is None else self.all_dem[ind, j]

        # Creating the input of the model based on the different models. 
        # This part should be updated as new models get involved in the project
        if self.model_type == 'lstm_dem':
            model_input = [ts_x_, dem_x_]
        elif self.model_type == 'grud':
            model_input = [ts_x_, mask_x_, tstep]
        elif self.model_type == 'lstm':
            model_input = ts_x_

        return self.model.predict(model_input)

    def shap_values(self, num_output=1):
        self.explainer = shap.KernelExplainer(self.wraper_predict, self.background_data)
        shap_values = self.explainer.shap_values(self.test_data)
        shap_values = np.array(shap_values)

        self.dem_phi = shap_values[:, :, :self.num_dem_ftr]
        ts_shap_values = shap_values[:, :, self.num_dem_ftr:]
        self.ts_phi = ts_shap_values.reshape(
            (num_output, self.num_test, self.num_window, self.num_ts_ftr))

        # assign values to each single time step by deviding the values by window length
        self.ts_phi = np.repeat(self.ts_phi / self.window_len, self.window_len, axis=2)[:, :,
                      :self.num_ts_step, :]

        # Reporting only the first output
        self.ts_phi = self.ts_phi[0]
        self.dem_phi = self.dem_phi[0]

        return self.ts_phi if self.num_dem_ftr == 0 else (self.dem_phi, self.ts_phi)


class SlidingWindowSHAP():
    '''
    A class for computing the shapely values for time sereis data. Only the shap values for the first output
    is reported.
    
    Parameters:
    model: A model object that will be used for prediction. The model object must have a method called predict() which produces the model output for a given input
    stride: The stride parameter for the Sliding WindowSHAP algorithm
    window_len: The length of the window for the algorithm
    B_ts: A 3D numpy array of background time series data
    test_ts: A 3D numpy array of test time series data
    B_mask: A 3D numpy array of background masking data. It is only used for specific models such as GRUD where a masking variable is passed to the model alongside the time series data. (default: None)
    B_dem: A 2D numpy array of background demographic data (non-temporal data). It is only used for specific models with both modelities of temporal and non-temporal variables. (default: None)
    test_mask: A 3D numpy array of test mask data (default: None)
    test_dem: A 2D numpy array of test demographic data (default: None)
    model_type: The type of model being used. Set the parameter to 'lstm' when time series data is the only input, pick 'lstm_dem' when input includes both time sereis and demographic (non-termporal) data, and 'grud' when you are using GRUD structure.  (default: 'lstm')
    '''

    def __init__(self, model, stride, window_len, B_ts, test_ts, B_mask=None,
                 B_dem=None, test_mask=None, test_dem=None, model_type='lstm'):
        self.model = model
        self.model_type = model_type
        self.stride = stride
        self.window_len = window_len
        self.num_window = 2  # Specific to the sliding time window
        self.num_background = len(B_ts)
        self.num_test = len(test_ts)
        self.background_ts = B_ts
        self.background_mask = B_mask
        self.background_dem = B_dem
        self.test_ts = test_ts
        self.test_mask = test_mask
        self.test_dem = test_dem
        self.ts_phi = None
        self.dem_phi = None
        self.explainer = None

        # Problem sizes
        self.num_ts_ftr = B_ts.shape[2]
        self.num_ts_step = B_ts.shape[1]
        self.num_dem_ftr = 0 if B_dem is None else B_dem.shape[1]

        # Creating all data (background and test together)
        self.all_ts = np.concatenate((self.background_ts, self.test_ts), axis=0)
        self.all_mask = None if test_mask is None else np.concatenate(
            (self.background_mask, self.test_mask), axis=0)
        self.all_dem = None if test_dem is None else np.concatenate(
            (self.background_dem, self.test_dem), axis=0)

        # Creating converted data for SHAP
        self.background_data = self.data_prepare(ts_x=self.background_ts, dem_x=self.background_dem,
                                                 start_idx=0)
        self.test_data = self.data_prepare(ts_x=self.test_ts, dem_x=self.test_dem,
                                           start_idx=self.num_background)

    def data_prepare(self, ts_x, dem_x=None, start_idx=0):
        # Modified for sliding time window
        assert len(ts_x.shape) == 3
        assert dem_x is None or len(dem_x.shape) == 2

        total_num_features = self.num_dem_ftr + self.num_ts_ftr * self.num_window

        x_ = [[i] * total_num_features for i in range(start_idx, start_idx + ts_x.shape[0])]

        return np.array(x_)

    def wraper_predict(self, x, start_ind=0):
        assert len(x.shape) == 2

        # Calculating the indices inside the time window
        inside_ind = list(range(start_ind, start_ind + self.window_len))

        dem_x, ts_x = x[:, :self.num_dem_ftr].copy(), x[:, self.num_dem_ftr:].copy()

        # initializing the value of all arrays
        ts_x_ = np.zeros((x.shape[0], self.num_ts_step, self.num_ts_ftr))
        mask_x_ = np.zeros_like(ts_x_)
        dem_x_ = np.zeros_like(dem_x, dtype=float)
        tstep = np.ones((x.shape[0], self.num_ts_step, 1)) * \
                np.reshape(np.arange(0, self.num_ts_step), (1, self.num_ts_step, 1))

        # Reshaping the ts indices based on the num time windows and features
        ts_x = ts_x.reshape((ts_x.shape[0], self.num_window, self.num_ts_ftr))

        for i in range(x.shape[0]):
            # creating time series data
            for t in range(self.num_ts_step):
                for j in range(self.num_ts_ftr):
                    # Finding the corresponding time interval
                    wind_t = 0 if (t in inside_ind) else 1
                    ind = ts_x[i, wind_t, j]
                    ts_x_[i, t, j] = self.all_ts[ind, t, j]
                    mask_x_[i, t, j] = None if self.all_mask is None else self.all_mask[ind, t, j]
            # creating static data
            for j in range(dem_x.shape[1]):
                ind = dem_x[i, j]
                dem_x_[i, j] = None if self.all_dem is None else self.all_dem[ind, j]

        # Creating the input of the model based on the different models. 
        # This part should be updated as new models get involved in the project
        if self.model_type == 'lstm_dem':
            model_input = [ts_x_, dem_x_]
        elif self.model_type == 'grud':
            model_input = [ts_x_, mask_x_, tstep]
        elif self.model_type == 'lstm':
            model_input = ts_x_

        return self.model.predict(model_input)

    def shap_values(self, num_output=1, nsamples='auto'):
        # Initializing number of time windows and contribution score matrices
        seq_len = self.background_ts.shape[1]
        num_sw = np.ceil((seq_len - self.window_len) / self.stride).astype('int') + 1
        ts_phi = np.zeros((self.num_test, num_sw, 2, self.background_ts.shape[2]))
        dem_phi = np.zeros((self.num_test, num_sw, self.num_dem_ftr))

        # Determining the number of samples
        if nsamples == 'auto':
            nsamples = 10 * self.num_ts_ftr + 5 * self.num_dem_ftr

        # Main loop on different possible windows
        for stride_cnt in range(num_sw):
            predict = lambda x: self.wraper_predict(x, start_ind=stride_cnt * self.stride)

            # Running SHAP
            self.explainer = shap.KernelExplainer(predict, self.background_data)
            shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples)
            shap_values = np.array(shap_values)

            # Extracting the SHAP values and storing them
            dem_shap_values_ = shap_values[:, :, :self.num_dem_ftr]
            ts_shap_values = shap_values[:, :, self.num_dem_ftr:]
            ts_shap_values = ts_shap_values.reshape((num_output, self.num_test, 2, self.num_ts_ftr))

            ts_phi[:, stride_cnt, :, :] = ts_shap_values[0]
            dem_phi[:, stride_cnt, :] = dem_shap_values_[0]

        # Averaging shap values from different windows
        ts_phi_agg = np.empty((self.num_test, num_sw, self.num_ts_step, self.num_ts_ftr))
        ts_phi_agg[:] = np.nan
        for k in range(num_sw):
            ts_phi_agg[:, k, k * self.stride:k * self.stride + self.window_len, :] = ts_phi[:, k, 0,
                                                                                     :][:,
                                                                                     np.newaxis, :]
        ts_phi_agg = np.nanmean(ts_phi_agg, axis=1)
        dem_phi = np.nanmean(dem_phi, axis=1)

        self.dem_phi = dem_phi
        self.ts_phi = ts_phi_agg

        return ts_phi_agg if self.num_dem_ftr == 0 else (dem_phi, ts_phi_agg)


class DynamicWindowSHAP():
    '''
    A class for computing the shapely values for time sereis data. Only the shap values for the first output
    is reported.
    
    Parameters:
    model: A model object that will be used for prediction. The model object must have a method called predict() which produces the model output for a given input
    delta: The treshold value in Dynamic WindowSHAP algorithm
    n_w: The maximum allowed number of time windows for each variable
    B_ts: A 3D numpy array of background time series data
    test_ts: A 3D numpy array of test time series data
    B_mask: A 3D numpy array of background masking data. It is only used for specific models such as GRUD where a masking variable is passed to the model alongside the time series data. (default: None)
    B_dem: A 2D numpy array of background demographic data (non-temporal data). It is only used for specific models with both modelities of temporal and non-temporal variables. (default: None)
    test_mask: A 3D numpy array of test mask data (default: None)
    test_dem: A 2D numpy array of test demographic data (default: None)
    model_type: The type of model being used. Set the parameter to 'lstm' when time series data is the only input, pick 'lstm_dem' when input includes both time sereis and demographic (non-termporal) data, and 'grud' when you are using GRUD structure.  (default: 'lstm')
    '''

    def __init__(self, model, delta, n_w, B_ts, test_ts, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='lstm'):
        self.model = model
        self.model_type = model_type
        self.num_background = len(B_ts)
        self.num_test = len(test_ts)
        self.background_ts = B_ts
        self.background_mask = B_mask
        self.background_dem = B_dem
        self.test_ts = test_ts
        self.test_mask = test_mask
        self.test_dem = test_dem
        self.ts_phi = None
        self.dem_phi = None
        self.explainer = None

        # Problem sizes
        self.num_ts_ftr = B_ts.shape[2]
        self.num_ts_step = B_ts.shape[1]
        self.num_dem_ftr = 0 if B_dem is None else B_dem.shape[1]

        ## Specific to Binary Time Window
        assert self.num_test == 1  # For binary time window algorithm, samples should be fed to the algorithm one-by-one
        self.delta = delta
        self.n_w = n_w
        self.split_points = [[self.num_ts_step - 1]] * self.num_ts_ftr  # Splitting points
        self.num_window = [1] * self.num_ts_ftr

        # Creating all data (background and test together)
        self.all_ts = np.concatenate((self.background_ts, self.test_ts), axis=0)
        self.all_mask = None if test_mask is None else np.concatenate(
            (self.background_mask, self.test_mask), axis=0)
        self.all_dem = None if test_dem is None else np.concatenate(
            (self.background_dem, self.test_dem), axis=0)

        # Creating converted data for SHAP
        self.background_data = self.data_prepare(ts_x=self.background_ts, dem_x=self.background_dem,
                                                 start_idx=0)
        self.test_data = self.data_prepare(ts_x=self.test_ts, dem_x=self.test_dem,
                                           start_idx=self.num_background)

    def data_prepare(self, ts_x, dem_x=None, start_idx=0):
        assert len(ts_x.shape) == 3
        assert dem_x is None or len(dem_x.shape) == 2
        total_num_features = self.num_dem_ftr + sum(
            self.num_window)  ## Specific to Binary Time Window

        x_ = [[i] * total_num_features for i in range(start_idx, start_idx + ts_x.shape[0])]

        return np.array(x_)

    def wraper_predict(self, x):
        assert len(x.shape) == 2

        dem_x, ts_x = x[:, :self.num_dem_ftr].copy(), x[:, self.num_dem_ftr:].copy()

        # initializing the value of all arrays
        ts_x_ = np.zeros((x.shape[0], self.num_ts_step, self.num_ts_ftr))
        mask_x_ = np.zeros_like(ts_x_)
        dem_x_ = np.zeros_like(dem_x, dtype=float)
        tstep = np.ones((x.shape[0], self.num_ts_step, 1)) * \
                np.reshape(np.arange(0, self.num_ts_step), (1, self.num_ts_step, 1))

        # Reshaping the ts indices based on the time windows for each feature
        ## Specific to Binary Time Window
        temp_ts_x = np.zeros((ts_x.shape[0], max(self.num_window), self.num_ts_ftr), dtype=int)
        for i in range(self.num_ts_ftr):
            temp_ts_x[:, :self.num_window[i], i] = ts_x[:, sum(self.num_window[:i]):sum(
                self.num_window[:i + 1])]
        ts_x = temp_ts_x

        for i in range(x.shape[0]):
            # creating time series data
            for j in range(self.num_ts_ftr):
                # Finding the corresponding time interval
                wind_t = np.searchsorted(self.split_points[j], np.arange(
                    self.num_ts_step))  ## Specific to Binary Time Window
                for t in range(self.num_ts_step):
                    ind = ts_x[i, wind_t[t], j]
                    ts_x_[i, t, j] = self.all_ts[ind, t, j]
                    mask_x_[i, t, j] = None if self.all_mask is None else self.all_mask[ind, t, j]
            # creating static data
            for j in range(dem_x.shape[1]):
                ind = dem_x[i, j]
                dem_x_[i, j] = None if self.all_dem is None else self.all_dem[ind, j]

        # Creating the input of the model based on the different models. 
        # This part should be updated as new models get involved in the project
        if self.model_type == 'lstm_dem':
            model_input = [ts_x_, dem_x_]
        elif self.model_type == 'grud':
            model_input = [ts_x_, mask_x_, tstep]
        elif self.model_type == 'lstm':
            model_input = ts_x_

        return self.model.predict(model_input)

    def shap_values(self, num_output=1, nsamples_in_loop='auto', nsamples_final='auto'):
        flag = 1
        while flag:
            flag = 0
            # Updating the number of time windows for each time series feature
            self.num_window = [len(self.split_points[i]) for i in range(self.num_ts_ftr)]

            # Updating converted data for SHAP
            self.background_data = self.data_prepare(ts_x=self.background_ts,
                                                     dem_x=self.background_dem, start_idx=0)
            self.test_data = self.data_prepare(ts_x=self.test_ts, dem_x=self.test_dem,
                                               start_idx=self.num_background)

            # Running SHAP
            if nsamples_in_loop == 'auto':
                nsamples = 2 * sum(self.num_window)
            else:
                nsamples = nsamples_in_loop

            self.explainer = shap.KernelExplainer(self.wraper_predict, self.background_data)
            shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples)
            shap_values = np.array(shap_values)
            dem_phi = shap_values[0, :, :self.num_dem_ftr]  # Extracting dem SHAP values
            ts_shap_values = shap_values[:, :, self.num_dem_ftr:]  # Extracting ts SHAP values

            # Checking the maximum number of windows condition
            if max(self.num_window) >= self.n_w: break

            for i in range(self.num_ts_ftr):
                S = set(self.split_points[i])
                for j in range(self.num_window[i]):
                    if abs(ts_shap_values[0, 0, sum(self.num_window[:i]) + j]) > self.delta:
                        S.add(int(self.split_points[i][j] / 2) if j == 0 else int(
                            (self.split_points[i][j - 1] + self.split_points[i][j]) / 2))
                if set(S) != set(self.split_points[i]):
                    flag += 1
                    self.split_points[i] = list(S)
                    self.split_points[i].sort()

        # Running SHAP with large number of samples for the final evaluation of Shapely values
        self.explainer = shap.KernelExplainer(self.wraper_predict, self.background_data)
        shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples_final)
        shap_values = np.array(shap_values)
        dem_phi = shap_values[0, :, :self.num_dem_ftr]  # Extracting dem SHAP values
        ts_shap_values = shap_values[:, :, self.num_dem_ftr:]  # Extracting ts SHAP values

        # Assigning Shap values to each single time step
        ts_phi = np.zeros((self.num_test, self.num_ts_step, self.num_ts_ftr))
        for i in range(self.num_ts_ftr):
            for j in range(self.num_window[i]):
                # This part of the code is written in a way that each splitting point belongs to the time window that starts from that point
                # For the last time window, both splitting points at the end and start of the time window belong to it
                start_ind = 0 if j == 0 else self.split_points[i][j - 1]
                end_ind = self.split_points[i][j] + int((j + 1) / self.num_window[i])
                ts_phi[0, start_ind:end_ind, i] = ts_shap_values[0, :,
                                                  sum(self.num_window[:i]) + j] / (
                                                              end_ind - start_ind)
        self.dem_phi = dem_phi
        self.ts_phi = ts_phi

        return ts_phi if self.num_dem_ftr == 0 else (dem_phi, ts_phi)


if __name__ == "__main__":
    pass
