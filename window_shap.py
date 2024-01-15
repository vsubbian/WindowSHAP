import numpy as np
import shap
import ruptures as rpt

def data_prepare(ts_x, num_dem_ftr, num_window, num_ts_ftr=None, start_idx=0, dynamic=False):
    """returns prepared data for SHAP"""
    total_num_features = num_dem_ftr + num_ts_ftr * num_window if not dynamic \
        else num_dem_ftr + sum(num_window)

    x_ = [[i] * total_num_features for i in range(start_idx, start_idx + ts_x.shape[0])]

    return np.array(x_)


class SHAP():
    """Template for SHAP descendants. Accumulates common fields and methods."""

    def __init__(self, model, B_ts, test_ts, num_window, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='lstm'):
        self.model = model
        self.B_ts = B_ts
        self.test_ts = test_ts
        self.num_window = num_window
        self.B_mask = B_mask
        self.B_dem = B_dem
        self.test_mask = test_mask
        self.test_dem = test_dem
        self.model_type = model_type
        self.num_ts_ftr = B_ts.shape[2]
        self.num_ts_step = B_ts.shape[1]
        self.num_dem_ftr = 0 if B_dem is None else B_dem.shape[1]
        self.background_data = None
        self.test_data = NotImplementedError

        self.explainer = None
        self.dem_phi = None
        self.ts_phi = None

        self.all_ts = np.concatenate((B_ts, test_ts), axis=0)
        self.all_mask = None if test_mask is None else np.concatenate(
            (B_mask, self.test_mask), axis=0)
        self.all_dem = None if test_dem is None else np.concatenate(
            (B_dem, test_dem), axis=0)

    def prepare_data(self, dynamic=False):
        """prepares SHAP data."""
        self.background_data = data_prepare(self.B_ts,
                                            self.num_dem_ftr,
                                            self.num_window,
                                            num_ts_ftr=self.num_ts_ftr,
                                            dynamic=dynamic)
        self.test_data = data_prepare(self.test_ts,
                                      self.num_dem_ftr,
                                      self.num_window,
                                      num_ts_ftr=self.num_ts_ftr,
                                      start_idx=len(self.B_ts),
                                      dynamic=dynamic)

    def get_ts_x_(self, x):
        """returns ts_x_ (with _)"""
        return np.zeros((x.shape[0], self.all_ts.shape[1], self.all_ts.shape[2]))


    def get_ts_x(self, x):
        """returns ts_x."""
        ts_x = x[:, self.num_dem_ftr:].copy()
        return ts_x.reshape((ts_x.shape[0], self.num_window, self.num_ts_ftr))

    def create_static_data(self, dem_x, dem_x_, i):
        """returns dem_x_ (with _)"""
        for j in range(dem_x.shape[1]):
            ind = dem_x[i, j]
            dem_x_[i, j] = None if self.all_dem is None else self.all_dem[ind, j]

        return dem_x_

    def get_wind_t(self, t, start_ind=0):
        """template for descendants."""

    def creating_data(self, x, ts_x, ts_x_, mask_x_, start_ind=0):
        """returns filled ts_x_, mask_x_"""
        for i in range(x.shape[0]):
            # creating time series data
            for t in range(self.num_ts_step):
                for j in range(self.num_ts_ftr):
                    # Finding the corresponding time interval
                    ind = ts_x[i, self.get_wind_t(t, start_ind), j]
                    ts_x_[i, t, j] = self.all_ts[ind, t, j]
                    mask_x_[i, t, j] = None if self.all_mask is None else self.all_mask[ind, t, j]
        return ts_x_, mask_x_

    def get_tstep(self, x):
        """returns tstep"""
        return np.ones((x.shape[0], self.num_ts_step, 1)) * \
            np.reshape(np.arange(0, self.num_ts_step), (1, self.num_ts_step, 1))


    def get_model_inputs(self, x, start_ind=0):
        """returns model input for delivered possible models."""
        dem_x, ts_x = x[:, :self.num_dem_ftr].copy(), x[:, self.num_dem_ftr:].copy()
        ts_x_ = self.get_ts_x_(x)
        ts_x = self.get_ts_x(x)
        tstep = self.get_tstep(x)
        mask_x_ = np.zeros_like(ts_x_)
        dem_x_ = np.zeros_like(dem_x, dtype=float)
        ts_x_, mask_x_ = self.creating_data(x, ts_x, ts_x_, mask_x_, start_ind)
        return ts_x_, dem_x_, mask_x_, tstep

    def wrapper_predict(self, x, start_ind=0):
        """predicts for delivered model."""
        ts_x_, dem_x_, mask_x_, tstep = self.get_model_inputs(x, start_ind)

        model_dict = {'lstm_dem': [ts_x_, dem_x_],
                      'grud': [ts_x_, mask_x_, tstep],
                      'lstm': ts_x_}
        return self.model.predict(model_dict[self.model_type])


class StationaryWindowSHAP(SHAP):
    """StationaryWindowSHAP - SHAP with established window_len"""

    def __init__(self, model, window_len, B_ts, test_ts, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='lstm'):

        num_window = np.ceil(B_ts.shape[1] / window_len).astype('int')
        self.window_len = window_len

        super().__init__(model, B_ts, test_ts, num_window, B_mask,
                         B_dem, test_mask, test_dem, model_type)

        self.prepare_data()


    def get_wind_t(self, t, start_ind=0):
        return np.ceil((t + 1) / self.window_len).astype('int') - 1


    def shap_values(self, num_output=1):
        """ shap values for Static Window"""
        self.explainer = shap.KernelExplainer(self.wrapper_predict, self.background_data)
        shap_values = self.explainer.shap_values(self.test_data)
        shap_values = np.array(shap_values)

        self.dem_phi = shap_values[:, :, :self.num_dem_ftr]
        ts_shap_values = shap_values[:, :, self.num_dem_ftr:]
        self.ts_phi = ts_shap_values.reshape(
            (num_output, len(self.test_ts), self.num_window, self.num_ts_ftr))

        # assign values to each single time step by deviding the values by window length
        self.ts_phi = np.repeat(self.ts_phi / self.window_len, self.window_len, axis=2)[:, :,
                      :self.num_ts_step, :]

        # Reporting only the first output
        self.ts_phi = self.ts_phi[0]
        self.dem_phi = self.dem_phi[0]

        return self.ts_phi if self.num_dem_ftr == 0 else (self.dem_phi, self.ts_phi)


class SlidingWindowSHAP(SHAP):
    """SlidingWindowSHAP class"""

    def __init__(self, model, stride, window_len, B_ts, test_ts, B_mask=None,
                 B_dem=None, test_mask=None, test_dem=None, model_type='lstm'):

        self.stride = stride
        num_window = 2
        self.window_len = window_len

        super().__init__(model, B_ts, test_ts, num_window, B_mask,
                         B_dem, test_mask, test_dem, model_type)

        self.prepare_data()

    def get_ts_x_(self, x):
        return np.zeros((x.shape[0], self.num_ts_step, self.num_ts_ftr))

    def get_wind_t(self, t, start_ind):
        inside_ind = list(range(start_ind, start_ind + self.window_len))
        return 0 if (t in inside_ind) else 1

    def shap_values(self, num_output=1, nsamples='auto'):
        """shap values for sliding window"""

        # Initializing number of time windows and contribution score matrices
        seq_len = self.B_ts.shape[1]
        num_sw = np.ceil((seq_len - self.window_len) / self.stride).astype('int') + 1
        ts_phi = np.zeros((len(self.test_ts), num_sw, 2, self.B_ts.shape[2]))
        dem_phi = np.zeros((len(self.test_ts), num_sw, self.num_dem_ftr))

        # Determining the number of samples
        if nsamples == 'auto':
            nsamples = 10 * self.num_ts_ftr + 5 * self.num_dem_ftr

        # Main loop on different possible windows
        for stride_cnt in range(num_sw):
            predict = lambda x: self.wrapper_predict(x, start_ind=stride_cnt * self.stride)

            # Running SHAP
            self.explainer = shap.KernelExplainer(predict, self.background_data)
            shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples)
            shap_values = np.array(shap_values)

            # Extracting the SHAP values and storing them
            dem_shap_values_ = shap_values[:, :, :self.num_dem_ftr]
            ts_shap_values = shap_values[:, :, self.num_dem_ftr:]
            ts_shap_values = ts_shap_values.reshape((num_output, len(self.test_ts),
                                                     2, self.num_ts_ftr))

            ts_phi[:, stride_cnt, :, :] = ts_shap_values[0]
            dem_phi[:, stride_cnt, :] = dem_shap_values_[0]

        # Averaging shap values from different windows
        ts_phi_agg = np.empty((len(self.test_ts), num_sw, self.num_ts_step, self.num_ts_ftr))
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


class DynamicWindowSHAP(SHAP):
    """DynamicWindowSHAP class"""

    def __init__(self, model, delta, n_w, B_ts, test_ts, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='lstm'):

        self.delta = delta
        self.n_w = n_w
        num_window = [1] * B_ts.shape[2]
        super().__init__(model, B_ts, test_ts, num_window, B_mask,
                         B_dem, test_mask, test_dem, model_type)

        self.split_points = [[self.num_ts_step - 1]] * self.num_ts_ftr  # Splitting points

        self.prepare_data(dynamic=True)

    def get_ts_x_(self, x):
        return np.zeros((x.shape[0], self.num_ts_step, self.num_ts_ftr))

    def get_ts_x(self, x):
        ts_x = x[:, self.num_dem_ftr:].copy()

        temp_ts_x = np.zeros((ts_x.shape[0], max(self.num_window), self.num_ts_ftr), dtype=int)
        for i in range(self.num_ts_ftr):
            temp_ts_x[:, :self.num_window[i], i] = ts_x[:, sum(self.num_window[:i]):sum(
                self.num_window[:i + 1])]
        return temp_ts_x

    def get_tstep(self, x):
        return np.ones((x.shape[0], self.num_ts_step, 1)) * \
            np.reshape(np.arange(0, self.num_ts_step), (1, self.num_ts_step, 1))

    def creating_data(self, x, ts_x, ts_x_, mask_x_, start_ind=0):
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
        return ts_x_, mask_x_

    def shap_values(self, num_output=1, nsamples_in_loop='auto', nsamples_final='auto'):
        """shap value for dynamic window."""

        flag = 1
        while flag:
            flag = 0

            # Updating the number of time windows for each time series feature
            self.num_window = [len(self.split_points[i]) for i in range(self.num_ts_ftr)]

            # Updating converted data for SHAP
            self.background_data = data_prepare(self.B_ts, 
                                                self.num_dem_ftr,
                                                self.num_window,
                                                self.num_ts_ftr,
                                                dynamic=True)
            self.test_data = data_prepare(self.test_ts,
                                          self.num_dem_ftr,
                                          self.num_window,
                                          self.num_ts_ftr,
                                          len(self.B_ts),
                                          dynamic=True)

            # Running SHAP
            if nsamples_in_loop == 'auto':
                nsamples = 2 * sum(self.num_window)
            else:
                nsamples = nsamples_in_loop

            self.explainer = shap.KernelExplainer(self.wrapper_predict, self.background_data)
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
        self.explainer = shap.KernelExplainer(self.wrapper_predict, self.background_data)
        shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples_final)
        shap_values = np.array(shap_values)
        dem_phi = shap_values[0, :, :self.num_dem_ftr]  # Extracting dem SHAP values
        ts_shap_values = shap_values[:, :, self.num_dem_ftr:]  # Extracting ts SHAP values

        # Assigning Shap values to each single time step
        ts_phi = np.zeros((len(self.test_ts), self.num_ts_step, self.num_ts_ftr))
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
