    def get_model_inputs(self, x, window_len):

        dem_x, ts_x = x[:, :self.num_dem_ftr].copy(), x[:, self.num_dem_ftr:].copy()

        # initializing the value of all arrays
        ts_x_ = np.zeros((x.shape[0], self.all_ts.shape[1], self.all_ts.shape[2]))
        mask_x_ = np.zeros_like(ts_x_)
        dem_x_ = np.zeros_like(dem_x, dtype=float)
        tstep = np.ones((x.shape[0], self.all_ts.shape[1], 1)) * \
                np.reshape(np.arange(0, self.all_ts.shape[1]), (1, self.all_ts.shape[1], 1))

        # Reshaping the ts indices based on the num time windows and features
        ts_x = ts_x.reshape((ts_x.shape[0], self.num_windows, self.num_ts_ftr))

        for i in range(x.shape[0]):
            # creating time series data
            for t in range(self.num_ts_step):
                for j in range(self.num_ts_ftr):
                    # Finding the corresponding time interval
                    wind_t = np.ceil((t + 1) / window_len).astype('int') - 1
                    ind = ts_x[i, wind_t, j]
                    ts_x_[i, t, j] = self.all_ts[ind, t, j]
                    mask_x_[i, t, j] = None if self.all_mask is None else self.all_mask[ind, t, j]
            # creating static data
            for j in range(dem_x.shape[1]):
                ind = dem_x[i, j]
                dem_x_[i, j] = None if self.all_dem is None else self.all_dem[ind, j]

        # Creating the input of the model based on the different models. 
        # This part should be updated as new models get involved in the project

        return ts_x_, dem_x_, tstep