    def wraper_predict(self, x, num_window):
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
        temp_ts_x = np.zeros((ts_x.shape[0], max(num_window), self.num_ts_ftr), dtype=int)
        for i in range(self.num_ts_ftr):
            temp_ts_x[:, :num_window[i], i] = ts_x[:, sum(num_window[:i]):sum(
                num_window[:i + 1])]
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
        return ts_x_, dem_x_, tstep