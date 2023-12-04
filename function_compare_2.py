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