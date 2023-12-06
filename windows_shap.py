import numpy as np

def data_prepare(ts_x, num_dem_ftr, num_window, num_ts_ftr = None, start_idx=0):
    total_num_features = num_dem_ftr + num_ts_ftr * num_window if num_ts_ftr\
                             else num_dem_ftr + sum(num_window)
  
    x_ = [[i] * total_num_features for i in range(start_idx, start_idx + ts_x.shape[0])]

    return np.array(x_)

class SHAP():
    def __init__(self, model, B_ts, test_ts, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='lstm'):
         self.model = model
         self.B_ts = B_ts
         self.test_ts = test_ts
         self.B_mask = B_mask
         self.B_dem = B_dem
         self.test_mask = test_mask
         self.test_dem = test_dem
         self.model_type = model_type
         self.num_ts_ftr = B_ts.shape[2]
         self.num_ts_step = B_ts.shape[1]
         self.num_dem_ftr = 0 if B_dem is None else B_dem.shape[1]


class StationaryWindowSHAP(SHAP):
       def __init__(self, model, window_len, B_ts, test_ts, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='lstm'):
        
        self.num_window = np.ceil(B_ts.shape[1] / window_len).astype('int')
        self.window_len = window_len

        super().__init__(model, B_ts, test_ts, B_mask,
                         B_dem, test_mask, test_dem, model_type)
        
        self.background_data = data_prepare(B_ts, self.num_dem_ftr, 
                                            self.num_window,
                                            start_idx=0)
        self.test_data = data_prepare(test_ts, self.num_dem_ftr, 
                                    self.num_window, start_idx=len(B_ts))

class SlidingWindowSHAP(SHAP):
     def __init__(self, model, stride, window_len, B_ts, test_ts, B_mask=None,
                 B_dem=None, test_mask=None, test_dem=None, model_type='lstm'):
        
        self.stride = stride
        self.num_window = 2
        self.window_len = window_len

        super().__init__(model, B_ts, test_ts, B_mask, 
                       B_dem, test_mask, test_dem, model_type)

        self.background_data = data_prepare(B_ts, self.num_dem_ftr, 
                                            self.num_window,
                                            start_idx=0)
        self.test_data = data_prepare(test_ts,self.num_dem_ftr, 
                                    self.num_window,
                                      start_idx=len(B_ts))

class DynamicWindowSHAP(SHAP):
    def __init__(self, model, delta, n_w, B_ts, test_ts, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='lstm'):
        
        self.num_window = [1] * self.num_ts_ftr
        super().__init__(model, B_ts, test_ts, B_mask, 
                       B_dem, test_mask, test_dem, model_type)

        self.background_data = data_prepare(B_ts, self.num_dem_ftr,
                                            self.num_window, 
                                            self.num_ts_ftr)
        self.test_data = data_prepare(test_ts, 
                                      self.num_dem_ftr,
                                      self.num_window,
                                      self.num_ts_ftr,
                                      len(B_ts))
             