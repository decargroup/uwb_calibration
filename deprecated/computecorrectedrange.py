import pickle
import numpy as np

class ComputeCorrectedRange(object):
    """
    # TODO: 
    
    Object to retrieve and correct range measurements.
    """
    
    _c = 299702547 # speed of light
    _dwt_to_ns = 1e9 * (1.0 / 499.2e6 / 128.0) # DW time unit to nanoseconds

    def __init__(self, in_ns=False):
        """
        Constructor
        """
        if in_ns:
            self._dwt_to_ns = 1

        # Retrieve pre-determined calibration results
        with open("calib_results.pickle", 'rb') as pickle_file:
            calib_results = pickle.load(pickle_file)
        
        self.delays = calib_results['delays']
        self.bias_spl = calib_results['bias_spl']
        self.std_spl = calib_results['std_spl']

    def get_corrected_range(self,uwb_data):
        """
        Extracts and corrects the range measurement and 
        associated information.

        PARAMETERS:
        -----------
        uwb_data: RangeStamped
            One instance of UWB data. Can also pass many instances for one pair.

        RETURNS:
        --------
         dict: Dictionary with 4 fields.
            from_id: int
                ID of initiating tag.
            to_id: int
                ID of target tag.
            range: float
                Corrected range measurement.
            std: float
                Standard deviation of corrected range measurement.
        """
        # Get tag IDs
        from_id = int(np.array(uwb_data["from_id"])[0])
        to_id = int(np.array(uwb_data["to_id"])[0])

        # Get timestamps
        tx1 = uwb_data["tx1"]*self._dwt_to_ns
        rx1 = uwb_data["rx1"]*self._dwt_to_ns
        tx2 = uwb_data["tx2"]*self._dwt_to_ns
        rx2 = uwb_data["rx2"]*self._dwt_to_ns
        tx3 = uwb_data["tx3"]*self._dwt_to_ns
        rx3 = uwb_data["rx3"]*self._dwt_to_ns

        # Correct clock wrapping 
        rx2, rx3 = self._unwrap_ts(tx1, rx2, rx3)
        tx2, tx3 = self._unwrap_ts(rx1, tx2, tx3)

        # Compute time intervals
        Ra1 = rx2 - tx1
        Ra2 = rx3 - rx2
        Db1 = tx2 - rx1
        Db2 = tx3 - tx2

        # Get antenna delays
        delay_0 = self.delays[from_id]
        delay_1 = self.delays[to_id]

        # Correct time intervals for antenna delays
        Ra1 += delay_0
        Db1 -= delay_1

        # Get power 
        fpp1 = uwb_data["fpp1"]
        fpp2 = uwb_data["fpp2"]

        # Implement lifting function
        fpp1_lift = self.lift(fpp1)
        fpp2_lift = self.lift(fpp2)

        # Get average lifted power
        fpp_lift_avg = 0.5 * (fpp1_lift + fpp2_lift)

        # Power-induced bias 
        bias = self.bias_spl(fpp_lift_avg)

        # Compute range measurement
        range = self._compute_range(Ra1, Ra2, Db1, Db2, bias)

        # Get standard deviation of measurement
        std = self.std_spl(fpp_lift_avg)

        return {
                'from_id': from_id,
                'to_id': to_id,
                'range': range,
                'std': std
               }


    def _unwrap_ts(self, ts1, ts2, ts3):
        """
        Corrects the UWB-module's clock unwrapping.

        PARAMETERS:
        -----------
        ts1: int
            First timestamp in a sequence of timestamps registered 
            on the same clock.
        ts2: int
            Second timestamp in a sequence of timestamps registered 
            on the same clock.
        ts3: int
            Third timestamp in a sequence of timestamps registered 
            on the same clock.

        RETURNS:
        --------
        ts2: int
            Unwrapped second timestamp in a sequence of timestamps 
            registered on the same clock.
        ts3: int
            Unwrapped third timestamp in a sequence of timestamps 
            registered on the same clock.
        """
        # The timestamps are registered as type uint32.
        max_time_ns = 2**32 * self._dwt_to_ns
        
        idx = ts2 < ts1
        ts2 += idx*max_time_ns
        ts3 += idx*max_time_ns
        
        idx = ts3 < ts2
        ts3 += idx*max_time_ns

        return ts2, ts3

    @staticmethod
    def lift(x, alpha=-82):
        """
        Lifting function for better visualization and calibration. 
        Based on Cano, J., Pages, G., Chaumette, E., & Le Ny, J. (2022). Clock 
                 and Power-Induced Bias Correction for UWB Time-of-Flight Measurements.
                 IEEE Robotics and Automation Letters, 7(2), 2431-2438. 
                 https://doi.org/10.1109/LRA.2022.3143202

        PARAMETERS:
        -----------
        x: np.array(n,1)
            Input to lifting function. Received Power in dBm in this context.
        alpha: scalar
            Centering parameter. Default: -82 dBm.

        RETURNS:
        --------
        np.array(n,1)
            Array of lifted received power. 
        """
        return 10**((x - alpha) /10)

    def _compute_range(self, Ra1, Ra2, Db1, Db2, bias):
        """
        Compute the bias-corrected range measurement.

        PARAMETERS:
        -----------
        Ra1: int
            rx2 - rx1.
        Ra2: int
            rx3 - rx2.
        Db1: int
            tx2 - rx1.
        Db2: int
            tx3 - tx2.
        bias: float
            Estimated power-induced bias.

        RETURNS:
        --------
        float
            Bias-corrected range measurement.
        """
        return 0.5 * self._c / 1e9 * \
                (Ra1 - (Ra2 / Db2) * Db1) - bias

if __name__ == '__main__':
    obj = ComputeCorrectedRange()