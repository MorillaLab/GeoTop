
from histogram_utils import calculate_histograms

def process_wrapper(i, U_train_rgb, U_train, XX_train_geo_shared, n):
    calculate_histograms(i, U_train_rgb, U_train, XX_train_geo_shared, n)
