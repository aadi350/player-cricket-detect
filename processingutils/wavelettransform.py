import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pywt
from typing import Tuple

from plotly.subplots import make_subplots

bior_1 = [
    'bior1.1',
    'bior1.3',
    'bior1.5'
]

bior_2 = [
    'bior2.2',
    'bior2.4',
    'bior2.6',
    'bior2.8'
]

bior_3 = [
    'bior3.1',
    'bior3.3',
    'bior3.5',
    'bior3.7',
    'bior3.9'
]

waves = []
wave_fam = pywt.wavelist(kind='discrete')
for fam in wave_fam:
    waves.append(fam)


def _normalise(frame, min: int = 0, max: int = 255):
    return np.array(frame / (max - min))


def _center_range(frame, midpoint: int = 0, max_in: int = 1, min_in: int = 0):
    return frame - (max_in - min_in) / 2 + midpoint


def _visualise_single(frame, wavelet: str = 'bior3.9', mode: str = 'sym', level: int = 2):
    """
    :param frame:
    :param wavelet:
    :param mode:
    :param level:
    :return:
    coeff_arr : array-like
    Wavelet transform coefficient array.

    coeff_slices : list
    List of slices corresponding to each coefficient.
        As a 2D example, coeff_arr[coeff_slices[1]['dd']] would extract the first level detail coefficients from coeff_arr.
    """
    arr = _transform_single(frame, wavelet, mode, level)[0][0]
    arr, slices = pywt.coeffs_to_array(arr)
    return arr, slices


def _transform_single(frame, wavelet: str = 'bior3.9', mode: str = 'sym', level: int = 2) -> list:
    """
    :param frame:
    :param wavelet:
    :param mode:
    :param level:
    :return: list in the form
        [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)]
    """
    # wavelet = pywt.ContinuousWavelet(wavelet)
    arr = pywt.wavedec2(np.array(frame), wavelet, mode='periodization', level=level)
    arr[0] /= np.abs(arr[0]).max()
    for detail_level in range(level):
        arr[detail_level + 1] = [d / np.abs(d).max() for d in arr[detail_level + 1]]
    return arr


def hist_single(frame, wavelet: str = 'bior3.9', mode: str = 'sym', level: int = 2, show: bool = False,
                return_bins: bool = True):
    """computes centred histogram of a single frame and returns bins with count-per-bin
    :param return_bins:
    :param frame: input frame
    :param wavelet: string specifying wavelet type (e.g.'bior1.1', 'db1')
    :param mode:
    :param level: depth of wavelet transform
    :param show: plot using plotly
    :return: count per bins
    """
    # arr = [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)]
    arr = _transform_single(frame, wavelet, mode, level)
    cAn = arr[0]
    if show:
        count, bin_edges = np.histogram(cAn * 255, bins=range(0, 255, 1))
        bins = bin_edges[:-1] + np.diff(bin_edges) / 2
        _show_single_histogram(count, bins)
    count, bin_edges = np.histogram(cAn * 255, bins=range(0, 255, 1))
    bins = bin_edges[:-1] + np.diff(bin_edges) / 2
    if return_bins:
        return (count, bins)

    return count


def hist(frames, show: bool = False):
    """computes centred histogram of a single frame and returns bins with count-per-bin
       :param return_bins:
       :param frame: input frame
       :param wavelet: string specifying wavelet type (e.g.'bior1.1', 'db1')
       :param mode:
       :param level: depth of wavelet transform
       :param show: plot using plotly
       :return: count per bins
       """
    count, bins = [], []
    for frame in frames:
        h = hist_single(frame, show=False, return_bins=True)
        count.append(h[0])
        bins.append(h[1])

    if show:
        ROWS, COLS = 4, 3
        fig = make_subplots(rows=ROWS, cols=COLS, print_grid=True)
        col, row = 0, 1
        for count_, bins_ in zip(count, bins):

            if col >= COLS:
                row += 1
                col = 0
            if row > ROWS: break
            col += 1
            print(count_)

            fig.add_trace(
                go.Histogram(x=count_, nbinsx=100, xbins=dict({'start': 1, 'end': 255, 'size': 1})),
                col=col,
                row=row,
            )
        fig.show()

    return count


def _show_single_histogram(count, bins):
    fig = px.histogram(x=bins, y=count, nbins=255)
    fig.show()


def transform_array(frames):
    approx = [_transform_single(frame) for frame in frames]
    return approx


def transform_array_hist(frames, wavelet: str = 'bior3.9'):
    histarray = [hist_single(frame, return_bins=False, show=False, wavelet=wavelet) for frame in frames]
    return histarray


def vis_array(frames):
    vis_frame = [_visualise_single(frame) for frame in frames]
    return vis_frame
