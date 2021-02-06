import cv2
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ORB_DET = cv2.ORB_create(nfeatures=150)


class ORBExtractor:
    def __init__(self, nfeatures: int = 150, denormalise: bool= True):
        self._ORB_DET = cv2.ORB_create(nfeatures=nfeatures)

    def orb_single(self, frame, show: bool = False):
        """computes and shows (optionally) ORB feature-keypoints of frame
        :param frame: input frame
        :param show: plot using plotly
        :return: keypoints and descriptors
        """
        # arr = [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)]
        import plotly.express as px
        kp, des = self._ORB_DET.detectAndCompute(np.float32(frame * 255), None)
        if show:
            img_kp = cv2.drawKeypoints(np.uint8(frame * 255), kp, None)
            fig2 = px.imshow(img_kp)
            fig2.show()

        return kp, des

    def orb_array(self, frames, show: bool = False, denormalise: int = 255):
        """computes centred histogram of a single frame and returns bins with count-per-bin
           :param denormalise: integral rescale frames
           :param frame: input frame
           :param show: plot using plotly
           :return: returns keypoints and descriptor locations
           """
        kp, des = [], []
        for frame in tqdm(frames):
            _kp = self._ORB_DET.detect(np.uint8(frame * denormalise), None)
            _kp, _des = ORB_DET.compute(np.uint8(frame * denormalise), _kp)
            kp.append(_kp)
            des.append(_des)
        if show:
            self._show(frames, kp, denormalise)

        return kp, des

    @staticmethod
    def _show(frames, kp, denormalise: int = 255) -> None:
        ROWS, COLS = 4, 3
        fig = make_subplots(rows=ROWS, cols=COLS, print_grid=True)
        col, row = 0, 1
        for frame, kp_ in zip(frames, kp):
            if col >= COLS:
                row += 1
                col = 0
            if row > ROWS: break
            col += 1
            img_kp = cv2.drawKeypoints(np.uint8(frame * denormalise), kp_, None)
            fig.add_trace(
                go.Image(z=img_kp),
                col=col,
                row=row,
            )
        fig.show()
        return None