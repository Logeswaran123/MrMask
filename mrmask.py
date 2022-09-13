import argparse

from utils.mask import Mask
from utils.const import MASK

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--video", required=False, default=0,
                                        help="Path to video source file. Used webcam, if not provided",
                                        type=str)
    parser.add_argument('-f', "--face", required=False, default='single', choices=['single', 'multi'],
                                        help="Single or multi face", type=str)
    parser.add_argument('-m', "--mask", required=False, default=int(-1), choices=range(0, len(MASK)),
                                        help="Choose face mask number", type=int)
    parser.add_argument('-mesh', "--mesh", required=False, default=False,
                                        help="Add face mesh to detected faces (True/False)", type=bool)
    args = parser.parse_args()
    video = args.video
    face = args.face
    mask = args.mask
    mesh = args.mesh
    mask = Mask(video, face, mask, mesh)
    mask.run()