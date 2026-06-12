import cv2 as cv
import numpy as np

from fisheye_to_equidistant import build_maps
from camera_projections import fisheye_perspective_map

# ============================================================
# CALIBRATION
# ============================================================

F_EQ = 206.26
R_MAX = 360.0
hfov = 120.0
def get_virtual_K():
    return K.copy()
# ============================================================
# BUILD MAPS ONCE
# ============================================================

EQ_MAP_X, EQ_MAP_Y = build_maps(
    out_w=720,
    out_h=720,
    r_max=R_MAX,
    f_eq=F_EQ
)

P0_X, P0_Y,K = fisheye_perspective_map(
    src_w=720,
    src_h=720,
    lens_fov_deg=200.0,
    perspective_hfov_deg=hfov,
    view_azimuth_deg=0.0,
    camera_yaw_offset_deg=0.0
)

P90_X, P90_Y,_ = fisheye_perspective_map(
    src_w=720,
    src_h=720,
    lens_fov_deg=200.0,
    perspective_hfov_deg=hfov,
    view_azimuth_deg=90.0,
    camera_yaw_offset_deg=0.0
)

P180_X, P180_Y,_ = fisheye_perspective_map(
    src_w=720,
    src_h=720,
    lens_fov_deg=200.0,
    perspective_hfov_deg=hfov,
    view_azimuth_deg=180.0,
    camera_yaw_offset_deg=0.0
)

P270_X, P270_Y,_= fisheye_perspective_map(
    src_w=720,
    src_h=720,
    lens_fov_deg=200.0,
    perspective_hfov_deg=hfov,
    view_azimuth_deg=270.0,
    camera_yaw_offset_deg=0.0
)
def crop_black_blob(img,thresh=5):

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # # black pixels
    # mask = gray > 5

    # ys, xs = np.where(mask)

    # if len(ys) == 0:
    #     return img

    # # highest valid row
    # y_cut = ys.max()

    # return img[:y_cut+1, :]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    h, w = gray.shape

    center_x = w // 2

    # scan upward along center column
    for y in range(h):

        if gray[y, center_x] < thresh:
            return img[:y, :]

    return img
# ============================================================
# PROCESS FRAME
# ============================================================

def preprocess_frame(frame):

    # --------------------------------------------------------
    # OCam fisheye -> equidistant fisheye
    # --------------------------------------------------------

    equidistant = cv.remap(
        frame,
        EQ_MAP_X,
        EQ_MAP_Y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT
    )

    # --------------------------------------------------------
    # Equidistant -> perspective views
    # --------------------------------------------------------

    view0 = cv.remap(
        equidistant,
        P0_X,
        P0_Y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT
    )

    view90 = cv.remap(
        equidistant,
        P90_X,
        P90_Y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT
    )

    view180 = cv.remap(
        equidistant,
        P180_X,
        P180_Y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT
    )

    view270 = cv.remap(
        equidistant,
        P270_X,
        P270_Y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT
    )
    v0   = crop_black_blob(view0)
    v90  = crop_black_blob(view90)
    v180 = crop_black_blob(view180)
    v270 = crop_black_blob(view270)

    # h = 300
    # w = 450
    # #+----------------Visualisation----------------+
    # # |      0°        |      90°       |
    # # +----------------+----------------+
    # # |     180°       |     270°       |
    # # +----------------+----------------+

    # v0s   = cv.resize(v0,   (w, h))
    # v90s  = cv.resize(v90,  (w, h))
    # v180s = cv.resize(v180, (w, h))
    # v270s = cv.resize(v270, (w, h))

    # top = np.hstack([v0s, v90s])
    # bottom = np.hstack([v180s, v270s])

    # grid = np.vstack([top, bottom])

    # cv.imshow("All Views", grid)

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return v0, v90, v180, v270

# ============================================================
# TEST
# ============================================================
if __name__=="__main__":
    img = cv.imread(r"/home/kunjika/recondash_kunjika/3D_reconstruction_Fisheye-main/rdot_images/pillar.jpg")

    v0, v90, v180, v270 = preprocess_frame(img)
    
    # resize all views

    h = 300
    w = 450
    #+----------------Visualisation----------------+
    # |      0°        |      90°       |
    # +----------------+----------------+
    # |     180°       |     270°       |
    # +----------------+----------------+

    v0s   = cv.resize(v0,   (w, h))
    v90s  = cv.resize(v90,  (w, h))
    v180s = cv.resize(v180, (w, h))
    v270s = cv.resize(v270, (w, h))

    top = np.hstack([v0s, v90s])
    bottom = np.hstack([v180s, v270s])

    grid = np.vstack([top, bottom])

    cv.imshow("All Views", grid)

    cv.waitKey(0)
    cv.destroyAllWindows()