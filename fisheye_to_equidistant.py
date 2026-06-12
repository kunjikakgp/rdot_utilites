import cv2 as cv
import numpy as np

# ============================================================
# CALIBRATION
# ============================================================

u0 = 357.65625
v0 = 361.171875

a0 = 2.34020848e+02
a2 = -1.32921571e-03
a3 = -7.69024901e-07

f_eq = 206.26
r_max = 360.0

# ============================================================
# FISHEYE POLYNOMIAL
# ============================================================

def poly(rho):

    return (
        a0
        + a2 * rho**2
        + a3 * rho**3
    )

# ============================================================
# SOLVE
#
# rho = tan(theta)*poly(rho)
#
# Newton iteration
# ============================================================

def solve_rho(theta):

    t = np.tan(theta)

    rho = a0 * t

    for _ in range(20):

        fr = poly(rho)

        F = rho - t * fr

        dF = (
            1.0
            - t * (
                2*a2*rho
                + 3*a3*rho*rho
            )
        )

        step = F / dF

        rho -= step

        if abs(step) < 1e-6:
            break

    return rho

# ============================================================
# BUILD REMAP TABLES
# ============================================================

def build_maps(
    out_w,
    out_h,
    r_max,
    f_eq
):

    cx = out_w / 2.0
    cy = out_h / 2.0

    map_x = np.zeros(
        (out_h, out_w),
        dtype=np.float32
    )

    map_y = np.zeros(
        (out_h, out_w),
        dtype=np.float32
    )

    for y in range(out_h):

        for x in range(out_w):

            # ----------------------------------
            # equidistant radius
            # ----------------------------------

            dx = x - cx
            dy = y - cy

            r_eq = np.sqrt(
                dx*dx +
                dy*dy
            )
            if r_eq > r_max:

                map_x[y,x] = -1
                map_y[y,x] = -1
                continue

            # ----------------------------------
            # theta
            # ----------------------------------

            theta = r_eq / f_eq

            # ----------------------------------
            # center pixel
            # ----------------------------------

            if r_eq < 1e-12:

                map_x[y, x] = u0
                map_y[y, x] = v0
                continue

            # ----------------------------------
            # recover fisheye radius
            # ----------------------------------

            rho = solve_rho(theta)

            # ----------------------------------
            # radial direction
            # ----------------------------------

            ux = dx / r_eq
            uy = dy / r_eq

            # ----------------------------------
            # fisheye pixel
            # ----------------------------------

            u = u0 + rho * ux
            v = v0 + rho * uy

            map_x[y, x] = u
            map_y[y, x] = v

    return map_x, map_y

# ============================================================
# MAIN
# ============================================================
if __name__=="main":
    img = cv.imread("/home/kunjika/recondash_kunjika/3D_reconstruction_Fisheye-main/rdot_images/pillar.jpg")

    H = 720
    W = 720

    map_x, map_y = build_maps(
        W,
        H
    )

    equidistant = cv.remap(
        img,
        map_x,
        map_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT
    )

    cv.imshow("fisheye", img)
    cv.imshow("equidistant", equidistant)

    cv.waitKey(0)
    cv.destroyAllWindows()