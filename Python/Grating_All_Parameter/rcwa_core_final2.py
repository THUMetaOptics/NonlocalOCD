from __future__ import annotations

from types import SimpleNamespace as SN
from typing import Tuple

from rcwa_procedure_final2 import rcwa_procedure, trapezoidal_grating_setup


def help_control_file(
    para: SN,
    number_of_orders: int,
    matrix_algorithm: int,
    change_matrix_base: int,
    polarization: int,
    faktorization: int,
) -> Tuple:
    if int(para.grating.grating_type) != 7:
        raise NotImplementedError('This streamlined implementation only keeps grating_type == 7.')
    layer_thickness, duty_cycle, shift = trapezoidal_grating_setup(para)
    para.grating.layer_thickness = layer_thickness
    para.grating.duty_cycle = duty_cycle
    para.grating.shift = shift
    return rcwa_procedure(
        para=para,
        number_of_orders=number_of_orders,
        change_matrix_base=change_matrix_base,
        polarization=polarization,
        faktorization=faktorization,
        matrix_algorithm=matrix_algorithm,
    )


def simu_rcwa1d_main(para: SN):
    number_of_orders = int(para.basic.number_of_orders)
    R_s, R_p = help_control_file(
        para=para,
        number_of_orders=number_of_orders,
        matrix_algorithm=1,
        change_matrix_base=1,
        polarization=0,
        faktorization=1,
    )
    mid_order = (number_of_orders - 1) // 2
    Rs_0 = R_s[mid_order]
    Rp_0 = (1j / para.grating.n1) * R_p[mid_order]
    return Rs_0, Rp_0
