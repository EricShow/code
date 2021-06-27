import glob
import time

import numpy as np
import SimpleITK as sitk
import torch

from .utils.common import get_all_lines, show_center_line
from .utils.line import Line

level_value_interval = 5


def relabelConnectedComponent(im):
    return sitk.RelabelComponent(sitk.ConnectedComponent(im > 0))


def line_process(input_main_component, gpu):
    lines_list, main_part, center_line = get_all_lines(input_main_component, gpu)
    line_mask = np.zeros_like(input_main_component)
    level_label_mask = np.zeros_like(input_main_component)
    second_lines = []
    for line_id, lines in enumerate(lines_list):
        if line_id > 2:
            second_lines.extend(lines)
        for line in lines:
            line.get_color_litter_trick(input_main_component)
            line.paint_color_on_mask(line_mask)
    for line in second_lines:
        line.set_color(10)
        line.paint_color_on_mask(level_label_mask)
    left_line = center_line - (line_mask > 0)
    left_line_colored = input_main_component * left_line
    final_mask = line_mask + left_line_colored
    line_mask = show_center_line(final_mask)
    level_label_mask = show_center_line(level_label_mask)
    total_mask = line_mask * (main_part == 0) + main_part
    return total_mask, level_label_mask, lines_list


if __name__ == '__main__':
    first_path = './record_for_vessel_0908/nrrd/*'
    for file in glob.iglob(first_path):
        input_sitk = sitk.ReadImage(file)
        file_name = file.split('/')[-1]
        input_array = sitk.GetArrayFromImage(input_sitk)
        total_mask = final_process(input_array)
        total_sitk = sitk.GetImageFromArray(total_mask)
        sitk.WriteImage(total_sitk, 'result_%s.nii.gz' % file_name.strip('.nrrd'))
