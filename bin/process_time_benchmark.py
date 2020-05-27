import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib import rc

output_folder = "bin/output/"

if __name__ == "__main__":
    # crop_and_pad,transfer_to_gpu,cnn,transfer_to_cpu,interpolation,displacement
    file_name = '2020_05_08_11_53_10_benchmark_doubletracker'
    data = np.loadtxt(os.path.join(output_folder, file_name + '.csv'), delimiter=',').astype(np.int)

    # convert to ms
    avgs = np.average(data, axis=0) / 1000.0
    stds = np.std(data, axis=0) / 1000.0
    loop_sum = np.sum(data, axis=1) / 1000.0
    total_avgs = np.average(loop_sum)
    total_std = np.std(loop_sum)

    print("Avgs: ")
    string = ""
    for avg in avgs:
        string += f" & {np.round(avg, 2)}"
    print(f"{string} & {np.round(total_avgs, 2)}")

    print("Std: ")
    string = ""
    for std in stds:
        string += f" & {np.round(std, 2)}"
    print(f"{string} & {np.round(total_std, 2)}")

    exit()
    # Data to plot
    labels = ["Crop and pad", "Transfer to GPU", "CNN", "Transfer to CPU", "Bicubic interpolation",
              "Position displacement"]
    sizes = avgs
    colors = ['#2D87BB', '#F66D44', '#FEAE65', '#E6F69D', '#AADEA7', '#64C2A6']

    # Plot
    patches, texts, autotexts = plt.pie(sizes, explode=None, labels=labels, colors=colors,
                                        autopct='%.2f%%', pctdistance=0.7, shadow=False, startangle=0,
                                        textprops=dict(color="#2f2f2f"))

    for text in texts:
        text.set_size('12')
        # text.set_fontname('Computer Modern')
    for autotext in autotexts:
        autotext.set_size('12')
        # autotext.set_fontname('Computer Modern')
    for patch in patches:
        patch.set_path_effects([patheffects.PathPatchEffect(edgecolor='#1f1f1f', alpha=0.3),
                                patheffects.Normal()])

    for patch, txt in zip(patches, autotexts):
        # the angle at which the text is located
        ang = (patch.theta2 + patch.theta1) / 2.
        # new coordinates of the text, 0.7 is the distance from the center
        distance = 0.7
        # if patch is narrow enough, move text to new coordinates
        angle = patch.theta2 - patch.theta1
        if angle < 15.:
            distance = 0.74
        if angle < 12.:
            distance = 0.85
        if angle < 10.:
            distance = 0.8
        x = patch.r * distance * np.cos(ang * np.pi / 180)
        y = patch.r * distance * np.sin(ang * np.pi / 180)
        txt.set_position((x, y))

    plt.axis('equal')
    plt.savefig(os.path.join(output_folder, file_name + '.pdf'), bbox_inches='tight', transparent="True",
                pad_inches=0.01)
    plt.show()
