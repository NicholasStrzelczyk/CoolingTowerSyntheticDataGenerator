import pandas as pd
from tqdm import tqdm

from utils.constants import *


def make_platform_ds_list(data_dir, partition):
    base_path = os.path.join(data_dir, str(partition.value))

    list_name = 'list.txt'
    list_file_path = os.path.join(base_path, list_name)
    open(list_file_path, 'w+').close()  # overwrite/ make new blank file

    # ----- begin generating list ----- #
    list_file = open(list_file_path, "a")

    for scenario in tqdm(range(1, 5), desc='Creating dataset list'):
        scenario_dir = os.path.join(base_path, "scenario_{}".format(scenario))
        total_days = pd.read_csv(os.path.join(scenario_dir, "timeline.csv"))['day'].values[-1]

        for day in range(1, total_days + 1):
            label_name = "LABEL_day_{}.png".format(day)

            for hour in hour_list:
                image_name = "SYNTH_day_{}_{}.png".format(day, hour)

                if os.path.exists(os.path.join(scenario_dir, "images", image_name)):  # skip over missing hours
                    img_path = os.path.join(str(partition.value), "scenario_{}".format(scenario), "images", image_name)
                    tgt_path = os.path.join(str(partition.value), "scenario_{}".format(scenario), "targets", label_name)
                    list_file.write(img_path + " " + tgt_path + "\n")

    list_file.close()


if __name__ == '__main__':
    # ----- hyperparameters ----- #
    part = Partition.TRAIN
    data_dir_path = '/Users/nick_1/Bell_5G_Data/synth_datasets'

    # ----- ----- ----- #
    make_platform_ds_list(data_dir_path, part)

