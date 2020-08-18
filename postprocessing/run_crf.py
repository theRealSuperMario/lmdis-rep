from __future__ import print_function

import functools
import glob
import os
import sys
from contextlib import closing
from multiprocessing import Pool
from typing import *
import click
import cv2
import numpy as np
import pandas as pd
import tqdm
import yaml
from tabulate import tabulate

from supermariopy import crf, denseposelib, imageutils


def list_of_dicts2dict_of_lists(list_of_dicts: List[Dict]) -> Dict[Hashable, List]:
    """

    Parameters
    ----------
    list_of_dicts

    Returns
    -------

    Examples
    --------

    list_of_dicts = [{1 : 1}, {1 : 2}]
    list_of_dicts2dict_of_lists(list_of_dicts)
    >>> {1: [1, 2]}

    """
    arregated_keys = set()
    for d in list_of_dicts:
        arregated_keys.update(list(d.keys()))

    dict_of_lists = {k: [] for k in arregated_keys}

    for d in list_of_dicts:
        for k in arregated_keys:
            dict_of_lists[k].append(d[k])

    return dict_of_lists


def batched_keypoints_to_segments(img, keypoints, segmentation_algorithm):
    n_keypoints = keypoints.shape[0]
    MAP = segmentation_algorithm(img, keypoints)
    MAP_colorized = imageutils.make_colors(n_keypoints + 1, with_background=True, background_id=0)[MAP]
    heatmaps = imageutils.keypoints_to_heatmaps(
        img.shape[:2], keypoints, segmentation_algorithm.var
    )
    heatmaps *= heatmaps > 0.8
    heatmaps_rgb = imageutils.colorize_heatmaps(
        heatmaps[np.newaxis, ...], imageutils.make_colors(n_keypoints)
    )

    img_resized = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
    img_resized = imageutils.convert_range(img_resized, [0, 255], [0, 1])
    im_with_keypoints = imageutils.draw_keypoint_markers(img_resized, keypoints, marker_list=[str(i) for i in range(10)] + ["x", "o", "v", "<", ">", "*"], font_scale=1, thickness=4)
    im_with_keypoints = cv2.resize(im_with_keypoints, (img.shape[1], img.shape[1]), cv2.INTER_LINEAR)

    # im_with_keypoints = imageutils.draw_keypoint_markers(
    #     imageutils.convert_range(img, [0, 255], [0, 1]),
    #     keypoints,
    #     marker_list=[str(i) for i in range(10)] + ["x", "o", "v", "<", ">", "*"],
    #     font_scale=0.5,
    # )
    return MAP, MAP_colorized, heatmaps_rgb, im_with_keypoints


def process_batches(t, segmentation_algorithm):
    """Load data from .npy file and infer segmentation
    """
    print("processing batches")
    img_batch, keypoints_batch = t
    func = functools.partial(
        batched_keypoints_to_segments,
        **{"segmentation_algorithm": segmentation_algorithm}
    )
    labels, labels_rgb, heatmaps, ims_with_keypoints = imageutils.np_map_fn(
        lambda x: func(x[0], x[1]), (img_batch, keypoints_batch)
    )
    heatmaps = np.squeeze(heatmaps).astype(np.float32)
    labels_rgb = labels_rgb.astype(np.float32)

    processed_data = {
        "labels": labels,
        "labels_rgb": labels_rgb,
        "heatmaps": heatmaps,
        "ims_with_keypoints": ims_with_keypoints,
    }
    return processed_data


def load_npz(fname):
    print("loading {}".format(fname))
    loaded_data = np.load(fname)
    loaded_data = {key: loaded_data[key] for key in ["data", "encoded_structure_param"]}
    return loaded_data


def get_iuv_files(densepose_csv_path, root, max_n_samples, fname_col="im1"):
    # TODO: Note: the assumption here is that the order items in all_labels is the same as in the iuv files path
    iuv_files = pd.read_csv(densepose_csv_path)
    iuv_files = iuv_files[fname_col].apply(lambda x: os.path.join(root, x))[
        :max_n_samples
    ]
    return iuv_files


def _write_rgb(t):
    im, i, target_dir = t
    cv2.imwrite(
        os.path.join(target_dir, "{:06d}.png".format(i)),
        cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
    )


def write_rgb(img_rgb, target_dir, n_processes=8):
    """
    img_rgb : [N, H, W, 3] shaped array
    target_dir : str
    """

    img_rgb = imageutils.convert_range(img_rgb, [0, 1], [0, 255])

    arg_tuples = list(zip(img_rgb, range(len(img_rgb)), [target_dir] * len(img_rgb)))
    with closing(Pool(n_processes)) as p:
        p.map(_write_rgb, arg_tuples)


def _write_labels(t):
    l, i, output_dir = t
    cv2.imwrite(
        os.path.join(output_dir, "{:06d}.png".format(i)),
        cv2.cvtColor(l, cv2.COLOR_RGB2BGR),
    )


def write_labels(labels, output_dir, colors, n_processes=8):
    """
    labels: [N, H, W] shaped array
    output_dir : str
    colors : [n_classes, 3] shaped array
    """
    labels = imageutils.convert_range(
        colors[labels].astype(np.float32), [0, 1], [0, 255]
    )

    arg_tuples = list(zip(labels, range(len(labels)), [output_dir] * len(labels)))
    with closing(Pool(n_processes)) as p:
        p.map(_write_labels, arg_tuples)


@click.command()
@click.argument("infer-dir")
@click.argument("output-folder")
@click.argument("run-crf-config")
@click.option("--n-processes", default=1)
def main(infer_dir, output_folder, run_crf_config, n_processes):
    img_key = "data"
    mu_key = "encoded_structure_param"
    os.makedirs(output_folder, exist_ok=True)

    with open(run_crf_config, "r") as f:
        config = yaml.load(f)

    segmentation_algorithm_args = config["segmentation_algorithm_args"]
    npz_files = glob.glob(os.path.join(infer_dir, "*.npz"))
    npz_files = sorted(npz_files)

    print("Using files :")
    print(npz_files)

    segmentation_algorithm = crf.SegmentationFromKeypoints(
        **segmentation_algorithm_args
    )

    data = []
    with closing(Pool(n_processes)) as p:
        for outputs in tqdm.tqdm(p.imap(load_npz, npz_files)):
            data.append(outputs)
    data = list_of_dicts2dict_of_lists(data)
    data = {k: np.concatenate(data[k]) for k in [img_key, mu_key]}
    data[mu_key] = (data[mu_key][..., ::-1] - 0.5) * 1.8
    data[img_key] = imageutils.convert_range(data[img_key], [0, 1], [0, 255])

    # data[mu_key] = data[mu_key][:16]
    # data[img_key] = data[img_key][:16]

    process_func = functools.partial(
        process_batches, **{"segmentation_algorithm": segmentation_algorithm}
    )
    tuples = list(
        zip(
            np.array_split(data[img_key], n_processes, 0),
            np.array_split(data[mu_key], n_processes, 0),
        )
    )
    processed_data = []
    with closing(Pool(n_processes)) as p:
        for outputs in tqdm.tqdm(p.imap(process_func, tuples)):
            processed_data.append(outputs)

    print("writing labels")
    labels = np.concatenate([p["labels"] for p in processed_data], 0)
    labels_rgb = np.concatenate([p["labels_rgb"] for p in processed_data], 0)
    heatmaps = np.concatenate([p["heatmaps"] for p in processed_data], 0)
    ims_with_keypoints = np.concatenate(
        [p["ims_with_keypoints"] for p in processed_data], 0
    )

    target_dir = os.path.join(output_folder, "01_keypoints")
    os.makedirs(target_dir, exist_ok=True)
    write_rgb(ims_with_keypoints, target_dir, n_processes)

    target_dir = os.path.join(output_folder, "02_heatmaps")
    os.makedirs(target_dir, exist_ok=True)
    write_rgb(heatmaps, target_dir, n_processes)

    target_dir = os.path.join(output_folder, "03_labels_rgb")
    os.makedirs(target_dir, exist_ok=True)
    write_rgb(labels_rgb, target_dir, n_processes)

    densepose_csv_path = config["densepose_csv_path"]
    data_root = config["data_root"]
    fname_col = config["data_fname_col"]

    iuv_files = get_iuv_files(densepose_csv_path, data_root, len(labels), fname_col)
    iuvs = np.stack([cv2.imread(x, -1) for x in iuv_files], axis=0)[..., 0]

    dp_semantic_remap_dict = config["dp_semantic_remap_dict"]
    dp_new_part_list = sorted(list(dp_semantic_remap_dict.keys()))
    dp_remap_dict = denseposelib.semantic_remap_dict2remap_dict(
        dp_semantic_remap_dict, dp_new_part_list
    )

    iuvs = denseposelib.resize_labels(iuvs, labels.shape[1:])

    remapped_gt_segmentation, remapped_inferred = denseposelib.get_best_segmentation(
        iuvs, labels, dp_remap_dict
    )

    df = pd.DataFrame(columns=["batch_idx"] + dp_new_part_list)

    df = denseposelib.calculate_iou_df(
        remapped_inferred, remapped_gt_segmentation, dp_new_part_list
    )
    df.to_csv(os.path.join(output_folder, "part_ious.csv"), index=False, header=True)
    df_mean = denseposelib.calculate_overall_iou_from_df(df)
    with open(os.path.join(output_folder, "mean_part_ios.csv"), "w") as f:
        print(
            tabulate(df_mean, headers="keys", tablefmt="psql", showindex="never"),
            file=f,
        )

    target_dir = os.path.join(output_folder, "04_compare")
    os.makedirs(target_dir, exist_ok=True)

    background_color = np.array([1, 1, 1])
    colors1 = imageutils.make_colors(config["n_inferred_parts"] + 1, with_background=True, background_id=0)
    colors2 = imageutils.make_colors(len(dp_new_part_list), with_background=True, background_id=dp_new_part_list.index("background"))
    for i, (im1, im2, im3) in enumerate(
        zip(labels, remapped_inferred, remapped_gt_segmentation)
    ):
        canvas = np.concatenate([colors1[im1], colors2[im2], colors2[im3]], 1).astype(
            np.float32
        )
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        fname = os.path.join(target_dir, "{:06d}.png".format(i))
        cv2.imwrite(fname, imageutils.convert_range(canvas, [0, 1], [0, 255]))
        # batches.plot_batch(
        #     imageutils.convert_range(canvas, [0, 1], [-1, 1]), fname, cols=3
        # )

    target_dir = os.path.join(output_folder, "05_remapped_inferred")
    os.makedirs(target_dir, exist_ok=True)
    write_labels(remapped_inferred, target_dir, colors2, n_processes)

    target_dir = os.path.join(output_folder, "06_remapped_labels")
    os.makedirs(target_dir, exist_ok=True)
    write_labels(remapped_gt_segmentation, target_dir, colors2, n_processes)


if __name__ == "__main__":
    main()
