from cProfile import label
import pickle
import numpy as np
import matplotlib.pyplot as plt

from os.path import join, dirname, realpath

location = dirname(realpath(__file__))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

OBJECT_TYPES = [
    {
        "type": "cylinder",
        "size": [0.04, 0.04, 0.05],
        "names": ["cylinder"],  # , "tube", "can", "tin"]
    },
    {
        "type": "cube",
        "size": [0.05, 0.05, 0.05],
        "names": ["cube"],  # , "block"]
    },
    {
        "type": "box",
        "size": [0.03, 0.05, 0.06],
        "names": ["box"],  # , "cuboid"]
    },
    {
        "type": "sphere",
        "size": [0.05, 0.05, 0.05],
        "names": ["sphere"],  # , "ball", "orb", "globe"]
    },
    {"type": "slice", "size": [0.05, 0.05, 0.02], "names": ["slice"]},
]

COLOURS = {
    "white": [1.0, 1.0, 1.0],
    "grey": [0.5, 0.5, 0.5],
    "black": [0.0, 0.0, 0.0],
    "red": [1.0, 0.1, 0.1],
    "green": [0.1, 0.7, 0.1],
    "blue": [0.1, 0.1, 1.0],
    "yellow": [1.0, 1.0, 0.1],
}


# names = ["Grab", "Place on", "Place by", "Place in"]
# bar1 = np.arange((len(names)))

# with open(f"model/models/eval_info/grab/{name}_ob.npy", "rb") as f:
#         r1 = np.load(f)
#         r1 = 100 * np.sum(r1, axis=0)/np.sum(r1)

# with open(f"model/models/eval_info/place_on/{name}_ob.npy", "rb") as f:
#         r2 = np.load(f)
#         r2 = 100 * np.sum(r2, axis=0)/np.sum(r2)

# with open(f"model/models/eval_info/place_by/{name}_ob.npy", "rb") as f:
#         r3 = np.load(f)
#         r3 = 100 * np.sum(r3, axis=0)/np.sum(r3)

# with open(f"model/models/eval_info/place_in/{name}_ob.npy", "rb") as f:
#         r4 = np.load(f)
#         r4 = 100 * np.sum(r4, axis=0)/np.sum(r4)

# r = np.stack([r1, r2, r3, r4]).T
# print(r)

# plt.bar([i-0.2 for i in bar1], r[0], 0.2,label="Success")
# plt.bar(bar1, r[1], 0.2, label="Wrong object/Wrong command")
# plt.bar([i+0.2 for i in bar1], r[2], 0.2, label="Failure")
# plt.xticks(bar1, names)
# plt.ylim(0, 100)
# plt.legend()
# plt.ylabel("(%)")
# plt.show()

task = "place_by"
description = "Fine tuned BERT"
name = "pre_bert"

names = ["pre_bert", "full_bert", "std_bert"]
descriptions = ["Fine tuned BERT", "In model BERT", "Base BERT"]

for i, name in enumerate(names):
    description = descriptions[i]
    with open(f"model/models/eval_info/{task}/{name}_ob.npy", "rb") as f:
        r = np.load(f)
    r = 100 * r / np.sum(r, axis=1)[:, None]
    objs = [obj["type"].capitalize() for obj in OBJECT_TYPES] + [
        "Plate",
        "Bin",
    ]
    df_cm = pd.DataFrame(
        r, index=objs, columns=["Success", "To side", "Failed/Wrong"],
    )
    plt.figure(figsize=(10, 7))
    sn.heatmap(
        df_cm,
        annot=True,
        cmap="Reds",
        vmin=0,
        vmax=100,
        fmt=".2f",
        cbar=False,
        annot_kws={"size": 14},
    )
    plt.title(f"Per shape accuracy (%) for {description}")
    plt.savefig(
        join(location, "imgs", f"{task}_{name}_objs.png"), bbox_inches="tight"
    )

    with open(f"model/models/eval_info/{task}/{name}_col.npy", "rb") as f:
        r = np.load(f)

    r = 100 * r / np.sum(r, axis=1)[:, None]
    cols = [x.capitalize() for x in list(COLOURS.keys())]
    df_cm = pd.DataFrame(
        r, index=cols, columns=["Success", "Wrong Object", "Failed"]
    )
    plt.figure(figsize=(10, 7))
    sn.heatmap(
        df_cm,
        annot=True,
        cmap="Reds",
        vmin=0,
        vmax=100,
        fmt=".2f",
        cbar=False,
        annot_kws={"size": 14},
    )
    plt.title(f"Per colour accuracy (%) for {description}")
    plt.savefig(
        join(location, "imgs", f"{task}_{name}_col.png"), bbox_inches="tight"
    )

# file_loc = join(location, "models", "run_info")

# with open("model/models/run_info/multi_data_750_train.ob", "rb") as f:
#         train_50 = pickle.load(f)
# with open("model/models/run_info/multi_data_1500_train.ob", "rb") as f:
#         train_100 = pickle.load(f)
# with open("model/models/run_info/single_data_250_train.ob", "rb") as f:
#         train_250 = pickle.load(f)
# with open("model/models/run_info/single_data_500_train.ob", "rb") as f:
#         train_500 = pickle.load(f)

# plt.plot(train_50[1:], label="750 demonstrations")
# plt.plot(train_100[1:], label="1500 demonstrations")
# plt.plot(train_250[1:], label="250 demonstrations")
# plt.plot(train_500[1:], label="500 demonstrations")
# plt.xlabel("Epochs")
# plt.ylabel("MSE Loss")
# plt.title("Train loss of models trained with a different number of demonstrations")
# plt.yscale("log")
# plt.legend()
# plt.show()


# with open("model/models/run_info/single_data_50_07-06-2022-23-19_val.ob", "rb") as f:
#         val_50 = pickle.load(f)
# with open("model/models/run_info/single_data_100_07-06-2022-23-28_val.ob", "rb") as f:
#         val_100 = pickle.load(f)
# with open("model/models/run_info/single_data_250_07-06-2022-23-36_val.ob", "rb") as f:
#         val_250 = pickle.load(f)
# with open("model/models/run_info/single_data_500_08-06-2022-00-07_val.ob", "rb") as f:
#         val_500 = pickle.load(f)

# # plt.plot(val_50[1:], label="50")
# # plt.plot(val_100[1:], label="100")
# # plt.plot(val_250[1:], label="250")
# # plt.plot(val_500[1:], label="500")
# # plt.yscale("log")
# # plt.legend()
# # plt.show()
