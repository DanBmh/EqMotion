import argparse
import copy
import torch
from h36m.model_t import EqMotion
import os
from torch import nn, optim
import time
import numpy as np
import random

import sys
import tqdm

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

# datamode = "gt-gt"
datamode = "pred-gt"
# datamode = "pred-pred"

config_sk = {
    # "item_step": 2,
    # "window_step": 2,
    "item_step": 1,
    "window_step": 1,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        "nose",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        "shoulder_middle",
    ],
}

datasets_train = [
    "/datasets/preprocessed/human36m/train_forecast_kppspose_10fps.json",
    # "/datasets/preprocessed/human36m/train_forecast_kppspose.json",
]

# datasets_train = [
#     "/datasets/preprocessed/mocap/train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/bmlmovi_train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/bmlrub_train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/kit_train_forecast_samples_10fps.json"
# ]

# datasets_train = [
#     "/datasets/preprocessed/mocap/train_forecast_samples_4fps.json",
#     "/datasets/preprocessed/amass/bmlmovi_train_forecast_samples_4fps.json",
#     "/datasets/preprocessed/amass/bmlrub_train_forecast_samples_4fps.json",
#     "/datasets/preprocessed/amass/kit_train_forecast_samples_4fps.json"
# ]

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_4fps.json"


num_joints = len(config_sk["select_joints"])
in_features = num_joints * 3
dim_used = list(range(in_features))

# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, scale):
    sequences = utils_pipeline.make_input_sequence(batch, split, datamode)

    # Convert to decimeters
    sequences = sequences / scale

    return sequences


def calc_delta(all_seqs):
    all_seqs_vel = np.zeros_like(all_seqs)
    all_seqs_vel[:, :, 1:] = all_seqs[:, :, 1:] - all_seqs[:, :, :-1]
    all_seqs_vel[:, :, 0] = all_seqs_vel[:, :, 1]

    return all_seqs_vel


# ==================================================================================================


parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument(
    "--exp_name", type=str, default="exp_1", metavar="N", help="experiment_name"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=80,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--past_length",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--future_length",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=-1, metavar="S", help="random seed (default: -1)"
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=1,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--test_interval",
    type=int,
    default=1,
    metavar="N",
    help="how many epochs to wait before logging test",
)
parser.add_argument(
    "--outf",
    type=str,
    default="n_body_system/logs",
    metavar="N",
    help="folder to output vae",
)
parser.add_argument("--lr", type=float, default=5e-4, metavar="N", help="learning rate")
parser.add_argument(
    "--epoch_decay",
    type=int,
    default=2,
    metavar="N",
    help="number of epochs for the lr decay",
)
parser.add_argument(
    "--lr_gamma", type=float, default=0.8, metavar="N", help="the lr decay ratio"
)
parser.add_argument("--nf", type=int, default=64, metavar="N", help="learning rate")
parser.add_argument(
    "--n_layers",
    type=int,
    default=4,
    metavar="N",
    help="number of layers for the autoencoder",
)
parser.add_argument(
    "--channels", type=int, default=72, metavar="N", help="number of channels"
)
parser.add_argument(
    "--dataset", type=str, default="nbody", metavar="N", help="nbody_small, nbody"
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-12, metavar="N", help="timing experiment"
)
parser.add_argument(
    "--div", type=float, default=1, metavar="N", help="timing experiment"
)
parser.add_argument(
    "--norm_diff", type=eval, default=False, metavar="N", help="normalize_diff"
)
parser.add_argument("--tanh", type=eval, default=False, metavar="N", help="use tanh")
parser.add_argument(
    "--model_save_dir",
    type=str,
    default="h36m/saved_models",
    help="Path to save models",
)
parser.add_argument("--scale", type=float, default=100, metavar="N", help="data scale")
parser.add_argument(
    "--model_name", type=str, default="ckpt_short", help="Name of the model."
)
parser.add_argument("--weighted_loss", action="store_true")
parser.add_argument("--apply_decay", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--add_agent_token", action="store_true")
parser.add_argument("--category_num", type=int, default=4)
parser.add_argument("--test", action="store_true")
parser.add_argument(
    "--model_weights_path",
    type=str,
    default="",
    help="directory with the model weights to copy",
)

time_exp_dic = {"time": 0, "counter": 0}

args = parser.parse_args()
args.cuda = True
args.add_agent_token = True
if args.future_length == 25:
    args.weighted_loss = True

device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def lr_decay(optimizer, lr_now, gamma):
    lr_new = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_new
    return lr_new


def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)
    print("The seed is :", seed)

    config_sk["input_n"] = args.past_length
    config_sk["output_n"] = args.future_length

    # Load preprocessed datasets
    print("Loading datasets ...")
    if not args.test:
        dataset_train, dlen_train = [], 0
        for dp in datasets_train:
            cfg = copy.deepcopy(config_sk)
            if "mocap" in dp:
                cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"

            ds, dlen = utils_pipeline.load_dataset(dp, "train", cfg)
            dataset_train.extend(ds["sequences"])
            dlen_train += dlen
        esplit = "test" if "mocap" in dataset_eval_test else "eval"
        cfg = copy.deepcopy(config_sk)
        if "mocap" in dataset_eval_test:
            cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
        dataset_eval, dlen_eval = utils_pipeline.load_dataset(
            dataset_eval_test, esplit, cfg
        )
        dataset_eval = dataset_eval["sequences"]
    cfg = copy.deepcopy(config_sk)
    if "mocap" in dataset_eval_test:
        cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
    dataset_test, dlen_test = utils_pipeline.load_dataset(
        dataset_eval_test, "test", cfg
    )
    dataset_test = dataset_test["sequences"]

    # dataset_test, dlen_test = utils_pipeline.load_dataset(
    #     datapath_preprocessed, "eval", config_sk
    # )
    # dataset_train, dlen_train = utils_pipeline.load_dataset(
    #     datapath_preprocessed, "eval", config_sk
    # )
    # dataset_eval, dlen_eval = utils_pipeline.load_dataset(
    #     datapath_preprocessed, "eval", config_sk
    # )

    model = EqMotion(
        in_node_nf=args.past_length,
        in_edge_nf=2,
        hidden_nf=args.nf,
        in_channel=args.past_length,
        hid_channel=args.channels,
        out_channel=args.future_length,
        device=device,
        n_layers=args.n_layers,
        recurrent=True,
        norm_diff=args.norm_diff,
        tanh=args.tanh,
        add_agent_token=args.add_agent_token,
        n_agent=num_joints,
        category_num=args.category_num,
    )

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}

    print(get_parameter_number(model))

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.test:
        model_path = args.model_save_dir + "/" + args.model_name + "_best.pth.tar"
        print("Loading model from:", model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt["state_dict"], strict=False)
        epoch = 0

        label_gen_test = utils_pipeline.create_labels_generator(dataset_test, config_sk)

        mpjpe = test(
            model,
            optimizer,
            epoch,
            label_gen_test,
            dim_used,
            backprop=False,
            dlen=dlen_test,
        )

        return

    if args.model_weights_path != "":
        print("Loading model weights from:", args.model_weights_path)
        model.load_state_dict(torch.load(args.model_weights_path), strict=False)

    best_eval_loss = 1e8
    best_epoch = 0
    lr_now = args.lr

    for epoch in range(0, args.epochs):
        label_gen_train = utils_pipeline.create_labels_generator(
            dataset_train, config_sk
        )
        label_gen_eval = utils_pipeline.create_labels_generator(dataset_eval, config_sk)

        if args.apply_decay:
            if epoch % args.epoch_decay == 0 and epoch > 0:
                lr_now = lr_decay(optimizer, lr_now, args.lr_gamma)

        train(model, optimizer, epoch, label_gen_train, dim_used, dlen=dlen_train)

        if epoch % args.test_interval == 0:
            mpjpe = test(
                model,
                optimizer,
                epoch,
                label_gen_eval,
                dim_used,
                backprop=False,
                dlen=dlen_eval,
            )
            avg_mpjpe = np.mean(mpjpe)

            if avg_mpjpe < best_eval_loss:
                best_eval_loss = avg_mpjpe
                best_epoch = epoch
                state = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                file_path = (
                    args.model_save_dir + "/" + args.model_name + "_best.pth.tar"
                )
                torch.save(state, file_path)

            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            file_path = (
                args.model_save_dir
                + "/"
                + args.model_name
                + "_"
                + str(epoch)
                + ".pth.tar"
            )
            torch.save(state, file_path)

            print(
                "Best Eval Loss: %.5f \t Best epoch %d" % (best_eval_loss, best_epoch)
            )
            print("The seed is :", seed)

    return


def train(model, optimizer, epoch, data_loader, dim_used=[], backprop=True, dlen=0):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {"epoch": epoch, "loss": 0, "coord_reg": 0, "counter": 0}

    nbatch = args.batch_size
    batch_size = nbatch
    for batch in tqdm.tqdm(
        utils_pipeline.batch_iterate(data_loader, batch_size=nbatch),
        total=int(dlen / nbatch),
    ):
        sequences_train = prepare_sequences(batch, nbatch, "input", args.scale)
        sequences_gt = prepare_sequences(batch, nbatch, "target", args.scale)

        augment = True
        if augment:
            sequences_train, sequences_gt = utils_pipeline.apply_augmentations(
                sequences_train, sequences_gt
            )

        sequences_train = sequences_train.transpose([0, 2, 1, 3])
        sequences_gt = sequences_gt.transpose([0, 2, 1, 3])
        seq_train_vel = calc_delta(sequences_train)

        loc = torch.from_numpy(sequences_train.astype(np.float32)).to(device)
        vel = torch.from_numpy(seq_train_vel.astype(np.float32)).to(device)
        loc_end = torch.from_numpy(sequences_gt.astype(np.float32)).to(device)

        optimizer.zero_grad()

        nodes = torch.sqrt(torch.sum(vel**2, dim=-1)).detach()
        loc_pred, category = model(nodes, loc.detach(), vel)

        if args.weighted_loss:
            weight = np.arange(1, 5, (4 / args.future_length))
            weight = args.future_length / weight
            # weight = weight / np.sum(weight)
            weight = torch.from_numpy(weight).type_as(loc_end)
            weight = weight[None, None]
            loss = torch.mean(weight * torch.norm(loc_pred - loc_end, dim=-1))
        else:
            loss = torch.mean(torch.norm(loc_pred - loc_end, dim=-1))

        if backprop:
            loss.backward()
            optimizer.step()
        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print(
        "%s epoch %d avg loss: %.5f"
        % (prefix + "train", epoch, res["loss"] / res["counter"])
    )

    return res["loss"] / res["counter"]


def test(model, optimizer, epoch, data_loader, dim_used=[], backprop=False, dlen=0):
    model.eval()

    res = {"epoch": epoch, "loss": 0, "coord_reg": 0, "counter": 0, "ade": 0}

    output_n = args.future_length
    eval_frame = list(range(output_n))
    t_3d = np.zeros(len(eval_frame))

    nbatch = args.batch_size
    batch_size = nbatch
    with torch.no_grad():
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(data_loader, batch_size=nbatch),
            total=int(dlen / nbatch),
        ):
            sequences_train = prepare_sequences(batch, nbatch, "input", args.scale)
            sequences_gt = prepare_sequences(batch, nbatch, "target", args.scale)

            sequences_train = sequences_train.transpose([0, 2, 1, 3])
            seq_train_vel = calc_delta(sequences_train)

            loc = torch.from_numpy(sequences_train.astype(np.float32)).to(device)
            vel = torch.from_numpy(seq_train_vel.astype(np.float32)).to(device)
            loc_end = torch.from_numpy(sequences_gt.astype(np.float32)).to(device)

            nodes = torch.sqrt(torch.sum(vel**2, dim=-1)).detach()
            loc_pred, _ = model(nodes, loc.detach(), vel)

            loc_pred = loc_pred.transpose(1, 2)
            loc_pred = loc_pred.contiguous().view(batch_size, output_n, in_features)

            pred_p3d = loc_pred.contiguous().view(
                batch_size, output_n, -1, 3
            )  # [:, input_n:, :, :]
            targ_p3d = loc_end.contiguous().view(
                batch_size, output_n, -1, 3
            )  # [:, input_n:, :, :]

            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += (
                    torch.mean(
                        torch.norm(
                            targ_p3d[:, j, :, :].contiguous().view(-1, 3)
                            - pred_p3d[:, j, :, :].contiguous().view(-1, 3),
                            2,
                            1,
                        )
                    ).item()
                    * batch_size
                )

            res["counter"] += batch_size

    t_3d *= args.scale
    N = res["counter"]
    t_3d = t_3d / N
    print(t_3d)

    return t_3d


if __name__ == "__main__":
    stime = time.time()
    main()
    ftime = time.time()
    print("Script took {} seconds".format(int(ftime - stime)))
