import os
import sys
import uuid
from tqdm import tqdm

import torch
import torch.backends.cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

import robosat_pink as rsp
from robosat_pink.core import load_config, load_module, check_model, check_channels, check_classes, Logs
from robosat_pink.metrics.core import Metrics


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser("train", help="Trains a model on a dataset", formatter_class=formatter_class)
    parser.add_argument("--config", type=str, help="path to config file [required]")

    data = parser.add_argument_group("Dataset")
    data.add_argument("dataset", type=str, help="training dataset path")
    data.add_argument("--loader", type=str, help="dataset loader name [if set override config file value]")
    data.add_argument("--workers", type=int, help="number of pre-processing images workers [default: batch size]")

    hp = parser.add_argument_group("Hyper Parameters [if set override config file value]")
    hp.add_argument("--bs", type=int, help="batch size")
    hp.add_argument("--lr", type=float, help="learning rate")
    hp.add_argument("--ts", type=str, help="tile size")
    hp.add_argument("--nn", type=str, help="neurals network name")
    hp.add_argument("--loss", type=str, help="model loss")
    hp.add_argument("--da", type=str, help="kind of data augmentation")
    hp.add_argument("--dap", type=float, default=1.0, help="data augmentation probability [default: 1.0]")

    mt = parser.add_argument_group("Model Training")
    mt.add_argument("--epochs", type=int, default=10, help="number of epochs to train [default: 10]")
    mt.add_argument("--resume", action="store_true", help="resume model training, if set imply to provide a checkpoint")
    mt.add_argument("--checkpoint", type=str, help="path to a model checkpoint. To fine tune or resume a training")
    mt.add_argument("--no_validation", action="store_true", help="No validation, training only")
    mt.add_argument("--no_training", action="store_true", help="No training, validation only")

    out = parser.add_argument_group("Output")
    out.add_argument("--saving", type=int, default=1, help="number of epochs beetwen checkpoint .pth saving [default: 1]")
    out.add_argument("out", type=str, help="output directory path to save checkpoint .pth files and logs [required]")

    parser.set_defaults(func=main)


def main(args):
    config = load_config(args.config)
    args.out = os.path.expanduser(args.out)
    config["model"]["loader"] = args.loader if args.loader else config["model"]["loader"]
    config["model"]["bs"] = args.bs if args.bs else config["model"]["bs"]
    config["model"]["lr"] = args.lr if args.lr else config["model"]["lr"]
    config["model"]["ts"] = tuple(map(int, args.ts.split(","))) if args.ts else config["model"]["ts"]
    config["model"]["nn"] = args.nn if args.nn else config["model"]["nn"]
    config["model"]["loss"] = args.loss if args.loss else config["model"]["loss"]
    config["model"]["da"] = args.da if args.da else config["model"]["da"]
    config["model"]["dap"] = args.dap if args.dap else config["model"]["dap"]
    args.workers = config["model"]["bs"] if not args.workers else args.workers
    check_classes(config)
    check_channels(config)
    check_model(config)

    assert os.path.isdir(os.path.expanduser(args.dataset)), "Dataset is not a directory"
    if args.no_training and args.no_validation:
        sys.exit()

    log = Logs(os.path.join(args.out, "log"))

    if torch.cuda.is_available():
        log.log("RoboSat.pink - training on {} GPUs, with {} workers".format(torch.cuda.device_count(), args.workers))
        log.log("(Torch:{} Cuda:{} CudNN:{})".format(torch.__version__, torch.version.cuda, torch.backends.cudnn.version()))
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        log.log("RoboSat.pink - training on CPU, with {} workers - (Torch:{})".format(args.workers, torch.__version__))
        log.log("")
        log.log("==========================================================")
        log.log("WARNING: Are you -really- sure about not training on GPU ?")
        log.log("==========================================================")
        log.log("")
        device = torch.device("cpu")

    log.log("--- Input tensor from Dataset: {} ---".format(args.dataset))
    num_channel = 1  # 1-based numerotation
    for channel in config["channels"]:
        for band in channel["bands"]:
            log.log("Channel {}:\t\t {}[band: {}]".format(num_channel, channel["name"], band))
            num_channel += 1

    log.log("--- Output Classes ---")
    for c, classe in enumerate(config["classes"]):
        log.log("Class {}:\t\t {}".format(c, classe["title"]))

    log.log("--- Hyper Parameters ---")
    for hp in config["model"]:
        log.log("{}{}".format(hp.ljust(25, " "), config["model"][hp]))

    loader = load_module("robosat_pink.loaders.{}".format(config["model"]["loader"].lower()))
    loader_train = getattr(loader, config["model"]["loader"])(
        config, config["model"]["ts"], os.path.join(args.dataset, "training"), "train"
    )
    loader_val = getattr(loader, config["model"]["loader"])(
        config, config["model"]["ts"], os.path.join(args.dataset, "validation"), "train"
    )

    model_module = load_module("robosat_pink.models.{}".format(config["model"]["nn"].lower()))

    nn = getattr(model_module, config["model"]["nn"])(loader_train.shape_in, loader_train.shape_out, config).to(device)
    nn = torch.nn.DataParallel(nn)
    optimizer = Adam(nn.parameters(), lr=config["model"]["lr"])

    resume = 0
    if args.checkpoint:
        chkpt = torch.load(os.path.expanduser(args.checkpoint), map_location=device)
        nn.load_state_dict(chkpt["state_dict"])
        log.log("--- Using Checkpoint ---")
        log.log("Path:\t\t {}".format(args.checkpoint))
        log.log("UUID:\t\t {}".format(chkpt["uuid"]))

        if args.resume:
            optimizer.load_state_dict(chkpt["optimizer"])
            resume = chkpt["epoch"]
            assert resume < args.epochs, "Epoch asked, already reached by the given checkpoint"

    loss_module = load_module("robosat_pink.losses.{}".format(config["model"]["loss"].lower()))
    criterion = getattr(loss_module, config["model"]["loss"])().to(device)

    bs = config["model"]["bs"]
    train_loader = DataLoader(loader_train, batch_size=bs, shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(loader_val, batch_size=bs, shuffle=False, drop_last=True, num_workers=args.workers, pin_memory=True)

    if args.no_training:
        process(val_loader, config, log, device, nn, criterion, "eval")
        sys.exit()

    for epoch in range(resume + 1, args.epochs + 1):  # 1-N based
        UUID = uuid.uuid1()
        log.log("---{}Epoch: {}/{} -- UUID: {}".format(os.linesep, epoch, args.epochs, UUID))

        process(train_loader, config, log, device, nn, criterion, "train", optimizer)

        try:  # https://github.com/pytorch/pytorch/issues/9176
            nn_doc = nn.module.doc
            nn_version = nn.module.version
        except AttributeError:
            nn_doc = nn.doc
            nn_version = nn.version

        states = {
            "uuid": UUID,
            "model_version": nn_version,
            "producer_name": "RoboSat.pink",
            "producer_version": rsp.__version__,
            "model_licence": "MIT",
            "domain": "pink.RoboSat",  # reverse-DNS
            "doc_string": nn_doc,
            "shape_in": loader_train.shape_in,
            "shape_out": loader_train.shape_out,
            "state_dict": nn.state_dict(),
            "epoch": epoch,
            "nn": config["model"]["nn"],
            "optimizer": optimizer.state_dict(),
            "loader": config["model"]["loader"],
        }
        checkpoint_path = os.path.join(args.out, "checkpoint-{:05d}.pth".format(epoch))
        if epoch == args.epochs or not (epoch % args.saving):
            log.log("[Saving checkpoint]")
            torch.save(states, checkpoint_path)

        if not args.no_validation:
            process(val_loader, config, log, device, nn, criterion, "eval")


def process(loader, config, log, device, nn, criterion, mode, optimizer=None):
    def _process():
        num_samples = 0
        running_loss = 0
        metrics = Metrics(config["model"]["metrics"], config=config)

        for images, masks, tiles in tqdm(loader, desc=mode.title(), unit="batch", ascii=True):
            images = images.to(device)
            masks = masks.to(device)

            assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"
            num_samples += int(images.size(0))

            if mode == "train":
                optimizer.zero_grad()
            outputs = nn(images)

            assert outputs.size()[2:] == masks.size()[1:], "resolutions for predictions and masks are in sync"
            assert outputs.size()[1] == len(config["classes"]), "classes for predictions and dataset are in sync"

            loss = criterion(outputs, masks, config)
            if mode == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.item()

            for mask, output in zip(masks, outputs):
                if mode == "train":
                    output.detach()
                metrics.add(mask, torch.argmax(output, 0))

        assert num_samples > 0, "dataset contains training images and labels"

        log.log("{}{:.3f}".format("Loss:".ljust(25, " "), running_loss / num_samples))
        for classe in config["classes"][1:]:
            for metric, value in metrics.get().items():
                log.log("{}{:.3f}".format((classe["title"] + " " + metric).ljust(25, " "), value))

    if mode == "train":
        nn.train()
        _process()

    if mode == "eval":
        nn.eval()
        with torch.no_grad():
            _process()
