import os
import oyaml as yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
# from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm
# from encoding.parallel import DataParallelModel, DataParallelCriterion
from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict
from tensorboardX import SummaryWriter


def init_seed(manual_seed, en_cudnn=False):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)


def train(cfg, writer, logger):
    # Setup seeds
    init_seed(11733, en_cudnn=False)

    # Setup Augmentations
    train_augmentations = cfg["training"].get("train_augmentations", None)
    t_data_aug = get_composed_augmentations(train_augmentations)
    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])

    t_loader = data_loader(cfg=cfg["data"], mode='train', augmentations=t_data_aug)
    v_loader = data_loader(cfg=cfg["data"], mode='val', augmentations=v_data_aug)

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg["training"]["batch_size"],
                                  num_workers=cfg["training"]["n_workers"],
                                  shuffle=True,
                                  drop_last=True)
    valloader = data.DataLoader(v_loader,
                                batch_size=cfg["validating"]["batch_size"],
                                num_workers=cfg["validating"]["n_workers"])

    logger.info("Using training seting {}".format(cfg["training"]))

    # Setup Metrics
    running_metrics_val = runningScore(t_loader.n_classes,t_loader.unseen_classes)
    
    model_state = torch.load('./runs/deeplabv3p_ade_25unseen/84253/deeplabv3p_ade20k_best_model.pkl')
    running_metrics_val.confusion_matrix = model_state['results']
    score, a_iou = running_metrics_val.get_scores()
    
    pdb.set_trace()
    # Setup Model and Loss
    loss_fn = get_loss_function(cfg["training"])
    logger.info("Using loss {}".format(loss_fn))
    model = get_model(cfg["model"], t_loader.n_classes, loss_fn=loss_fn)

    # Setup optimizer
    optimizer = get_optimizer(cfg["training"], model)

    # Initialize training param
    start_iter = 0
    best_iou = -100.0

    # Resume from checkpoint
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info("Resuming training from checkpoint '{}'".format(cfg["training"]["resume"]))
            model_state = torch.load(cfg["training"]["resume"])["model_state"]
            model.load_state_dict(model_state)
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    # Setup Multi-GPU
    if torch.cuda.is_available():
        model = model.cuda()  # DataParallelModel(model).cuda()
        logger.info("Model initialized on GPUs.")

    time_meter = averageMeter()
    i = start_iter

    embd = t_loader.embeddings
    ignr_idx = t_loader.ignore_index
    embds = embd.cuda()
    while i <= cfg["training"]["train_iters"]:
        for (images, labels) in trainloader:
            images = images.cuda()
            labels = labels.cuda()

            i += 1
            model.train()
            optimizer.zero_grad()

            start_ts = time.time()
            loss_sum = model(images,labels,embds,ignr_idx)
            if loss_sum==0: # Ignore samples contain unseen cat
                continue         # To enable non-transductive learning, set transductive=0 in the config

            loss_sum.backward()

            time_meter.update(time.time() - start_ts)

            optimizer.step()

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_sum.item(),
                    time_meter.avg / cfg["training"]["batch_size"], )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss_sum.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"]["train_iters"]:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                        images_val = images_val.cuda()
                        labels_val = labels_val.cuda()
                        outputs = model(images_val,labels_val,embds,ignr_idx)
                        # outputs = gather(outputs, 0, dim=0)

                        running_metrics_val.update(outputs)

                score, a_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print("{}: {}".format(k, v))
                    logger.info("{}: {}".format(k, v))
                    #writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                #for k, v in class_iou.items():
                #    logger.info("{}: {}".format(k, v))
                #    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)


                if a_iou >= best_iou:
                    best_iou = a_iou
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "best_iou": best_iou,
                        "results": running_metrics_val.confusion_matrix
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)

                running_metrics_val.reset()
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    print(args)
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(logdir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)
