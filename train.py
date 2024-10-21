import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from model.mamba import Mamba, MambaConfig
import argparse
from utils.util import *
from utils.dataset import StockPrice
from utils.scheduler import *
from utils.logger import create_logger
import os
import datetime
import time
import wandb
from model.liquidnet import LiquidNet


class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
       
        self.proj = LiquidNet(in_dim, args.hidden//2, args.hidden)
        self.mamba = Mamba(self.config)
        self.dense = LiquidNet(args.hidden, args.hidden*2, out_dim)
        self.act = nn.Tanh()
    
    def forward(self,x):
        out = self.proj(x).unsqueeze(1)
        out = self.mamba(out)
        out = self.dense(out)
        out = self.act(out)
        return out.flatten()


def validation(model, test_loader):
    model.eval()
    mse =  0 
    rmse = 0
    mae = 0
    r2 = 0

    logger.info("=============  VALIDATION  =============")
    for i, batch in enumerate(test_loader):
        x, y = batch
        x, y = x.float(), y.float()
        x, y = x.unsqueeze(1), y.squeeze(1)

        if args.cuda:
            x = x.cuda()
            y = y.cuda()

        pred = model(x)

        MSE, RMSE, MAE, R2 = evaluation_metric(pred.cpu().detach(),y.cpu().detach())

        wandb.log({"val/loss": MSE})
        wandb.log({"val/RMSE": RMSE})
        wandb.log({"val/MAE": MAE})
        wandb.log({"val/R2": R2})

        mse += MSE
        rmse += RMSE
        mae += MAE
        r2 += R2

    logger.info("MSE: %.6f || RMSE: %.6f || MAE: %.6f || R2: %.6f" % (mse/i, rmse/i, mae/i, r2/i))



def train(args, model, logger):
    train_set = StockPrice(root=args.data_root, ratio=0.8, mode='train')
    test_set =  StockPrice(root=args.data_root, ratio=0.2, mode='test')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.tr_bs, shuffle=True, 
                                               num_workers=args.workers, drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.val_bs, shuffle=False, 
                                              num_workers=args.workers, drop_last=True) 

    mse = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    t_total = (len(train_set) // args.tr_bs) * args.epochs

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    logger.info("  Instantaneous batch size per GPU = %d", args.tr_bs)

    step_per_epoch = len(train_set) // args.tr_bs

    logger.info("Start training")

    model.zero_grad()
    for epoch in range(args.epochs):
        model.train()

        epoch_time = time.time()
        start_time = time.time()

        all_loss = 0
        rmse_loss = 0
        mae_loss = 0
        r2_loss = 0

        for i, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.float(), y.float()
            x, y = x.unsqueeze(1), y.squeeze(1)

            if args.cuda:
                x = x.cuda()
                y = y.cuda()

            pred = model(x)

            loss = mse(pred, y)

            _, RMSE, MAE, R2 = evaluation_metric(pred.cpu().detach(),y.cpu().detach())

            wandb.log({"train/loss": loss.item()})
            wandb.log({"train/RMSE": RMSE})
            wandb.log({"train/MAE": MAE})
            wandb.log({"train/R2": R2})
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            total_norm = get_grad_norm(model.parameters())

            wandb.log({"train/grad_norm": total_norm})
            wandb.log({"train/lr": scheduler.get_last_lr()[0]})

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            all_loss += loss.item()
            rmse_loss += RMSE
            mae_loss += MAE
            r2_loss += R2

            get_lr = scheduler.get_last_lr()[0]

            if i % args.disp_interval == 0:
                all_loss /= args.disp_interval
                end_time = time.time()

                logger.info('[epoch %2d][iter %4d/%4d]|| Loss: %.6f || RMSE: %.6f || MAE: %.6f || R2: %.6f || lr: %.2e || grad_norm: %.4f || Time: %.2f sec' \
                      % (epoch, i, step_per_epoch, all_loss, rmse_loss, mae_loss, r2_loss, get_lr, total_norm, end_time - start_time))

                all_loss = 0
                rmse_loss = 0
                mae_loss = 0
                r2_loss = 0

                start_time = time.time()

        logger.info('This epoch cost %.4f sec'%(time.time()-epoch_time))

        if (epoch+1) % 10 == 0:
            validation(model, test_loader)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))



parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=1e-5,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,help='Dimension of representations')
parser.add_argument('--layer', type=int, default=64,help='Num of layers')
parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for.")
parser.add_argument('--workers', type=int, default=4,help='Num of workers')
parser.add_argument('--tr_bs', type=int, default=512,help='train bs')
parser.add_argument('--val_bs', type=int, default=4,help='validation bs')
parser.add_argument('--decay_type', type=str, default='cosine',help='decay type')
parser.add_argument('--data_root', type=str, default='/AI/MambaLiquid/data/StockPrice/Data/Stocks',help='dataset root')
parser.add_argument('--disp_interval', default=100, type=int, help='Number of iterations to display')
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--wandb_name", default='training', type=str, help="wandb name")


if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    wandb.init(project="MambaLiquid", name=f'{args.wandb_name}')

    os.makedirs('logs', exist_ok=True)
    logger = create_logger(output_dir='logs', name=f"{str(datetime.datetime.today().strftime('_%d-%m-%H'))}")

    set_seed(args.seed,args.cuda)

    if args.cuda:
        model = Net(in_dim=4, out_dim=1).cuda()
    else:
        model = Net(in_dim=4, out_dim=1)

    logger.info(f"Creating model:{model}")

    train(args,model,logger)