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
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from model.liquidnet import LiquidNet


class Net(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
       
        self.proj = LiquidNet(in_dim, args.hidden, args.hidden)
        self.mamba = Mamba(self.config)
        self.dense = LiquidNet(args.hidden, out_dim, out_dim)
        self.act = nn.Tanhshrink()
    
    def forward(self,x):
        out = self.proj(x) 
        out = self.mamba(out) 
        out = self.dense(out)
        out = self.act(out)
        return out.flatten()


def validation(args, model, x_test, close, data):
    logger.info("=============  VALIDATION  =============")

    x = x_test
    x = x.unsqueeze(0)

    if args.cuda:
        x = x.cuda()

    predictions = model(x)
    predictions = predictions.cpu().detach().numpy().flatten()
    
    time = data['trade_date'][-args.n_test:]
    data1 = close[-args.n_test:]
    finalpredicted_stock_price = []
    pred = close[-args.n_test-1]
    for i in range(args.n_test):
        pred = close[-args.n_test-1+i]*(1+predictions[i])
        finalpredicted_stock_price.append(pred)
    
    MSE, RMSE, MAE, R2 = evaluation_metric(data1, finalpredicted_stock_price)

    wandb.log({"val/loss": MSE})
    wandb.log({"val/RMSE": RMSE})
    wandb.log({"val/MAE": MAE})
    wandb.log({"val/R2": R2})

    logger.info("MSE: %.6f || RMSE: %.6f || MAE: %.6f || R2: %.6f" % (MSE, RMSE, MAE, R2))

    dateinf(data['trade_date'],args.n_test)
    print('MSE RMSE MAE R2')
    evaluation_metric(data1, finalpredicted_stock_price)
    plt.figure(figsize=(10, 6))
    plt.plot(time, data1, label='Stock Price')
    plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.savefig(f"result/{args.data_name}.pdf", format='pdf', bbox_inches='tight')
    # plt.show()


def train(args, model, logger):
    dataset = StockPrice(args.root, args.data_file, args.n_test)

    x_train, y_train, x_test, close, data = dataset.get_data()

    mse = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.wd)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    logger.info("Start training")

    model.zero_grad()
    
    loss_tot = []
    for epoch in range(args.epochs):
        model.train()
        
        epoch_time = time.time()
        start_time = time.time()
        
        x, y = x_train, y_train
        x = x.unsqueeze(0)

        if args.cuda:
            x = x.cuda()
            y = y.cuda()

        pred = model(x)

        loss = mse(pred, y)

        loss_tot.append(loss.item())
        
        _, RMSE, MAE, _ = evaluation_metric(pred.cpu().detach(),y.cpu().detach())

        wandb.log({"train/loss": loss.item()})
        wandb.log({"train/RMSE": RMSE})
        wandb.log({"train/MAE": MAE})
        
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        total_norm = get_grad_norm(model.parameters())

        wandb.log({"train/grad_norm": total_norm})

        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"train/lr": current_lr})
        
        end_time = time.time()

        logger.info('[epoch %2d]|| Loss: %.6f || RMSE: %.6f || MAE: %.6f || grad_norm: %.4f || Time: %.2f sec' \
                % (epoch, loss.item(), RMSE, MAE, total_norm, end_time - start_time))

        logger.info('This epoch cost %.4f sec'%(time.time()-epoch_time))
    
    
    plt.figure(figsize=(8, 5))
    plt.plot(loss_tot, linestyle='-', label='Loss Value')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("result/loss_curve.pdf", format='pdf', bbox_inches='tight')


    validation(args, model, x_test, close, data)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))



parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=1e-4,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,help='Dimension of representations')
parser.add_argument('--layer', type=int, default=8,help='Num of layers')
parser.add_argument('--workers', type=int, default=4,help='Num of workers')
parser.add_argument('--data_root', type=str, default='/AI/MambaLiquid/data/StockPrice/Data/Stocks',help='dataset root')
parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
parser.add_argument("--root", default='data', type=str, help="wandb name")
parser.add_argument("--data_file", default='601988', type=str, help="data_file name")
parser.add_argument("--wandb_name", default='training', type=str, help="wandb name")
parser.add_argument('--n_test', type=int, default=300,help='Num of test')
parser.add_argument('--in_dim', type=int, default=15,help='input dimension')
parser.add_argument("--data_name", default='Bank of China', type=str, help="dataset name")


if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    wandb.init(project="MambaLiquid", name=f'{args.wandb_name}')

    os.makedirs('logs', exist_ok=True)
    logger = create_logger(output_dir='logs', name=f"{str(datetime.datetime.today().strftime('_%d-%m-%H'))}")

    set_seed(args.seed,args.cuda)

    if args.cuda:
        model = Net(in_dim=args.in_dim, out_dim=1).cuda()
    else:
        model = Net(in_dim=args.in_dim, out_dim=1)

    logger.info(f"Creating model:{model}")

    train(args,model,logger)