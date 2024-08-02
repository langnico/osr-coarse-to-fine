import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import wandb

from utils.arpl_utils import AverageMeter
from loss.SoftmaxMultilabel import SoftmaxMultilabel
from loss.SoftmaxMultilabelGRL import SoftmaxMultilabelGRL
from datasets.open_set_datasets import parse_batch_dict_or_tuple


def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()

    loss_all = 0
    for batch_idx, batch_data in enumerate(tqdm(trainloader)):
        # allow the option to return dictionaries
        data, labels, idx = parse_batch_dict_or_tuple(batch_data)

        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data, True)
            if isinstance(criterion, (SoftmaxMultilabel)):
                logits, loss = criterion(x, y, labels, batch_data)
            elif isinstance(criterion, (SoftmaxMultilabelGRL)):
                logits, loss, alpha = criterion(x, y, labels, batch_data, progress=epoch/options["max_epoch"])
                wandb.log({'progress': epoch/options["max_epoch"]}, step=epoch)
                wandb.log({'alpha': alpha}, step=epoch)
            else:
                logits, loss = criterion(x, y, labels)
            
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), data.size(0))
        
        loss_all += losses.avg

    print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

    return loss_all

