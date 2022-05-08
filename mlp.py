import argparse
import json
import logging
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Baseado em https://github.com/aws/amazon-sagemaker-examples
class MLP(nn.Module):
    def __init__(self, n_attrs):
        super(MLP, self).__init__()

        # Sequencia de tarefas da rede
        self.mlp_stack = nn.Sequential(            
            nn.Linear(n_attrs, 7), # primiera camada - aplicar transformação de linear. n características e 7 neurônios          
            nn.ReLU(),  # ativar saída - transformar entre 0 e x            
            nn.Linear(7, 5), # segunda camada - aplicar transformação de linear. 5 neurônios
            nn.ReLU(), # ativar saída - transformar entre 0 e x
            nn.Linear(5, 1),  # terceira camada - aplicar transformação de linear. 2 neurônios (2 categorias)
            nn.Sigmoid() # ativação da rede - Sigmoid - saída entre 0 e 1
        )

    def forward(self, x):
        y = self.mlp_stack(x)
        return y
    

class CustomDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
def _get_dataset(data_dir):
    file = os.listdir(data_dir)[0] 
    df = pd.read_csv(f'{data_dir}/{file}')
    
    X = df[['CLIENTE_NOVO',
           'CLIENTE_INVESTIDOR', 'EMPRESTIMO_CDC', 'EMPRESTIMO_PESSOAL', 'SEXO_M',
           'IDADE_NORM', 'QTD_DIVIDAS_NORM']]

    y = df['ALVO']
    
    return np.array(X), np.array(y)
    
    
def _get_data_loader(batch_size, X, y):
    logger.info("Carregando dados de treinamento")    

    dataset = CustomDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    
    contador=Counter(y)
    pesos_classes=1./np.array([contador[0],contador[1]])
    pesos_amostra = np.array([pesos_classes[int(t)] for t in y])
    pesos_amostra=torch.from_numpy(pesos_amostra)
    amostra = WeightedRandomSampler(pesos_amostra, len(pesos_amostra))

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=amostra)

def train(args):    
    use_cuda = args.num_gpus > 0
    logger.debug("GPUs disponíveis - {}".format(args.num_gpus))
    device = torch.device("cuda" if use_cuda else "cpu")   

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    X, y = _get_dataset(args.data_dir_training)
    train_loader = _get_data_loader(args.batch_size, X, y)
    validation_loader = _get_data_loader(args.test_batch_size,X, y)

    logger.debug("Processar {}/{} ({:.0f}%) dados de treinamento".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processar {}/{} ({:.0f}%) dados de testes".format(
        len(validation_loader.sampler), len(validation_loader.dataset),
        100. * len(validation_loader.sampler) / len(validation_loader.dataset)
    ))

    model = MLP(n_attrs=len(X[0])).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    custo_fn = nn.BCELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = custo_fn(output, target.unsqueeze(1))
            loss.backward()            
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info('Época de treinamento: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        test(model, validation_loader, device)
    save_model(model, args.model_dir)


def test(model, validation_loader, device):
    model.eval()
    validation_loss = 0
    correct = 0
    custo_fn = nn.BCELoss()
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += custo_fn(output, target.unsqueeze(1)).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)
    logger.info('Validação: Custo médio: {:.4f}, Acurácia: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Salvando modelo.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--data-dir-training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--data-dir-validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())