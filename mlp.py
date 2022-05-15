import argparse
import json
import logging
import os
import sys
import torch
import time
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Baseado em https://github.com/aws/amazon-sagemaker-examples
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # Sequencia de tarefas da rede
        self.mlp_stack = nn.Sequential(            
            nn.Linear(7, 7), # primiera camada - aplicar transformação de linear. n características e 7 neurônios          
            nn.ReLU(),  # ativar saída - transformar entre 0 e x            
            nn.Linear(7, 5), # segunda camada - aplicar transformação de linear. 5 neurônios
            nn.ReLU(), # ativar saída - transformar entre 0 e x
            nn.Linear(5, 1),  # terceira camada - aplicar transformação de linear. 2 neurônios (2 categorias)
            nn.Sigmoid() # ativação da rede - Sigmoid - saída entre 0 e 1
        )

    def forward(self, x):
        y = self.mlp_stack(x)
        return y
    
# criar dataset personalizado
class CustomDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
# retornar dataset
def _get_dataset(data_dir):
    file = os.listdir(data_dir)[0] 
    df = pd.read_csv(f'{data_dir}/{file}')
    
    X = df[['CLIENTE_NOVO',
           'CLIENTE_INVESTIDOR', 'EMPRESTIMO_CDC', 'EMPRESTIMO_PESSOAL', 'SEXO_M',
           'IDADE_NORM', 'QTD_DIVIDAS_NORM']]

    y = df['ALVO']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=88)
    
    return np.array(X_train),  np.array(X_val), np.array(y_train), np.array(y_val)
    

# criar dataloader
def _get_data_loader(batch_size, X, y, is_distributed, **kwargs):
    logger.info("Carregando dados")    

    dataset = CustomDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **kwargs
    )

# treinamento
def train(args):    
    
    # processamento distribuído
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Treinamento Distribuído - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("No de GPUs disponíveis - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")    
        
    if is_distributed:
        # Iniciar ambiente distribuído.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info('Ambiente distribuído iniciado: \'{}\' em {} nós. '.format(
            args.backend, dist.get_world_size()) + 'Rank {}. No de gpus: {}'.format(
            dist.get_rank(), args.num_gpus))    
    
    X_train, X_val, y_train, y_val = _get_dataset(args.data_dir_training)
    train_loader = _get_data_loader(args.batch_size, X_train, y_train, is_distributed, **kwargs)
    validation_loader = _get_data_loader(1,X_val, y_val, False, **kwargs)

    logger.debug("Processar {}/{} ({:.0f}%) dados de treinamento".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processar {}/{} ({:.0f}%) dados de validacao".format(
        len(validation_loader.sampler), len(validation_loader.dataset),
        100. * len(validation_loader.sampler) / len(validation_loader.dataset)
    ))

    model = MLP().to(device)
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    custo_fn = nn.BCELoss()   
    
    valores_custo = []
    valores_acuracia = []

    for epoch in range(1, args.epochs + 1):
        custo_epoca = 0
        acuracia_epoca = 0
        logger.info(f"\n-------------------------------\nÉpoca {epoch}")
        model.train()
        for batch, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred = model(X_batch)
            
            custo = custo_fn(y_pred, y_batch.unsqueeze(1))
            
            optimizer.zero_grad()
            custo.backward()            
            optimizer.step()           
            
            custo_epoca +=  custo.item()
            pred_class = torch.round(y_pred)
            
            acuracia_epoca += accuracy_score(y_batch.cpu(), pred_class.detach().cpu())
            
            #if batch_idx % args.log_interval == 0:
            #    logger.info('Época de treinamento: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            #        epoch, batch_idx * len(data), len(train_loader.sampler),
            #        100. * batch_idx / len(train_loader), loss.item()))
        test(model, validation_loader, device)
        
        valores_custo.append(custo_epoca/len(train_loader))
        valores_acuracia.append(acuracia_epoca/len(train_loader))
        logger.info(f"custo: {valores_custo[-1]} | acurácia: {valores_acuracia[-1]}")
        
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
            validation_loss += custo_fn(output, target.unsqueeze(1)).item() 
            pred_class = torch.round(output)            
            correct += accuracy_score(target.cpu(), pred_class.detach().cpu())

    validation_loss /= len(validation_loader.dataset)
    logger.info('Validação: Custo médio: {:.4f}, Acurácia: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))


# carregar modelo
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(MLP())
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

# salvar modelo
def save_model(model, model_dir):
    logger.info("Salvando modelo.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    start_time = time.time()
    logger.info("###############################################################")        
    logger.info(f"TREINAMENTO INICIADO")
    logger.info("###############################################################")    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    #parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                    help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    #parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                    help='random seed (default: 1)')
    #parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                    help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default='gloo',
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--data-dir-training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    #parser.add_argument("--data-dir-validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
    
    logger.info("###############################################################")
    logger.info(f"TREINAMENTO FINALIZADO")
    logger.info(f"--- Tempo: {time.time() - start_time} segundos ---")
    logger.info("###############################################################")