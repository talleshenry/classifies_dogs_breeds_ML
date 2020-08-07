from datetime import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET # for parsing XML
import matplotlib.pyplot as plt # to show images
import os
import glob

ARQUIVO_REDE = 'racas_cachorros.pth'
NOMES_LABELS = ['beagle', 'golden_retriever', 'pug',
 'rottweiler', 'siberian_husky']

data_path = 'images_croped'

transformacoes = transforms.Compose([
  transforms.Resize([100, 100]),
  transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(
  root=data_path,
  transform=transformacoes
)

train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=1,
  num_workers=8,
  shuffle=True
)

class RacasCachorrosModel(nn.Module):
  def __init__(self):
    super(RacasCachorrosModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, 5) # canais, qtd filtros, kernel
    self.conv2 = nn.Conv2d(8, 16, 5)
    self.linear1 = nn.Linear(7056, 120)
    self.linear2 = nn.Linear(120, 84)
    self.linear3 = nn.Linear(84, 5)
    self.pool = nn.MaxPool2d(3, stride=2)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 7056)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x

rede = RacasCachorrosModel()

def total_certo(labels, saida):
  total = 0
  for i, val in enumerate(saida):
    val = val.tolist()
    max_idx = val.index(max(val))
    if labels[i] == max_idx:
      total += 1
  return total

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rede.parameters(), lr=0.001)

def treina(epochs = 100):
  print('Iniciando o treinamento...')
  for epoch in range(epochs):
    erro_total = 0
    acertou = 0
    total = 0
    for batch, (entrada, label) in enumerate(train_loader):
      optimizer.zero_grad()
      saida = rede(entrada)
      acertou += total_certo(label, saida)
      total += len(label)
      erro = criterion(saida, label)
      erro.backward()
      optimizer.step()
      erro_total += erro.item()
    acuracia =  100.0 * acertou / total
    print('Epoch: {} - Erro: {:.4f} - Acuracia: {:.2f}%'.
        format(epoch + 1, erro_total, acuracia))
    if acuracia == 100 and erro_total < 0.1:
      break

def load_image(path):
  image = Image.open(path)
  image = transformacoes(image).float()
  image = Variable(image, requires_grad=True)
  image = image.unsqueeze(0)
  return image

def carrega_rede():
  rede.load_state_dict(torch.load(ARQUIVO_REDE))
  rede.eval()

def testa_imagem(path):
  imagem = load_image(path)
  saida = rede(imagem)
  val = saida.squeeze().tolist()  
  max_idx = val.index(max(val))
  return NOMES_LABELS[max_idx]

def bounding_box(image):
    tree = ET.parse(image)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox') # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
    return (xmin,ymin,xmax,ymax)

#Cada imagem possui um arquivo annotation contendo a posição em que o animal está na image.
#Inseri esse corte para melhorar a precisão
def crop_images():
  print('cortando as imagens de acordo com a posição definida no annotation...')
  croped_images_directory='images_croped/'
  try:
    os.rmdir(croped_images_directory)
  except OSError as e:
    pass

  try:
    os.mkdir(croped_images_directory)
  except OSError as e:
    pass

  for raca in NOMES_LABELS:
    images_path='treinamento/'+raca
    annotations_path='annotations/'+raca

    all_images=os.listdir(images_path)
    try:
      os.mkdir(croped_images_directory+raca+'/')
    except OSError as e:
      pass

    for i,image in enumerate(all_images):
      bbox=bounding_box(annotations_path+'/'+image[:-4])
      im=Image.open(os.path.join(images_path,image))
      im=im.crop(bbox)
      im=im.save(croped_images_directory+raca+'/'+image)

def exibe_menu():
  while True:
    print('1. Treinar')
    print('2. Testar a rede')
    print('3. Sair')
    opcao = input('Digite sua opcao: ')
    if opcao == '1':
      crop_images()
      treina(100)
      print('Salvando a rede ...')
      torch.save(rede.state_dict(), ARQUIVO_REDE)
      print('Rede salva com sucesso')
    elif opcao == '2':
      print('Carregando a rede ...')
      carrega_rede()
      print(testa_imagem(input('Digite o caminho da imagem: ')))
    elif opcao == '3':            
      break
    else:
        print('Digite uma opcao valida')

if __name__ == '__main__':
  exibe_menu()