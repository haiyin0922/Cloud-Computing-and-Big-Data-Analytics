# Music Memorability Prediction

## Usage
```
python HW1_CNN.py test.csv
```
## Result
```
public score = 0.00686
（此結果為兩次0.008多submission之平均）
```

## Code structure

### Feature
```
先把音檔都轉成mel-frequency spectrogram後再開始操作
```

### Import the libraries
```
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, MSELoss
from torch.optim import Adam, SGD
```

### Set random seed
```
import random

my_seed = 3
random.seed(my_seed)
torch.manual_seed(my_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(my_seed)
```

### Loading dataset
```
train = pd.read_csv('./train.csv')
test = pd.read_csv('./' + input)

sample_submission = pd.read_csv('sample_submission.csv')

train.head()
```

### Loading training images
```
train_img = []
for img_name in tqdm(train['track']):
    # defining the image path
    image_path = './img/' + str(img_name) + '.jpg'
    # reading the image
    img = imread(image_path)
    img = resize(img, (28, 28))
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # normalizing the pixel values
    #img /= 255.0
    # appending the image into the list
    train_img.append(img)

# converting the list to numpy array
train_x = np.array(train_img)
# defining the target
train_y = train['score'].values

#print(train_x)
#print(train_y)

train_x.shape
```

### Create validation set
```
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, random_state=my_seed, test_size = 0.05)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
```

### Convert data into torch format
```
train_x = train_x.reshape(209, 3, 28, 28)
train_x  = torch.from_numpy(train_x)

train_y = train_y.astype(float);
train_y = torch.from_numpy(train_y)

train_x.shape, train_y.shape

val_x = val_x.reshape(11, 3, 28, 28)
val_x  = torch.from_numpy(val_x)

val_y = val_y.astype(float);
val_y = torch.from_numpy(val_y)

val_x.shape, val_y.shape
```

### Model architecture
```
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 1)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        #print(x)
        x = self.linear_layers(x)
        return x
```

### Model related
```
# defining the model
model = Net()

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.05)

# defining the loss function
criterion = MSELoss

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
print(model)
```

### Train model
```
def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train)
    output_val = model(x_val)

    # computing the training and validation loss
    output_train=output_train.squeeze(-1)
    output_val=output_val.squeeze(-1)
    loss_train = criterion(output_train, y_train.float())
    loss_val = criterion(output_val, y_val.float())
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    # printing the validation loss
    print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
```

### Training related
```
# defining the number of epochs
n_epochs = 300
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)
```

### Loading test image
```
test_img = []
for img_name in tqdm(test['track']):
    # defining the image path
    image_path = './img/' + str(img_name) + '.jpg'
    # reading the image
    img = imread(image_path)
    # normalizing the pixel values
    #img /= 255.0
    img = resize(img, (28, 28))
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    test_img.append(img)

# converting the list to numpy array
test_x = np.array(test_img)
test_x.shape
```

### Converting training images into torch format
```
test_x = test_x.reshape(15, 3, 28, 28)
test_x  = torch.from_numpy(test_x)
test_x.shape
```

### generating predictions for test set
```
with torch.no_grad():
    output = model(test_x.cuda())

print(output)

# replacing the label with prediction
sample_submission['score'] = output.cpu()
sample_submission.head()

# saving the file
sample_submission.to_csv('submission.csv', index=False)

torch.save(model, 'CNN.pth')
```