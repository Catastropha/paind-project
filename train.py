import train_utils
import torch
from torch import nn
from torch import optim


def main():
    # get command line arguments
    in_arg = train_utils.get_input_args()
    
    # load training data
    print('Loading training data...')
    train_dataset, trainloader = train_utils.data_loader(in_arg.data_dir)
    print('Training data loaded\n')
    
    # load pretrained model
    print('Loading pretrained model...')
    model = train_utils.load_pretrained_model(in_arg.arch)
    print('Pretrained model loaded\n')
    
    # create classifier
    print('Creating classifier...')
    model.classifier = train_utils.create_classifier(model, in_arg.hidden_units)
    print('Classifier created\n')
    
    # Start training
    print('Training...')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    device = 'cuda' if torch.cuda.is_available() and in_arg.gpu is True else 'cpu'
    model.to(device)
    
    model.train()

    print_every = 200
    steps = 0
    for e in range(in_arg.epochs):
        running_loss = 0
        total = 0
        correct = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if steps % print_every == 0 or steps % len(trainloader) == 0:
                print("Epoch: {}/{}... ".format(e+1, in_arg.epochs),
                      "Loss: {:.4f} ".format(running_loss/print_every),
                      "Accuracy: %d %%" % (100 * correct / total),
                )

                running_loss = 0
                total = 0
                correct = 0
                
    # END training
    model.to('cpu')
    print('Training finished\n')
    
    print('Saving model...')
    checkpoint = train_utils.save_the_checkpoint(model, train_dataset, in_arg.arch, in_arg.hidden_units, in_arg.save_dir)
    print()
    print('Model saved. Checkpoint at: ' + checkpoint)
    
    


if __name__ == "__main__":
    main()
