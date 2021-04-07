import logging
import time
import copy

import torch
from torch import nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train(model, dataloaders, optimizer, num_epochs, weighted_loss, device):
    model = model.to(device)

    start_time = time.time()

    val_acc_history = []
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if weighted_loss:
        weight = torch.tensor(dataloaders['training'].dataset.class_weights)
        weight = weight.to(device)
    else:
        weight = None
    loss_fct = nn.CrossEntropyLoss(weight=weight)

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            confusion_matrix = torch.zeros(len(dataloaders[phase].dataset.genre_to_idx),
                                           len(dataloaders[phase].dataset.genre_to_idx))

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = loss_fct(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                for l, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[l.long(), p.long()] += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logger.info('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            for genre_name, genre_idx in dataloaders[phase].dataset.genre_to_idx.items():
                correct = confusion_matrix[genre_idx, genre_idx].long()
                total = confusion_matrix[genre_idx, :].sum().long()
                genre_acc = correct / total
                logger.info(f'    {genre_name:<20}: {correct:<4} / {total:<4} = {genre_acc:.4f} acc')

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

            if phase == 'validation':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best validation acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_weights)

    return model, val_acc_history
