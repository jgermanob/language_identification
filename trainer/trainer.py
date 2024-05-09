import torch
import numpy as np
import copy
import time
import logging

logging.basicConfig(level=logging.DEBUG)

class LID_LSTM_Trainer():
    def __init__(self, args, optimizer, loss_function):
        self.args = args
        self.optimizer = optimizer
        self.loss_function = loss_function

    def _set_history(self):
        epochs = self.args.epochs
        history = {
            'train':{
                'loss': np.zeros(epochs)
                },
            'val':{
                'loss': np.zeros(epochs)
                },
            'test':{
                'loss': np.zeros(epochs)
                }
            }
        return history
    
    def _get_batch_data(self, batch, device):
        batch_x_char = batch['x_chars'].to(device)
        batch_x_char_len = batch['x_chars_len'].to(device)
        batch_labels = batch['labels'].to(device)

        return batch_x_char, batch_x_char_len, batch_labels
        
    def train_step(self, model, batch, device):
        self.optimizer.zero_grad()
        batch_x_char, batch_x_char_len, batch_labels = self._get_batch_data(batch, device)
        batch_logits = model(batch_x_char, batch_x_char_len)
        batch_loss = self.loss_function(batch_logits, batch_labels)

        batch_loss.backward()
        self.optimizer.step()

        return batch_loss.item()
    
    def validation_step(self, model, batch, device):
        batch_x_char, batch_x_char_len, batch_labels = self._get_batch_data(batch, device)
        batch_logits = model(batch_x_char, batch_x_char_len)
        batch_loss = self.loss_function(batch_logits, batch_labels)

        return batch_loss.item()
    
    def train(self, model, train_dataloader, val_dataloader, test_dataloader, device):
        epochs = self.args.epochs
        history = self._set_history()
        train_batch_number = len(train_dataloader)
        val_batch_number = len(val_dataloader)
        test_batch_number = len(test_dataloader)
        min_loss  = torch.inf
        best_model = copy.deepcopy(model)
        logging.debug('Beginning training\n')
        patience = 0
        for epoch in range(epochs):
            start = time.time()
            model.train()
            # Training loop
            for batch in train_dataloader:
                batch_loss = self.train_step(model, batch, device)
                history['train']['loss'][epoch] += batch_loss
            
            # Validation loop
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    batch_loss = self.validation_step(model, batch, device)
                    history['val']['loss'][epoch] += batch_loss

                for batch in test_dataloader:
                    batch_loss = self.validation_step(model, batch, device)
                    history['test']['loss'][epoch] += batch_loss
            
            #Update epoch history
            history['train']['loss'][epoch] /= train_batch_number
            history['val']['loss'][epoch] /= val_batch_number
            history['test']['loss'][epoch] /= test_batch_number

            #Saving checkpoint iff loss decrease
            if history['val']['loss'][epoch] < min_loss:
                min_loss = history['val']['loss'][epoch]
                best_model.load_state_dict(model.state_dict())
                patience = 0
            else:
                patience += 1
            
            #Printing epoch performance info
            logging.debug(f'Epoch {epoch+1}: '
                          f'train_loss = {history["train"]["loss"][epoch]:.4f}, '
                          f'val_loss = {history["val"]["loss"][epoch]:.4f}, '
                          f'test_loss = {history["test"]["loss"][epoch]:.4f}, '
                          f'patience = {patience}, '
                          f'time: {time.time() - start:.4f}[s]')
            
            if patience == self.args.patience:
                logging.debug(f'Training finished after {epoch+1} epochs.')
                return model, best_model, history
        
        return model, best_model, history
                
            






        


       