import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils.early_stopping import EarlyStopping

def train_student_model(teacher_model, student_model,
                weights_file, 
                train_loader,
                val_loader,
                epochs,
                lr,
                distil_weight=0.7,
                file_prefix= '',
                early_stop_patience = 10000):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')        
        student_model.to(device)
        teacher_model.to(device)
        
        teacher_model.eval()
        student_model.train()

        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=True, path=weights_file)
        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)

        num_train_samples = len(train_loader)
        print(f'Num Train Samples: {num_train_samples}')

        columns = ['train_ce_loss', 'train_kl_loss', 'train_total_loss',
                   'val_ce_loss', 'val_kl_loss', 'val_total_loss']

        train_results = pd.DataFrame(columns=columns)
        avg_losses = dict.fromkeys(columns)
        
        for epoch in range(epochs):

        ###################
        # train the model #
        ###################
    
            epoch_ce_loss = 0
            epoch_kl_loss = 0
            epoch_total_loss = 0

            ## For GPU 
            # epoch_ce_loss = torch.cuda.FloatTensor(1).fill_(0)
            # epoch_kl_loss = torch.cuda.FloatTensor(1).fill_(0)
            # epoch_total_loss = torch.cuda.FloatTensor(1).fill_(0)

            for i, data in enumerate(train_loader):
                input, _ = data
                input = input.to(device)
                out, mu, logvar = student_model(input)

                # if epoch == 75:
                #     for group in optimizer.param_groups:
                #         group['lr'] = 1e-6

                print(input.shape)

                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
                ce_loss = torch.nn.functional.binary_cross_entropy(
                    input=out,
                    target=input,
                    size_average=False)
                student_loss = ce_loss + torch.mul(kl_loss, student_model.beta)

                with torch.no_grad():
                    teacher_out, _, _ = teacher_model(input)

                print(out.shape)
                print(teacher_out.shape)

                distill_ce_loss = torch.nn.functional.binary_cross_entropy(
                    input=out,
                    target=teacher_out,
                    size_average=False)

                total_loss = (1-distil_weight) * student_loss + distil_weight * distill_ce_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_ce_loss += ce_loss.item()
                epoch_kl_loss += kl_loss.detach().numpy()
                epoch_total_loss += total_loss.detach().numpy()
                
                break
            
            avg_losses['train_ce_loss'] = epoch_ce_loss/num_train_samples
            avg_losses['train_kl_loss'] = epoch_kl_loss/num_train_samples
            avg_losses['train_total_loss'] = epoch_total_loss/num_train_samples

        ######################
        # validate the model #
        ######################

            num_val_samples = len(val_loader)

            with torch.no_grad():
                student_model.eval()  

                epoch_ce_loss = 0
                epoch_kl_loss = 0
                epoch_total_loss = 0

                ## For GPU
                # epoch_ce_loss = torch.cuda.FloatTensor(1).fill_(0)
                # epoch_kl_loss = torch.cuda.FloatTensor(1).fill_(0)
                # epoch_total_loss = torch.cuda.FloatTensor(1).fill_(0)

                for data  in val_loader:

                    input, _ = data
                    input = input.to(device)
                    out, mu, logvar = student_model(input)
                    
                    kl_loss = torch.mul(
                            input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                            other=0.5)
                    ce_loss = torch.nn.functional.binary_cross_entropy(
                            input=out,
                            target=input,
                            size_average=False)
                    loss = ce_loss + torch.mul(kl_loss, student_model.beta)

                    epoch_ce_loss += ce_loss.item()
                    epoch_kl_loss += kl_loss.detach().numpy()
                    epoch_total_loss += loss.detach().numpy()

                avg_losses['val_ce_loss'] = epoch_ce_loss/num_val_samples
                avg_losses['val_kl_loss'] = epoch_kl_loss/num_val_samples
                avg_losses['val_total_loss'] = epoch_total_loss/num_val_samples
                
                train_results = train_results.append(avg_losses, ignore_index=True)

                # If validation loss decreased, will create a checkpoint
                early_stopping(avg_losses['val_total_loss'], student_model)

                break
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
    
            print(f"Epoch: {epoch}; Train_loss: {avg_losses['train_total_loss']}; Val_loss: {avg_losses['val_total_loss']}")

        print('Training finished, saving results csv & saving weights...')

        train_results.to_csv('/Users/aditya/Desktop/Training_Results/out_files/'+file_prefix+'out.csv', index=False)
        torch.save(student_model.state_dict(), weights_file)