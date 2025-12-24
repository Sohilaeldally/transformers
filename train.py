import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset

# hugging face tokenizer and dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import warnings
from pathlib import Path

from model import build_transformer
from config import get_config,get_wieghts_file_path,latest_weights_file_path
from dataset import BilingualDataset, causal_mask

def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx=tokenizer_tgt.token_to_id('[SOS]')
    eos_idx=tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it gor every step
    encoder_output=model.encode(source,source_mask)
    # Initialize decoder input with the sos token
    decoder_input= torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask= causal_mask(decoder_input.size(1).type_as(source_mask).to(device))

        # calc output
        out = model.decode(encoder_output,source_mask,decoder_input,decoder_mask)

        # get next token
        prob = model.project(out[:,-1])
        _,next_word=torch.max(prob,dim=1)
        
        decoder_input= torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_step,num_examples=2):
    model.eval()
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)

            #check that the bach size is 1
            assert encoder_input.size(0)==1 , "batch size msut be for validation"

            model_out= greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device) 
             
            source_text=batch['src_text'][0]
            target_text=batch['tgt_text'][0]

            model_out_text=tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            print_msg('-'*console_width)
            print_msg(f"{'SOURCE: ':>12}{source_text}")
            print_msg(f"{'TARGET: ':>12}{target_text}")
            print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

            
            if count == num_examples:
                print_msg('-'*console_width)
                break
       
        ''' # Evaluate the character error rate
            # Compute the char error rate 
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            wandb.log({'validation/cer': cer, 'global_step': global_step})

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            wandb.log({'validation/wer': wer, 'global_step': global_step})

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            wandb.log({'validation/BLEU': bleu, 'global_step': global_step})'''


def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path= Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=['[UNK]','[PAD]','[SOS]','[EOS]'],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang=lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

    
def get_ds(config):
   # It only has the train split, so we divide it
   ds_raw = load_dataset(config['datasource'],f"{config['lang_src']}-{config['lang_tgt']}",split='train')
   
   # Build Tokenizers
   tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
   tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])
                                   
    # Keep 90% for training, 10% for validation
   train_ds_size=int(0.9 * len(ds_raw))
   val_ds_size=len(ds_raw) - train_ds_size
   train_ds_raw ,val_ds_raw= random_split(ds_raw,[train_ds_size,val_ds_size])
   
   train_ds=BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
   val_ds=BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
   
   max_len_src=0
   max_len_tgt=0

   for item in ds_raw:
       src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
       tgt_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
       max_len_src=max(max_len_src,len(src_ids))
       max_len_tgt=max(max_len_src,len(tgt_ids))

   print(f'Max length of source sentence: {max_len_src}')
   print(f'Max length of target sentence: {max_len_tgt}')
   
   train_loader=DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True,)
   val_loader=DataLoader(val_ds,batch_size=1)

   return train_loader,val_loader,tokenizer_src,tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_ds(config):
    # Choose device
   device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print("Using device: ", device)
   if device.type == "cuda":
    print("Device name: ",torch.cuda.get_device_name(0))
    print("Device memory: ",torch.cuda.get_device_properties(0).total_memory/1024**3,"GB")
   else:
    print("Training on CPU")             # If using GPU, print its info

   # Make sure the weights folder exists
   Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True,exist_ok=True)

   train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
   model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size())

   # Tensorboard
   writer = SummaryWriter(config['experiment_name'])

   optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],eps=10e-9)

   initial_epoch=0
   global_step=0

   preload=config['preload']
   model_filename= latest_weights_file_path(config) if preload =='latest' else get_wieghts_file_path(config,preload) if preload else None
   if model_filename:
        print(f'preloading model {model_filename}')
        state= torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']
   else:
        print('No model to preload,starting from scratch')

   loss_fn= nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'),label_smoothing=.1).to(device)

   for epoch in range(initial_epoch,config['num_epochs']):
      torch.cuda.empty_cache()
      model.train()
      batch_iterator = tqdm(train_loader,desc=f"processing Epoch {epoch:02d}")
      for batch in batch_iterator:   
        encoder_input=batch['encoder_input'].to(device) #(B,seq_len)
        decoder_input=batch['decoder_input'].to(device) #(B,seq_len)
        encoder_mask=batch['encoder_mask'].to(device) #(B,1,1,seq_len)
        decoder_mask=batch['decoder_mask'].to(device) #(B,1,seq_len,seq_len)

        encoder_output= model.encode(encoder_input,encoder_mask) # (B,seq_len,d_model)
        decoder_output= model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) # (b,seq_len,d_model)
        proj_output=model.project(decoder_output) # (B,seq_len,vocab_size)

        label=batch['label'].to(device) # (B,seq_len)
        
        loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
        batch_iterator.set_postfix({"loss":f"{loss.item():6.3f}"})        

        # Log the loss
        writer.add_scalar('train_loss',loss.item(),global_step)
        writer.flush()
        optimizer.zero_grad(set_to_none=None)

        # Backpropagate the loss
        loss.backward()
        # Update the weights
        optimizer.step()

        global_step +=1
    
    # Run validation at the end of every epoch
   run_validation(model,val_loader,tokenizer_src,tokenizer_tgt,config['seq_len'],device,lambda msg: batch_iterator.write(msg),global_step)
   
    # Save the model ath the end of every epoch
   model_filename = get_wieghts_file_path(config,f"{epoch:02d}")
   torch.save({
       'epoch':epoch,
       'model_state_dict':model.state_dict(),
       'optimizer_state_dict':optimizer.state_dict(),
       'global_step':global_step
   },model_filename)
   
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config= get_config()
    train_ds(config)