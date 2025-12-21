from pathlib import Path

def get_config():
    return{
        'batch' : 8,
        'epochs' : 20,
        'lr' : 10**-4,
        'seq_len' :350,
        'datasource' : 'opus_books',
        'lang_src' : 'en',
        'lang_tgt' : 'es',
        'd_model' : 512,
        'model_folder' : 'weights',
        'model_basename' : 'tmodel',
        'experiment_name' : 'runs/tmodel',
        'preload' : 'latest',
        'tokenizer_file' : 'tokenizer_{0}.json' 

    }

def get_wieghts_file_path(config,epoch:str):
   model_folder = f"{config['datasource']}_{config['model_folder']}"
   model_filename = f"{config['model_basename']}{epoch}.py"
   return str(Path('.')/model_folder/model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files=list(Path(model_folder).glob(model_filename))
    if len(weights_files)==0:
        return None
    weights_files.sort()
    return str(weights_files[-1])



