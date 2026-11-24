import os
import socket
import gc
import uuid
import shutil
import argparse
import wandb
import torch
import torch.nn as nn
import optuna
import pandas as pd
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from huggingface_hub import try_to_load_from_cache

torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

SEED = 42
torch.manual_seed(SEED)

def resolve_to_local_path(model_name):
    '''Resolves a HF repo name to its local snapshot directory, if cached.'''
    config_path = try_to_load_from_cache(model_name, "config.json")
    if config_path is None or isinstance(config_path, str) and not os.path.exists(config_path):
        return model_name
    return os.path.dirname(config_path)


class Gemma4TextDataCollator(DataCollatorForLanguageModeling):
    '''
        Data collator that adds zeros as token_type_ids and mm_token_type_ids to
        Gemma model inputs to mark text-only inputs.
    '''
    def __call__(self, features, return_tensors=None):
        
        clean_features = []
        for feature in features:
            clean_feature = {k: v for k, v in feature.items() if not isinstance(v, str)}
            clean_features.append(clean_feature)
            
        batch = super().__call__(clean_features, return_tensors)
        
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        if "mm_token_type_ids" not in batch:
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
            
        return batch


def get_text_column_exceptions(examples, input_column_name = "Unimorph_features_lemma", 
                               output_column_name = "Original_tokens", inflections_column_name="Inflected_forms"):
    '''
       Accepts a Dataset and adds a text column to it where each item consists of a list of compound-lemma pairs 
       followed by an equal sign followed by the expected output tokens.
    ''' 
    inputs = examples[input_column_name]
    original = examples[output_column_name]
    outputs = examples[inflections_column_name]
    texts = []
    for inp, orig, out in zip(inputs, original, outputs):
        this_output = []
        orig_words = orig.split()
        out_words = out.split()
        for orig_w, out_w in zip(orig_words, out_words):
            if not "|" in out_w:
                this_output.append(out_w)
            elif orig_w in out_w.split("|"):
                this_output.append(orig_w)
            else:
                this_output.append(out_w.split("|")[0])
                
        texts.append(f"{inp} = {" ".join(this_output)}")
    return {"text": texts}

def get_text_column(examples, input_column_name = "Unimorph_features_lemma", output_column_name = "Original_tokens"):
    '''Accepts a Dataset and adds a text column to it where each item consists of a list of compound-lemma pairs 
       followed by an equal sign followed by the original tokens.
    '''
    inputs = examples[input_column_name]
    outputs = examples[output_column_name]
    texts = []
    for inp, out in zip(inputs, outputs):
        texts.append(f"{inp} = {out}")
    return {"text": texts}

def get_train_val_datasets_with_text_column(train_data_path, val_data_path_list, exceptions_split = False):
    '''Accepts the path to the training data (an Excel-file) and a list of paths to validation/test 
       datasets (also Excel-files) and returns Datasets that can be fed to SFTTrainer.
    '''
    df_train = pd.read_excel(train_data_path)
    df_val_list = [pd.read_excel(val_data_path) for val_data_path in val_data_path_list]

    dataset_train = Dataset.from_pandas(df_train)
    dataset_val_list = [Dataset.from_pandas(df_val) for df_val in df_val_list]

    if exceptions_split:
        dataset_train = dataset_train.map(get_text_column_exceptions, batched=True)
        dataset_val_list = [ds.map(get_text_column_exceptions, batched=True) for ds in dataset_val_list]
    else:
        dataset_train = dataset_train.map(get_text_column, batched=True)
        dataset_val_list = [ds.map(get_text_column, batched=True) for ds in dataset_val_list]
                
    return dataset_train, dataset_val_list


def train_model(model_id, ds_train, ds_val, output_dir_adapters, max_len=2048, device_batch_size=8, gradient_accumulation_steps=4,
         lr = 2e-4, weight_decay=0.01, logging_steps = 20, num_train_epochs = 10, save_strategy="epoch", lora_r = 16, lora_alpha = 32,
         lora_dropout = 0.05, lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
    
    '''
        Main function for training the models using SFTTrainer in combination with LoRaConfig. 
        Accepts a HuggingFace model ID, a train and a validation dataset as well as an output directory where 
        the LoRA checkpoints are saved.
    '''
    
    model_path = resolve_to_local_path(model_id) #args.model_id
    print(f"using model path: {model_path}", flush=True)
    
    #Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if 'gemma' in model_id:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    elif model_id in ["Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3.5-35B-A3B-Base"]:
        config = AutoConfig.from_pretrained(model_path)
        #if model_id == "Qwen/Qwen3-30B-A3B-Base":
        #    config.output_router_logits = True
        #else:
        #    config.text_config.output_router_logits = True
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    
    model.enable_input_require_grads()

    if model_id == "google/gemma-4-26B-A4B":
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
                target_parameters=["experts.gate_up_proj", "experts.down_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
                )
    elif model_id == "google/gemma-4-31B":
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
                )
    elif model_id == "Qwen/Qwen3-30B-A3B-Base":
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #"mlp.gate"
                target_parameters=["experts.gate_up_proj", "experts.down_proj"],
                bias="none",
                task_type="CAUSAL_LM"
            )
    elif model_id == "Qwen/Qwen3.5-35B-A3B-Base":
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #"shared_expert_gate"
                target_parameters=["experts.gate_up_proj", "experts.down_proj"], #"gate.weight"
                bias="none",
                task_type="CAUSAL_LM"
            )
    else:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
                )
    


    config = SFTConfig(
    output_dir=output_dir_adapters,
    max_length=max_len,
    packing=False,
    dataset_text_field="text",
    per_device_train_batch_size=device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=lr,
    weight_decay=weight_decay,
    logging_steps=logging_steps,
    eval_strategy = save_strategy,
    eval_steps = logging_steps if save_strategy == "steps" else None,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    save_strategy=save_strategy,
    save_only_model=True,
    save_steps = logging_steps if save_strategy == "steps" else None,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="_".join(output_dir_adapters.split("/")[-2:]),
    remove_unused_columns = False if 'gemma' in model_id else True
    )

    if 'gemma' in model_id:
            custom_collator = Gemma4TextDataCollator(tokenizer=tokenizer, mlm=False)

            trainer = SFTTrainer(
                model=model,      
                train_dataset=ds_train,   
                eval_dataset=ds_val,
                peft_config=peft_config,
                processing_class=tokenizer,
                data_collator=custom_collator,
                args=config   
                )
    else:

            trainer = SFTTrainer(
                model=model,      
                train_dataset=ds_train,   
                eval_dataset=ds_val,
                peft_config=peft_config,
                processing_class=tokenizer,
                args=config   
                )

    trainer.train()

    wandb.finish()
    del tokenizer
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


#Example: accelerate launch --mixed_precision bf16 main.py --model_id "Qwen/Qwen3-4B-Base" --train_data_path "data/data_splits/random_splits/original_features/train/Estonian.xlsx" 
# --val_data_path_list "data/data_splits/random_splits/original_features/test1/Estonian.xlsx" "data/data_splits/random_splits/original_features/test2/Estonian.xlsx" --output_dir "./trained_adapters"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tuning script using trl and accelerate")
    
    
    parser.add_argument("--model_id", type=str, required=True, help="Model to use")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--val_data_path_list", nargs="+", required=True, help="List of paths to validation datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Save path for the checkpoints")
    parser.add_argument("--max_len", type=int, default=2048, help="Max_input_length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging steps")
    parser.add_argument("--train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Which saving strategy to use")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials for hyperparameter search")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--target_modules", nargs="+", type=str, default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
                        help="List of model layers to apply LoRA to")
    parser.add_argument("--exceptions_split", action="store_true")
    parser.add_argument("--substitutivity_split", action="store_true")

    args = parser.parse_args()

    accelerator = Accelerator()

    assert not args.exceptions_split or not args.substitutivity_split #Can't be both at the same time

    if args.substitutivity_split:
        #Make sure that there are an equal number of test sets containing and not containing the string "syn"
        assert len([p for p in args.val_data_path_list if "syn" in p]) == len([p for p in args.val_data_path_list if "syn" not in p])

    ds_train, ds_val_list = get_train_val_datasets_with_text_column(args.train_data_path, args.val_data_path_list, exceptions_split = args.exceptions_split)

    if args.substitutivity_split:
        ds_val_list_not_syn = []
        ds_val_list_syn = []
        ds_val_data_path_list_not_syn = []
        ds_val_data_path_list_syn = []
        for i, val_path in enumerate(args.val_data_path_list):
            if "syn" not in val_path:
                ds_val_list_not_syn.append(ds_val_list[i])
                ds_val_data_path_list_not_syn.append(val_path)
            else:
                ds_val_list_syn.append(ds_val_list[i])
                ds_val_data_path_list_syn.append(val_path)
        ds_val_list = [{"not_syn" : item1, "syn" : item2} for item1, item2 in zip(ds_val_list_not_syn, ds_val_list_syn)]
        ds_val_path_list = [[item1, item2] for item1, item2 in zip(ds_val_data_path_list_not_syn, ds_val_data_path_list_syn)]
    
    #local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    #is_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"
    #device_map = None if is_fsdp else {"": local_rank}

    for i, ds_val in enumerate(ds_val_list):
        if not args.substitutivity_split:
            print(f"Train data path: {args.train_data_path}")
            print(f"Validation data path: {args.val_data_path_list[i]}")
            run_name = args.model_id.split("/")[-1] + "_" + "_".join(args.train_data_path.split("/")[-4:-1]) + "_" + "_".join(args.val_data_path_list[i].split("/")[-4:-1])
            out_path = os.path.join(args.output_dir, run_name)
        else:
            print(f"Train data path: {args.train_data_path}")
            print(f"Validation data paths: {ds_val_path_list[i]}")
            run_name = args.model_id.split("/")[-1] + "_" + "_".join(args.train_data_path.split("/")[-4:-1]) + "_" + "_".join(ds_val_path_list[i][0].split("/")[-4:-1])
            out_path = os.path.join(args.output_dir, run_name)
        
        train_model(args.model_id, 
                    ds_train, 
                    ds_val, 
                    out_path,
                    max_len=args.max_len, 
                    device_batch_size=args.batch_size, 
                    gradient_accumulation_steps=args.grad_accum,
                    lr = args.lr,
                    weight_decay = args.weight_decay,
                    logging_steps = args.logging_steps, 
                    num_train_epochs = args.train_epochs, 
                    save_strategy=args.save_strategy, 
                    lora_r = args.lora_r, 
                    lora_alpha = args.lora_alpha,
                    lora_dropout = args.lora_dropout, 
                   )
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"Final model saved to {args.output_dir}")

        accelerator.end_training()