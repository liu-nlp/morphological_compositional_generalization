import os
import gc
import ast
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SEED = 42
torch.manual_seed(SEED)


def get_model_with_adapter(model_id, adapter_path):
    '''
        Accepts a HuggingFace model ID and the path to a LoRA checkpoint and returns a PeftModel.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #Get device map
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false") == "true"
    device_map = None if is_fsdp else {"": local_rank}

    if 'gemma' in model_id:
        base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2"
        )

    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer

def get_text_column_test_data(examples, input_column_name = "Unimorph_features_lemma"):
    '''
        Accepts a Dataset and adds a column text where each item consists of the input compound-lemma pairs 
        followed by an equal sign.
    '''
    inputs = examples[input_column_name]
    texts = []
    for inp in inputs:
        texts.append(f"{inp} =")
    return {"text": texts}

def get_test_dataset_with_text_column(test_data_path):
    '''
        Accepts the path to an Excel-file and returns a Dataset that can be fed to the PeftModel
    '''
    df_test = pd.read_excel(test_data_path)
    dataset_test = Dataset.from_pandas(df_test)
    dataset_test = dataset_test.map(get_text_column_test_data, batched=True)
    return dataset_test

def get_test_dataset_localism(localism_test_data_path, df_columns_list = ["Original_tokens", "Unimorph_features_lemma", "Inflected_forms"]):
    '''
        Accepts the path to an Excel-file containing the data for the localism split and turns it into the 
        correct format for evaluation by using pandas explode.
    '''
    df = pd.read_excel(localism_test_data_path)
    df["Row_ID"] = df.index

    for col in df_columns_list:
        df[col] = df[col].apply(ast.literal_eval)
    
    df_exploded = df.explode(df_columns_list).reset_index(drop=True)

    dataset_loc = Dataset.from_pandas(df_exploded)
    dataset_loc = dataset_loc.map(get_text_column_test_data, batched=True)
    return df_exploded, dataset_loc

def get_predictions(model, tokenizer, test_dataset, batch_size = 64, gemma_model=False):
    '''
        Accepts the model to be evaluated and its tokenizer as well as the Dataset to use for evaluation.
        Returns model predictions.
    '''
    prompts = test_dataset["text"]
    all_responses = []

    model.eval()

    for i in tqdm(range(0, len(prompts), batch_size)):
        current_batch = prompts[i : i + batch_size]
    
        inputs = tokenizer(
            current_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(model.device)

        if gemma_model:
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
            if "mm_token_type_ids" not in inputs:
                inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])


        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,     
                do_sample=False,        
                use_cache=True
            )
    
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
    
        decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_responses.extend(decoded_responses)

    return all_responses

def get_accuracy_default(model_predictions, test_dataset, column_name_inputs = "Unimorph_features_lemma",column_name_correct = "Inflected_forms"):
    '''
        Accepts a list of model_predictions and the evaluation Dataset and compares the word forms in the "Inflected_forms"
        column of the dataset to the model predictions to measure exact match. A prediction is considered an exact match if
        each word in the model prediction corresponds to a possible inflection given by the lemma and its UniMorph feature set.
        Returns the number of exact matches, the total number of samples, the exact match score as well as a list of mistakes
        made by the model.
    '''
    all_responses = [item.strip() for item in model_predictions]
    facit = test_dataset[column_name_correct]
    inputs = test_dataset[column_name_inputs]
    mistakes_list = []

    num_correct = 0
    for gen, correct, inp in zip(all_responses, facit, inputs):
        example_correct = True
        gen_words = gen.split()
        correct_words = correct.split()
        input_words = inp.split()
        if len(gen_words) != len(correct_words):
            example_correct = False
            mistakes_list.append({"Reason" : "Difference in the number of words", 
                                 "Number of words generated" : len(gen_words), 
                                 "Number of correct words" : len(correct_words),
                                 "Input" : inp,
                                 "Generation" : gen,
                                 "Correct" : correct})
        else:
            for gen_w, correct_w, input_w in zip(gen_words, correct_words, input_words):
                if "|" not in correct_w:
                    if gen_w != correct_w:
                        example_correct = False
                        mistakes_list.append({"Reason" : "Wrong word generated",
                                              "Input word" :  input_w,
                                              "Generated word" : gen_w, 
                                              "Correct word" : correct_w,
                                              "Input" : inp,
                                              "Generation" : gen,
                                              "Correct" : correct})
                elif gen_w not in correct_w.split("|"):
                    example_correct = False
                    mistakes_list.append({"Reason" : "Generated word not among alternatives",
                                          "Input word" : input_w, 
                                          "Generated word" : gen_w, 
                                          "Correct word" : correct_w,
                                          "Input" : inp,
                                          "Generation" : gen,
                                          "Correct" : correct})
        num_correct += example_correct

    return {'correct' : num_correct, 'total' : len(facit), 'accuracy' : num_correct/len(facit)}, mistakes_list


def get_accuracy_productivity(model_predictions, test_dataset, column_name_correct = "Inflected_forms", max_train_len=12):
    '''
        Accepts a list of model predictions and an evaluation Dataset and measures exact length by the number of
        compound-lemma pairs in the input string.
    '''
    all_responses = [item.strip() for item in model_predictions]
    facit = test_dataset[column_name_correct]

    results_detailed = dict()
    for gen, correct in zip(all_responses, facit):
        example_correct = True
        gen_words = gen.split()
        correct_words = correct.split()
        if len(correct_words) not in results_detailed:
            results_detailed[len(correct_words)] = {"correct" : 0, "total" : 0, "accuracy" : 0}
        if len(gen_words) != len(correct_words):
            example_correct = False
        else:
            for gen_w, correct_w in zip(gen_words, correct_words):
                if "|" not in correct_w:
                    if gen_w != correct_w:
                        example_correct = False
                elif gen_w not in correct_w.split("|"):
                    example_correct = False
        results_detailed[len(correct_words)]["correct"] += example_correct
        results_detailed[len(correct_words)]["total"] += 1
    results_detailed = {k: v for k, v in sorted(results_detailed.items(), key=lambda item: item[0])}

    for key, value in results_detailed.items():
        results_detailed[key]["accuracy"] = results_detailed[key]["correct"] / results_detailed[key]["total"] if results_detailed[key]["total"] > 0 else np.nan
    
    return results_detailed

def get_accuracy_overgeneralisation(exception_predictions, exceptions_test_dataset, column_name_correct = "Inflected_forms", 
                                    column_name_correct_original = "Inflected_forms_original", column_name_inputs="Unimorph_features_lemma",
                                    exceptions_dict = {"SG" : {"value" : "PL", "condition" : ["V", "IND", "PRS"]}, "FRML" : {"value" : "TRANS", "condition" : ["N", "SG"]}}):
    '''
        Accepts a list of model predictions and en evaluation Dataset and counts the number of words that match the
        original form or the exception or both for words that the exception applies to.
    '''
    
    model_predictions = [item.strip() for item in exception_predictions]
    facit_original = exceptions_test_dataset[column_name_correct_original]
    facit_exceptions = exceptions_test_dataset[column_name_correct]
    inputs = exceptions_test_dataset[column_name_inputs]
    matches_dict = {"matches_both" : 0, "matches_original" : 0, "matches_exception" : 0, "matches_neither" : 0, "different_word_count" : 0, "total_relevant" : 0}
    words_dict = {"matches_both" : [], "matches_original" : [], "matches_exception" : [], "matches_neither" : []}

    for pred, orig, new, inp in zip(model_predictions, facit_original, facit_exceptions, inputs):
        pred_words = pred.split()
        orig_words = orig.split()
        new_words = new.split()
        input_words = inp.split()
        if len(pred_words) == len(orig_words) == len(new_words) == len(input_words):
            for pred_w, orig_w, new_w, input_w in zip(pred_words, orig_words, new_words, input_words):
                feats = input_w.split("(")[0].split(";")
                for before, after in exceptions_dict.items():
                    subset = [before] + after["condition"]
                    if set(subset).issubset(set(feats)):
                        matches_dict["total_relevant"] += 1
                        if pred_w in orig_w.split("|") and pred_w in new_w.split("|"):
                            matches_dict["matches_both"] += 1
                            words_dict["matches_both"].append((pred_w, orig_w, new_w))
                        elif pred_w in orig_w.split("|"):
                            matches_dict["matches_original"] += 1
                            words_dict["matches_original"].append((pred_w, orig_w, new_w))
                        elif pred_w in new_w.split("|"):
                            matches_dict["matches_exception"] += 1
                            words_dict["matches_exception"].append((pred_w, orig_w, new_w))
                        else:
                            matches_dict["matches_neither"] += 1
                            words_dict["matches_both"].append((pred_w, orig_w, new_w))
        else:
            matches_dict["different_word_count"] += 1
    return matches_dict, words_dict
                
    
    

def get_consistency_score_localism(original_predictions, localism_predictions, test_dataset, column_name_correct = "Inflected_forms"):
    '''
        Accepts model predictions for the original evaluation Dataset as well as the ones predicted word-by-word for the
        same Dataset and counts the number of consistent and correct items, consistent and incorrect items as well as the
        number of inconsistent items.
    '''
    original_responses = [item.strip() for item in original_predictions]
    localism_responses = [item.strip() for item in localism_predictions]
    facit = test_dataset[column_name_correct]
    consistency_dict = {"consistent_correct" : 0, "consistent_incorrect" : 0, "inconsistent" : 0, "total" : 0}
    examples_dict = {"consistent_incorrect" : [], "inconsistent" : []}

    for orig, loc, correct in zip(original_responses, localism_responses, facit):
        orig_words = orig.split()
        loc_words = loc.split()
        correct_words = correct.split()
        if orig != loc:
            consistency_dict["inconsistent"] += 1
            examples_dict["inconsistent"].append((orig, loc, correct))
        elif len(orig_words) == len(correct_words):
            match = True
            for orig_w, correct_w in zip(orig_words, correct_words):
                if orig_w not in correct_w.split("|"):
                    match = False
            if match:
                consistency_dict["consistent_correct"] += 1
            else:
                consistency_dict["consistent_incorrect"] += 1
                examples_dict["consistent_incorrect"].append((orig, loc, correct))
        consistency_dict["total"] += 1
    return consistency_dict, examples_dict


def get_consistency_score_substitutivity(original_predictions, substitutivity_predictions, test_dataset, column_name_correct = "Inflected_forms",
                                         column_name_inputs = "Unimorph_features_lemma", synonyms_dict = {"GEN" : "ENAF", "CMPR" : "PAH", "PASS" : "EDLG", "SG" : "TEK"}):
    '''
        Accepts model predictions for the original evaluation Dataset as well as the copy where selected features
        have been replaced by a synonym and counts the number of consistent and correct cases, consistent and incorrect
        cases as well as the number of inconsistent cases.
    '''
    
    original_responses = [item.strip() for item in original_predictions]
    sub_responses = [item.strip() for item in substitutivity_predictions]
    facit = test_dataset[column_name_correct]
    inputs = test_dataset[column_name_inputs]
    consistency_dict = {"consistent_correct" : 0, "consistent_incorrect" : 0, "inconsistent" : 0, "total_relevant" : 0, "different_word_count" : 0}
    examples_dict = {"consistent_incorrect" : [], "inconsistent" : []}

    for orig, sub, correct, inp in zip(original_responses, sub_responses, facit, inputs):
        orig_words = orig.split()
        sub_words = sub.split()
        correct_words = correct.split()
        input_words = inp.split()
        if len(orig_words) == len(sub_words) == len(correct_words) == len(input_words):
            for orig_w, sub_w, correct_w, input_w in zip(orig_words, sub_words, correct_words, input_words):
                feats = input_w.split("(")[0].split(";")
                if len(set(feats).intersection(list(synonyms_dict.keys()))) > 0:
                    if orig_w != sub_w:
                        consistency_dict["inconsistent"] += 1
                        examples_dict["inconsistent"].append((orig_w, sub_w, correct_w))
                    elif orig_w == sub_w and orig_w in correct_w.split("|"):
                        consistency_dict["consistent_correct"] += 1
                    elif orig_w == sub_w and orig_w not in correct_w.split("|"):
                        consistency_dict["consistent_incorrect"] += 1
                        examples_dict["consistent_incorrect"].append((orig_w, sub_w, correct_w))
                    consistency_dict["total_relevant"] += 1
        else:
            consistency_dict["different_word_count"] += 1
    return consistency_dict, examples_dict
    


def main_evaluate(model_id, adapter_folder, test_data_path, save_folder_path, split="random", max_train_len=12, 
                 batch_size=64, batch_size_localism=128, other_test_data_path=None, column_name_correct = "Inflected_forms", 
                 column_name_correct_original = "Inflected_forms_original", column_name_inputs="Unimorph_features_lemma",
                 synonyms_dict = {"GEN" : "ENAF", "CMPR" : "PAH", "PASS" : "EDLG", "SG" : "TEK"},
                 exceptions_dict = {"SG" : {"value" : "PL", "condition" : ["V", "IND", "PRS"]}, "FRML" : {"value" : "TRANS", "condition" : ["N", "SG"]}},
                 gemma_model=False):
    '''
        Main evaluation function that accepts a HuggingFace model ID and the path to the LoRA checkpoint as well the path(s)
        to the evaluation data and applies the relevant evaluation functions depending on which data split is being evaluated.
        The evaluation results are saved to save_folder_path.
    '''
    accuracy_dict = dict()
    mistakes_dict = dict()
    if split == "productivity":
        accuracy_dict_per_length_detailed = dict()
    elif split in ["localism", "substitutivity"]:
        consistency_dicts = dict()
        examples_dicts = dict()
        if split == "substitutivity":
            accuracy_dict_sub = dict()
            mistakes_dict_sub = dict()
    elif split in ["overgeneralisation", "overgeneralization"]:
        exceptions_dicts = dict()
        exceptions_word_dicts = dict()
    
    #Get data
    test_dataset = get_test_dataset_with_text_column(test_data_path)
    if split == "localism" and other_test_data_path:
        df_exploded, test_dataset_localism = get_test_dataset_localism(other_test_data_path)
    elif split == "substitutivity" and other_test_data_path:
        test_dataset_sub = get_test_dataset_with_text_column(other_test_data_path)
    
    
    #Get checkpoint paths
    checkpoint_paths = dict()
    for item in os.listdir(adapter_folder):
        if "checkpoint" in item:
            checkpoint_paths[int(item.split("-")[-1])] = f"{adapter_folder}/{item}"
    if split in ["overgeneralisation", "overgeneralization"]:
        checkpoint_paths = {k: v for k, v in sorted(checkpoint_paths.items(), key=lambda item: item[0]) if k <= 500}
    else:
        checkpoint_paths = {k: v for k, v in sorted(checkpoint_paths.items(), key=lambda item: item[0])}

    for k, adapter_path in checkpoint_paths.items():
        model, tokenizer = get_model_with_adapter(model_id, adapter_path)
        all_predictions = get_predictions(model, tokenizer, test_dataset, batch_size=batch_size, gemma_model=gemma_model)
        accuracy, mistakes_list = get_accuracy_default(all_predictions, test_dataset)
        accuracy_dict[k] = accuracy
        mistakes_dict[k] = mistakes_list
        if split == "productivity":
            results_detailed = get_accuracy_productivity(all_predictions, test_dataset, max_train_len=max_train_len)
            accuracy_dict_per_length_detailed[k] = results_detailed
        elif split == "localism":
            localism_predictions = get_predictions(model, tokenizer, test_dataset_localism, batch_size = batch_size_localism, gemma_model=gemma_model)
            df_exploded["Prediction"] = localism_predictions
            df_grouped = df_exploded.groupby("Row_ID")["Prediction"].apply(list).reset_index()
            localism_final_predictions = ["".join(item) for item in df_exploded.groupby("Row_ID")["Prediction"].apply(list).reset_index()["Prediction"].tolist()]
            consistency_dict, examples_dict = get_consistency_score_localism(all_predictions, localism_final_predictions, test_dataset)
            consistency_dicts[k] = consistency_dict
            examples_dicts[k] = examples_dict
        elif split == "substitutivity":
            sub_predictions = get_predictions(model, tokenizer, test_dataset_sub, batch_size = batch_size, gemma_model=gemma_model)
            accuracy_sub, mistakes_list_sub = get_accuracy_default(sub_predictions, test_dataset_sub)
            accuracy_dict_sub[k] = accuracy_sub
            mistakes_dict_sub[k] = mistakes_list_sub
            consistency_dict, examples_dict = get_consistency_score_substitutivity(all_predictions, sub_predictions, test_dataset, synonyms_dict=synonyms_dict)
            consistency_dicts[k] = consistency_dict
            examples_dicts[k] = examples_dict
        elif split in ["overgeneralisation", "overgeneralization"]:
            matches_dict, word_matches_dict = get_accuracy_overgeneralisation(all_predictions, test_dataset, column_name_correct_original = column_name_correct_original,
                                                             exceptions_dict = exceptions_dict)
            exceptions_dicts[k] = matches_dict
            exceptions_word_dicts[k] = word_matches_dict
            
            
            

        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    # Create all the necessary folders for saving the results
    my_save_path = Path(f"{save_folder_path}/{adapter_folder}/accuracy.json")

    my_save_path.parent.mkdir(parents=True, exist_ok=True)

    if split != "localism":
        #Save accuracy scores
        with open(f"{save_folder_path}/{adapter_folder}/accuracy.json", "w", encoding="utf-8") as file:
            json.dump(accuracy_dict, file, indent=4)

        #Save mistakes list
        with open(f"{save_folder_path}/{adapter_folder}/errors.json", "w", encoding="utf-8") as file:
            json.dump(mistakes_dict, file, indent=4, ensure_ascii=False)

    if split == "productivity":
        with open(f"{save_folder_path}/{adapter_folder}/accuracy_per_length_detailed.json", "w", encoding="utf-8") as file:
            json.dump(accuracy_dict_per_length_detailed, file, indent=4)

    elif split in ["localism", "substitutivity"]:
        with open(f"{save_folder_path}/{adapter_folder}/{split}_consistency_scores.json", "w", encoding="utf-8") as file:
            json.dump(consistency_dicts, file, indent=4)
        
        with open(f"{save_folder_path}/{adapter_folder}/{split}_examples.json", "w", encoding="utf-8") as file:
            json.dump(examples_dicts, file, indent=4)
        

        if split == "substitutivity":
            with open(f"{save_folder_path}/{adapter_folder}/accuracy_sub.json", "w", encoding="utf-8") as file:
                json.dump(accuracy_dict_sub, file, indent=4)

            with open(f"{save_folder_path}/{adapter_folder}/errors_sub.json", "w", encoding="utf-8") as file:
                json.dump(mistakes_dict_sub, file, indent=4, ensure_ascii=False)

    elif split in ["overgeneralisation", "overgeneralization"]:
        with open(f"{save_folder_path}/{adapter_folder}/accuracy_exceptions.json", "w", encoding="utf-8") as file:
            json.dump(exceptions_dicts, file, indent=4)
        
        with open(f"{save_folder_path}/{adapter_folder}/examples.json", "w", encoding="utf-8") as file:
            json.dump(exceptions_word_dicts, file, indent=4)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating adapters")
    
    
    parser.add_argument("--model_id", type=str, required=True, help="Model to use")
    parser.add_argument("--adapter_folder", type=str, required=True, help="Path that contains the checkpoints of a specific adapter")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--save_folder_path", type=str, required=True, help="Path to folder where the results will be saved")
    parser.add_argument("--split", type=str, required=True, help="Which data split we're testing. One of the following: random, DBCA, systematicity, localism, substitutivity, productivity, overgeneralization")
    parser.add_argument("--max_train_length", type=int, default=12, help="The maximum length of compounds the adapter has been trained on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--batch_size_localism", type=int, default=128, help="Batch size for the localism split")
    parser.add_argument("--other_test_data_path", type=str, default=None, help="Path to the second test dataset for the localism and substitutivity splits")
    parser.add_argument("--column_name_inputs", type=str, default="Unimorph_features_lemma", help="Column that contains the input strings")
    parser.add_argument("--column_name_correct", type=str, default="Inflected_forms", help="Column that contains the correct inflected forms")
    parser.add_argument("--column_name_correct_original", type=str, default="Inflected_forms_original", help="Column that contains the original correct inflected forms for the exceptions split")
    parser.add_argument('--synonyms_dict', type=json.loads, help='Pass a JSON formatted dictionary string that contains the original features as keys and the synonymous features as values')
    parser.add_argument('--exceptions_dict', type=json.loads, help='Pass a JSON formatted dictionary string that contains the original features as keys and dictionaries with the keys value and condition where value is the new feature and condition shows in which cases the exception applies')
    parser.add_argument("--gemma_model", action="store_true")

    args = parser.parse_args()

    assert args.split in ["random", "DBCA", "localism", "substitutivity", "productivity", "systematicity", "overgeneralization", "overgeneralisation"]

    if args.split in ["localism", "substitutivity"]:
        assert args.other_test_data_path
    
    print("Model ID:", args.model_id)
    print("Adapter folder:", args.adapter_folder)
    print("Test data path:", args.test_data_path)
    print("Split:", args.split)

    main_evaluate(args.model_id, 
                  args.adapter_folder, 
                  args.test_data_path, 
                  args.save_folder_path, 
                  split=args.split, 
                  max_train_len=args.max_train_length,
                  batch_size = args.batch_size,
                  batch_size_localism = args.batch_size_localism,
                  other_test_data_path=args.other_test_data_path,
                  column_name_correct = args.column_name_correct, 
                  column_name_correct_original = args.column_name_correct_original, 
                  column_name_inputs=args.column_name_inputs,
                  synonyms_dict = args.synonyms_dict,
                  exceptions_dict = args.exceptions_dict,
                  gemma_model=args.gemma_model)
    