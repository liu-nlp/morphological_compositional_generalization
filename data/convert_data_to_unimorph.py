import os
import shutil
import pandas as pd
import requests
from pathlib import Path
from estnltk import Text
from estnltk.vabamorf.morf import synthesize
from estnltk.taggers import VabamorfAnalyzer
from estnltk_neural.taggers import GliLemTagger
import sparv
import xml.etree.ElementTree as ET
import xmltodict

# Processing Estonian data
def call_ekilex_api(url, headers):
    """Calls the Ekilex API (https://github.com/keeleinstituut/ekilex/wiki/Ekilex-API)

    Args:
        url (str): URL to use for calling the API
        headers (dict): dictionary with the key "ekilex-api-key" whose value is the user's Ekilex API key

    Returns:
        data (dict): dictionary containing information about a word; if the API call fails, returns the string of the error text instead
    """
    response = requests.get(url, headers=headers)

    # Check the response status
    if response.status_code == 200:
        data = response.json()  # Parse JSON response
        return data
    else:
        print(f"Request failed with status code {response.status_code}")
        return response.text


def get_adj_lemma(comp_sup_lemma, ekilex_api_key, comp_form = "posit"):
    """Gets a different degree of an adjective (positive/comparative/superlative) using the Ekilex API.

    Args:
        comp_sup_lemma (str): The original lemma of an adjective given by estNLTK
        ekilex_api_key (str): The user's Ekilex API key
        comp_form (str):  The degree of the required lemma (posit/komp/superl)

    Returns:
        lemma (str): The lemma for the required degree of the adjective, None if that was not found
    """

    headers = {"ekilex-api-key": ekilex_api_key}
    url_wordid = f"https://ekilex.ee//api/word/search/{comp_sup_lemma}"
    data_wordid = call_ekilex_api(url_wordid, headers)
    if data_wordid["totalCount"] == 1:
        wordid = data_wordid["words"][0]["wordId"]
        url_details = f"https://ekilex.ee//api/word/details/{wordid}"
        data_comp_lemma = call_ekilex_api(url_details, headers)
        for dictionary in data_comp_lemma["wordRelationDetails"]["secondaryWordRelationGroups"]:
            if "groupTypeCode" in dictionary and dictionary["groupTypeCode"] == comp_form:
                for item in dictionary["members"]:
                    if " " not in item["wordValue"]:
                        lemma = item["wordValue"]
                        return lemma
    return None
    
def get_vabamorf_to_unimorph_dict(filepath):
    """Creates a dictionary that converts Vabamorf's morphological tags to UniMorph.

    Args:
        filepath (str): Path to file with columns "vabamorf_cell" and "unimorph" that maps morphological tags between the two

    Returns:
        vabamorf_to_unimorph_dict (dict): Dictionary that can be used to convert morphological tags for verbs, nouns and adjectives 
        from Vabamorfo to UniMorph format.
    """
    df_vabamorf_to_unimorph = pd.read_csv(filepath)

    vabamorf_to_unimorph_dict = {"V" : {}, "N" : {}, "ADJ_POS" : {}, "ADJ_COMP" : {}, "ADJ_SUP" : {}}
    for vabamorf_item, unimorph_item in zip(df_vabamorf_to_unimorph["vabamorf_cell"].tolist(), df_vabamorf_to_unimorph["unimorph"].tolist()):
        if type(unimorph_item) == str and type(vabamorf_item) == str:
            if unimorph_item.startswith("V"):
                vabamorf_to_unimorph_dict["V"][vabamorf_item] = unimorph_item
            elif unimorph_item.startswith("N"):
                vabamorf_to_unimorph_dict["N"][vabamorf_item] = unimorph_item
                vabamorf_to_unimorph_dict["ADJ_POS"][vabamorf_item] = ";".join(["ADJ", "POS"] + unimorph_item.split(";")[1:])
                vabamorf_to_unimorph_dict["ADJ_COMP"][vabamorf_item] = ";".join(["ADJ", "CMPR"] + unimorph_item.split(";")[1:])
                vabamorf_to_unimorph_dict["ADJ_SUP"][vabamorf_item] = ";".join(["ADJ", "SPRL"] + unimorph_item.split(";")[1:])
    return vabamorf_to_unimorph_dict

def get_unimorph_to_vabamorf_dict(filepath):
    """Creates a dictionary that converts UniMorph's morphological tags to Vabamorf.

    Args:
        filepath (str): Path to file with columns "vabamorf_cell" and "unimorph" that maps morphological tags between the two

    Returns:
        unimorph_to_vabamorf_dict (dict): Dictionary that can be used to convert morphological tags for verbs, nouns and adjectives 
        from UniMorph to Vabamorf format.
    """
    df_vabamorf_to_unimorph = pd.read_csv(filepath)

    unimorph_to_vabamorf_dict = {}
    for vabamorf_item, unimorph_item in zip(df_vabamorf_to_unimorph["vabamorf_cell"].tolist(), df_vabamorf_to_unimorph["unimorph"].tolist()):
        unimorph_to_vabamorf_dict[unimorph_item] = vabamorf_item
        
    return unimorph_to_vabamorf_dict


def convert_sent_to_unimorph(tagged_sent, conversion_dict, ekilex_api_key):
    """Turns a sentence tagged with estNLTK into a list of lemma and UniMorph feature combinations and 
    another list with the resulting inflected forms.

    Args:
        tagged_sent (estnltk.text.Text): EstNLTK's Text object containing a tagged sentence
        conversion_dict (dict): Dictionary for converting Vabamorf's morphological tags to UniMorph features
        ekilex_api_key (str): The user's Ekilex API key

    Returns:
        text_tokens [List[str]]: List of compounds consisting of a lemma and a set of Unimorph features
        processed_tokens [List[str]]: List of word forms after applying the Unimorph features to the lemma. Alternatives are separated using |.
        original_tokens [List[str]]: List of word forms from the original text.
    """
    original_tokens = []
    text_tokens = []
    processed_tokens = []

    for token in tagged_sent["morph_analysis"]:
        token_text, token_lemma, token_form, token_pos = token.text, token.lemma[0], token.form[0], token.partofspeech[0]
        token_lemma_pos = "Irrelevant"
        if len(token_text) > 1:
            if token_form == "adt":
                token_form = "sg ill"
            if token_pos in ["U", "C", "A", "S", "V"] and token_form not in ['', '?'] and token_text.isalpha():
                if token_pos == "U":
                    pos = "ADJ_SUP"
                    token_lemma_pos = get_adj_lemma(token_lemma, ekilex_api_key, comp_form = "posit")
                elif token_pos == "C":
                    pos = "ADJ_COMP"
                    token_lemma_pos = get_adj_lemma(token_lemma, ekilex_api_key, comp_form = "posit")
                elif token_pos == "A":
                    pos = "ADJ_POS"
                elif token_pos == "S":
                    pos = "N"
                else:
                    pos = token_pos
                if token_form in conversion_dict[pos]:
                    unimorph_features = conversion_dict[pos][token_form]
                    possible_forms = synthesize(token_lemma, token_form)
                    if token_form == "sg ill":
                        possible_forms += synthesize(token_lemma, "adt")
                    possible_forms = list(set(possible_forms))
                    if unimorph_features and possible_forms and token_lemma_pos:
                        original_tokens.append(token_text.lower())
                        text_tokens.append("|".join(possible_forms))
                        if pos in ["ADJ_SUP", "ADJ_COMP"]:
                            processed_tokens.append(f"{unimorph_features}({token_lemma_pos})")
                        else:
                            processed_tokens.append(f"{unimorph_features}({token_lemma})")
    return text_tokens, processed_tokens, original_tokens

def tag_convert_to_unimorph_estonian(text_list, filepath_conversion, ekilex_api_key):
    """Main function for tagging Estonian text and converting it to UniMorph format. Takes a list of text strings, tags them using estNLTK and
      converts inflected forms into UniMorph format consisting of a lemma and a set of UniMorph features.

    Args:
        text_list (List[str]): List of text strings to be processed
        filepath_conversion (str): Path to file with columns "vabamorf_cell" and "unimorph" that maps morphological tags between the two
        ekilex_api_key (str): The user's Ekilex API key

    Returns:
        df_estonian_processed (DataFrame): Pandas DataFrame with two columns where the first column contains compounds consisting of lemmas and
        UniMorph features and the second column contains the corresponding inflected forms
    """
    glilem_disambiguator = GliLemTagger(output_layer='morph_analysis', disambiguate=True)
    morph_analyzer = VabamorfAnalyzer(guess=False, propername=False) ###
    vabamorf_to_unimorph_dict = get_vabamorf_to_unimorph_dict(filepath_conversion)
    
    #corpus = [Text(t).tag_layer(['morph_analysis']) for t in text_list] ###
    corpus = [Text(t).tag_layer(['words','sentences']) for t in text_list] ###
    corpus = [morph_analyzer.tag(t) for t in corpus] ####
    corpus = [glilem_disambiguator.retag(text) for text in corpus]

    original_texts = []
    sentences_original = []
    sentences_before = []
    sentences_after = []

    num_processed = 0
    for i, item in enumerate(corpus):
        text_tokens, processed_tokens, original_tokens = convert_sent_to_unimorph(item, vabamorf_to_unimorph_dict, ekilex_api_key)
        if text_tokens and processed_tokens and original_tokens and item.text:
            original_texts.append(item.text)
            sentences_before.append(' '.join(processed_tokens))
            sentences_after.append(' '.join(text_tokens))
            sentences_original.append(' '.join(original_tokens))
        num_processed += 1
        if num_processed % 1000 == 0:
            print(f"Done processing {num_processed} documents.")

    df_estonian_processed = pd.DataFrame()
    df_estonian_processed["Original_text"] = original_texts
    df_estonian_processed["Original_tokens"] = sentences_original
    df_estonian_processed["Unimorph_features_lemma"] = sentences_before
    df_estonian_processed["Inflected_forms"] = sentences_after
    df_estonian_processed.dropna(inplace=True)
    df_estonian_processed.drop_duplicates(inplace=True)
    df_estonian_processed.reset_index(inplace=True, drop=True)

    return df_estonian_processed

# Processing Swedish data
def prepare_swedish_for_tagging(text_list, destination_path, chunk_size = 1000):
    """Prepares a Swedish corpus for tagging with Sparv by converting to XML-format and saving to the destination path.

    Args:
        text_list (List[str]): List of text strings to be processed
        destination_path (str): Destination path for the Swedish corpus in XML-format
        chunk_size (int): The number of documents to include in a single file

    Returns: 
        None, writes the corpus to the destination path
    """
    for i in range(0, len(text_list), chunk_size):
        chunk = text_list[i : i + chunk_size]
    

        root = ET.Element("corpus")
    
        for text in chunk:
            doc = ET.SubElement(root, "document")
            doc.text = text 

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)

        write_path = f"{destination_path}corpus_{(i//chunk_size)+1}.xml"
    
        tree.write(write_path, encoding="utf-8", xml_declaration=True)


def get_ud_to_unimorph_dict(filepath):
    """Prepares a dictionary for converting morphological tags from Universal Dependencies to UniMorph format.

    Args:
        filepath (str): Path to file with columns "UD" and "UniMorph" that maps morphological tags between the two

    Returns: 
        ufeats_to_unimorph_dict (dict): Dictionary with UD tags as keys and UniMorph features as values
    """
    df_ufeats_to_unimorph = pd.read_csv(filepath, sep='\t', header=0)

    ufeats_to_unimorph_dict = {}
    for ufeats_item, unimorph_item in zip(df_ufeats_to_unimorph["UD"].tolist(), df_ufeats_to_unimorph["UniMorph"].tolist()):
        if type(ufeats_item) == str and type(unimorph_item) == str:
            ufeats_to_unimorph_dict[ufeats_item] = unimorph_item

    ufeats_to_unimorph_dict["Degree=Pos"] = "POS"
    ufeats_to_unimorph_dict["Degree=Sup"] = "SPRL"
    return ufeats_to_unimorph_dict


def get_one_feat_value(feat_name, feat_value):
    """Takes in a UD morphological feature name and its value and returns a single feature value in ambiguous cases.

    Args:
        feat_name (str): The name of the morphological feature in UD format.
        feat_value (str): The value of the morphological feature returned by a tagger in UD format.

    Returns: 
        feat_value (str): A single feature value
    """
    if "," in feat_value:
        feat_values = feat_value.split(",")
        if feat_name == "Case": 
            feat_values = [item for item in feat_value.split(",") if item in ["Nom", "Gen"]]
        return feat_values[0] #TODO: currently always taking the first one if there are several
    return feat_value
        


def convert_ufeats_to_unimorph_nouns(feat_str, ufeats_to_unimorph_dict):
    """Converts a morphological feature set for a noun from UD to UniMorph format.

    Args:
        feat_str (str): A UD feature set for a noun
        ufeats_to_unimorph_dict (dict): Dictionary for converting morphological features from UD to UniMorph

    Returns: 
        unimorph_features (str): A UniMorph feature set
    """
    feats_list = feat_str.split("|")
    unimorph_features = None
    case = None
    number = None
    definite = None

    for item in feats_list:
        feat_name, feat_value = item.split("=")
        feat_value = get_one_feat_value(feat_name, feat_value)
        feat = f"{feat_name}={feat_value}"
        if feat_name == "Case":
            case = ufeats_to_unimorph_dict[feat]
        elif feat_name == "Number":
            number = ufeats_to_unimorph_dict[feat]
        elif feat_name == "Definite":
            definite = ufeats_to_unimorph_dict[feat]
        
        if case and number and definite:
            unimorph_features = ["N", case, number, definite]
            unimorph_features = ";".join(unimorph_features)

    return unimorph_features

def convert_ufeats_to_unimorph_verbs(feat_str, ufeats_to_unimorph_dict):
    """Converts a morphological feature set for a verb from UD to UniMorph format.

    Args:
        feat_str (str): A UD feature set for a verb
        ufeats_to_unimorph_dict (dict): Dictionary for converting morphological features from UD to UniMorph

    Returns: 
        unimorph_features (str): A UniMorph feature set
    """
    feats_list = feat_str.split("|")
    unimorph_features = None
    mood = None
    tense = None
    verb_form = None
    voice = None
    
    for item in feats_list:
        feat_name, feat_value = item.split("=")
        feat_value = get_one_feat_value(feat_name, feat_value)
        feat = f"{feat_name}={feat_value}"
        if feat_name == "Mood":
            mood = ufeats_to_unimorph_dict[feat]
        elif feat_name == "Tense":
            tense = ufeats_to_unimorph_dict[feat]
        elif feat_name == "VerbForm":
            verb_form = ufeats_to_unimorph_dict[feat]
        elif feat_name == "Voice":
            voice = ufeats_to_unimorph_dict[feat]

        
        if mood and tense and voice:
            unimorph_features = ["V", mood, voice, tense]
        elif mood and verb_form and voice and not tense:
            unimorph_features = ["V", mood, voice]
        elif verb_form and voice and not mood and not tense:
            unimorph_features = ["V", verb_form, voice]
    
    if unimorph_features:
        unimorph_features = ";".join(unimorph_features)
    return unimorph_features

def convert_ufeats_to_unimorph_adj(feat_str, ufeats_to_unimorph_dict):
    """Converts a morphological feature set for an adjective from UD to UniMorph format.

    Args:
        feat_str (str): A UD feature set for an adjective
        ufeats_to_unimorph_dict (dict): Dictionary for converting morphological features from UD to UniMorph

    Returns: 
        unimorph_features (str): A UniMorph feature set
    """
    feats_list = feat_str.split("|")
    unimorph_features = None
    number = None
    definite = None
    gender = None
    degree = None
    tense = None
    verbform = None
    

    for item in feats_list:
        feat_name, feat_value = item.split("=")
        feat_value = get_one_feat_value(feat_name, feat_value)
        feat = f"{feat_name}={feat_value}"
        if feat_name == "Number":
            number = ufeats_to_unimorph_dict[feat]
        elif feat_name == "Definite":
            definite = ufeats_to_unimorph_dict[feat]
        elif feat_name == "Gender":
            gender = ufeats_to_unimorph_dict[feat]
        elif feat_name == "Degree":
            degree = ufeats_to_unimorph_dict[feat]

    if number == "SG":
        unimorph_features = ["ADJ", degree, gender, number, definite]
    else:
        unimorph_features = ["ADJ", degree, number, definite]
    
    if unimorph_features:
        unimorph_features = ";".join([item for item in unimorph_features if item])
        
    return unimorph_features

def convert_ufeats_to_unimorph_pc(feat_str, ufeats_to_unimorph_dict):
    """Converts a morphological feature set for a particle from UD to UniMorph format.

    Args:
        feat_str (str): A UD feature set for a particle
        ufeats_to_unimorph_dict (dict): Dictionary for converting morphological features from UD to UniMorph

    Returns: 
        unimorph_features (str): A UniMorph feature set
    """
    feats_list = feat_str.split("|")
    unimorph_features = None
    tense = None
    verbform = None
    number = None
    definite = None
    gender = None
    

    for item in feats_list:
        feat_name, feat_value = item.split("=")
        feat_value = get_one_feat_value(feat_name, feat_value)
        feat = f"{feat_name}={feat_value}"
        if feat_name == "Tense":
            tense = ufeats_to_unimorph_dict[feat]
        elif feat_name == "VerbForm":
            verbform = ufeats_to_unimorph_dict[feat]
        if "Tense=Past" in feat_str:
            if feat_name == "Number":
                number = ufeats_to_unimorph_dict[feat]
            elif feat_name == "Definite":
                definite = ufeats_to_unimorph_dict[feat]
            elif feat_name == "Gender":
                gender = ufeats_to_unimorph_dict[feat]
            
    
    if number and definite and gender and tense and verbform:
        if number == "SG":
            unimorph_features = ["V", verbform, tense, gender, number, definite]
        else:
            unimorph_features = ["V", verbform, tense, number, definite]
    elif tense and verbform:
        unimorph_features = ["V", verbform, tense]
    if unimorph_features:
        unimorph_features = ";".join(unimorph_features)
        
    return unimorph_features
    



def tag_swedish_text_with_sparv(corpus_folder):
    """Runs Sparv on a Swedish corpus

    Args:
        corpus_folder (str): Path to a folder containing the Swedish corpus to annotate in corpus_folder/source/ and 
        a config.yaml file in corpus_foler

    Returns: 
        None, uses Sparv to annotate a Swedish corpus and saves the resulting XML-file in corpus_folder/export/
    """
    original_directory = os.getcwd()
    try:
        os.chdir(corpus_folder)

        with sparv.call(["run", "xml_export:pretty", "--log"]) as sparv_call:
            for log_message in sparv_call:
                print(log_message)
            
            success = sparv_call.get_return_value()
            if success:
                print("Pipeline finished successfully!")
            else:
                print("Pipeline failed!")

    finally:
        os.chdir(original_directory)
    

def get_token_unimorph_features_swedish(token, ud_to_unimorph_dict):
    """Converts Sparv's annotation in UD format into UniMorph format for a specific token

    Args:
        token: XML-element, annotated with UD morphological tags
        ud_to_unimorph_dict: dictionary for converting from UD to UniMorph

    Returns:
        token_text (str): Word form
        token_lemma (str): Lemma
        token_pos (str): POS-tag
        token_complemgram (str): Lemgrams of the parts of compound words
        token_compwf (str): Compound words split into parts
        unimorph_features (str): Set of UniMorph features
    """
    unimorph_features = None
            
    token_data = token.attrib.copy()
        
    token_text = token.text
    token_lemma = token_data['baseform'][1:-1].split('|')[0]
    token_form = token_data['ufeats'][1:-1]
    token_pos = token_data['pos'] ## NN, VB, JJ, PC
    token_complemgram = token_data['complemgram']
    token_compwf = token_data['compwf']

    if token_text.isalpha() and token_lemma.isalpha() and token_form and len(token_text) > 1:
        if token_pos == 'NN':
            unimorph_features = convert_ufeats_to_unimorph_nouns(token_form, ud_to_unimorph_dict)
        elif token_pos == 'VB':
            unimorph_features = convert_ufeats_to_unimorph_verbs(token_form, ud_to_unimorph_dict)
        elif token_pos == 'JJ':
            unimorph_features = convert_ufeats_to_unimorph_adj(token_form, ud_to_unimorph_dict)
        elif token_pos == 'PC':
            unimorph_features = convert_ufeats_to_unimorph_pc(token_form, ud_to_unimorph_dict)
    return token_text, token_lemma, token_pos, token_complemgram, token_compwf, unimorph_features


def convert_saldom_to_dict(saldom_path):
    """Converts SALDO's morphological lexicon into a nested dictionary with lemmas as the first keys, morphological features as
    the second keys and inflected forms as the values.

    Args:
        saldom_path (str): Path to SALDO's morphological lexicon

    Returns:
        saldom_dict (dict): Dictionary for efficiently extracting inflected forms of a specific lemma.
    """
    saldom_dict = {"nn" : {}, "vb" : {}, "av" : {}}

    with open(saldom_path, 'r', encoding='utf-8') as file:
        xml_file = file.read()

    xml_dict = xmltodict.parse(xml_file)

    for item in xml_dict["LexicalResource"]["Lexicon"]["LexicalEntry"]:
        lemma_list = item["Lemma"]["FormRepresentation"]["feat"]
        lemma, pos_tag = lemma_list[0]["@val"], lemma_list[2]["@val"]
        if pos_tag in ["nn", "vb", "av"]:
            if lemma not in saldom_dict[pos_tag]:
                saldom_dict[pos_tag][lemma] = dict()

            #Add word forms
            word_forms_list = item["WordForm"]
            if type(word_forms_list) == dict:
                word_forms_list = [word_forms_list]
            for word_f in word_forms_list:
                word_form = word_f["feat"][0]["@val"]
                tags = word_f["feat"][1]["@val"]
                if not tags in saldom_dict[pos_tag][lemma]:
                    saldom_dict[pos_tag][lemma][tags] = {word_form}
                else:
                    saldom_dict[pos_tag][lemma][tags].add(word_form)
    return saldom_dict
    

def get_possible_forms_saldo(token_lemma, token_pos, token_complemgram, token_compwf, saldo_features, saldom_dict):
    """Extracts all possible inflected forms of a lemma given a set of morphological features.

    Args:
        token_lemma (str): Lemma
        token_pos (str): POS-tag 
        token_complemgram (str): Lemgrams of the parts of compound words
        token_compwf (str): Compound words split into parts
        saldo_features (str): Morphological features in SALDO-format
        saldom_dict (dict): Dictionary built from SALDO's morphological lexicon for extracting inflected forms of a lemma 

    Returns:
        possible_forms (str): All possible inflected forms given a lemma and a set of morphological features, separated by | 
    """
    target_pos_dict = {"NN" : "nn", "VB": "vb", "PC" : "vb", "JJ" : "av"}
    target_pos = target_pos_dict[token_pos]
    possible_forms = []

    if target_pos in saldom_dict:
        if token_lemma in saldom_dict[target_pos]:
            if saldo_features in saldom_dict[target_pos][token_lemma]:
                possible_forms = list(saldom_dict[target_pos][token_lemma][saldo_features])
        #Check for compound word, TODO: not sure if it works in all cases
        elif token_complemgram and token_compwf: 
            last_lemma = token_complemgram[1:-1].split("|")[0].split("+")[-1].split("..")[0]
            first_part = ''.join(token_compwf[1:-1].split("|")[0].split("+")[:-1])
            if last_lemma in saldom_dict[target_pos] and first_part+last_lemma == token_lemma:
                if saldo_features in saldom_dict[target_pos][last_lemma]:
                    possible_forms = list(saldom_dict[target_pos][last_lemma][saldo_features])
                    possible_forms = [first_part + form for form in possible_forms]

    if possible_forms:
        possible_forms = [item.lower() for item in possible_forms]
        possible_forms = list(set(possible_forms))
                
    return '|'.join(possible_forms)
    

def convert_unimorph_feats_to_saldo(token_pos, unimorph_feats):
    """Converts a token's UniMorph features into SALDO format.

    Args:
        token_pos (str): Token's POS-tag
        unimorph_feats (str): Set of UniMorph features

    Returns:
        feat_str (str): UniMorph features converted into SALDO format
    """
    
    msd_conversion_dict = {"NN" : {"SG" : "sg", "PL": "pl", "INDF" : "indef", "DEF" : "def", "NOM" : "nom", "GEN" : "gen"},
                           "VB" : {"PST" : "pret", "PRS" : "pres", "IND" : "ind", "IMP": "imper", "NFIN" : "inf", 
                                    "V.PTCP" : "part", "V.CVB" : "sup", "ACT" : "aktiv", "PASS" : "s-form"
                                  },
                           "JJ" : {"POS" : "pos", "CMPR" : "komp", "SPRL" : "super", "INDF" : "indef", "DEF" : "def", "SG" : "sg", "PL": "pl",
                                   "MASC+FEM" : "u", "NEUT" : "n", "MASC" : "masc", "NOM" : "nom", "GEN" : "gen"
                                  }
                          }

    if "V.PTCP;PRS" in unimorph_feats:
        return "pres_part nom"
    feat_list = []
    if "V.PTCP;PST" in unimorph_feats:
        feat_list.append("pret_part")
        
    if token_pos == "PC":
        token_pos = "JJ"
    feat_dict = msd_conversion_dict[token_pos]
    unimorph_feat_list = unimorph_feats.split(";")
    for unimorph_feature, saldo_feature in feat_dict.items():
        if unimorph_feature in unimorph_feat_list:
            feat_list.append(saldo_feature)

    if token_pos == "JJ":
        feat_list.append("nom")

    if "komp" in feat_list:
        feat_list = [item for item in feat_list if item in ["komp", "nom", "gen"]]
    elif "super" in feat_list:
        feat_list = [item if item != "pl" else "no_masc" for item in feat_list]
        feat_list = [item for item in feat_list if item in ["super", "def", "indef", "nom", "gen", "masc", "no_masc"]]
        
    feat_str = " ".join(feat_list)
    return feat_str

def tag_convert_to_unimorph_swedish(text_list, path_corpus_folder, config_path, filepath_conversion, saldom_path, chunk_size):
    """Main function for tagging a Swedish corpus and converting it to UniMorph format.

    Args:
        text_list (List[str]): List of text strings to process
        path_corpus_folder (str): Path to the folder that contains the Swedish corpus in XML-format
        config_path (str): Path to config.yaml for Sparv
        filepath_conversion (str): Path to file with columns "UD" and "UniMorph" that maps the morphological features between the two.
        saldom_path (str): Path to SALDO's morphological lexicon
        chunk_size(int): The number of documents to save in a single XML-file when preparing the data for tagging

    Returns:
        df_swedish_processed (DataFrame): Pandas DataFrame with two columns where the first column contains compounds consisting of lemmas and
        UniMorph features and the second column contains the corresponding inflected forms
    """

    data_folder = path_corpus_folder + "source/"
    path = Path(data_folder)
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, path_corpus_folder + "config.yaml")
    prepare_swedish_for_tagging(text_list, data_folder, chunk_size=chunk_size)
    tag_swedish_text_with_sparv(path_corpus_folder)
    
    ud_to_unimorph_dict = get_ud_to_unimorph_dict(filepath_conversion)
    saldom_dict = convert_saldom_to_dict(saldom_path)

    original_texts = []
    sentences_original = []
    sentences_before = []
    sentences_after = []

    filenames = os.listdir(path_corpus_folder + 'export/xml_export.pretty/')
    filenames = sorted(filenames, key=lambda x: int(x.split('_')[1]))

    for filename in filenames:
        if filename.endswith("export.xml"):
            tree = ET.parse(path_corpus_folder + 'export/xml_export.pretty/' + filename)
            root = tree.getroot()

            tree_original = ET.parse(data_folder + filename.replace('_export', ''))
            root_original = tree_original.getroot()

            for i, (doc, doc_orig) in enumerate(zip(root.findall('document'), root_original.findall('document'))):
                text_tokens = []
                processed_tokens = []
                original_tokens = []
                for token in doc.findall('token'):
                    token_text, token_lemma, token_pos, token_complemgram, \
                    token_compwf, unimorph_features = get_token_unimorph_features_swedish(token, ud_to_unimorph_dict)     
                    if unimorph_features:
                        saldo_features = convert_unimorph_feats_to_saldo(token_pos, unimorph_features)
                        possible_forms = get_possible_forms_saldo(token_lemma, token_pos, token_complemgram, 
                                                                  token_compwf, saldo_features, saldom_dict)
                        if possible_forms:
                            original_tokens.append(token_text.lower())
                            text_tokens.append(possible_forms)
                            processed_tokens.append(f"{unimorph_features}({token_lemma.lower()})")

                if text_tokens and processed_tokens and original_tokens and doc_orig.text:
                    original_texts.append(doc_orig.text)
                    sentences_before.append(' '.join(processed_tokens))
                    sentences_after.append(' '.join(text_tokens))
                    sentences_original.append(' '.join(original_tokens))

    df_swedish_processed = pd.DataFrame()
    df_swedish_processed["Original_text"] = original_texts
    df_swedish_processed["Original_tokens"] = sentences_original
    df_swedish_processed['Unimorph_features_lemma'] = sentences_before
    df_swedish_processed['Inflected_forms'] = sentences_after
    df_swedish_processed.dropna(inplace=True)
    df_swedish_processed.drop_duplicates(inplace=True)
    df_swedish_processed.reset_index(inplace=True, drop=True)

    return df_swedish_processed