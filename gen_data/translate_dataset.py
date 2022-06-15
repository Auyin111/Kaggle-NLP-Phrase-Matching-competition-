from googletrans import Translator
from deep_translator import GoogleTranslator

from cfg import Cfg
from gen_data.dataset import merge_context2
import os
import pandas as pd
import math

cfg = Cfg("translate", False, True, "cuda")
df_train = pd.read_csv("../kaggle/input/train.csv")

pd.options.display.max_colwidth = 1000
pd.set_option('display.max_columns', None)

def translate(words, language):
    translator = Translator()
    return translator.translate(str(words), dest=language).text

def translate2(words, language):
    translated = GoogleTranslator(source='auto', target=language).translate(words)
    return translated

def translate_context(language):

    grp1_df = pd.read_csv("../own_dataset/df_context_grp_1.csv")
    print("Translating group 1 context")
    grp1_df["text_grp_1"] = grp1_df["text_grp_1"].apply(lambda x: translate(x, language))
    gp1_1_file_name = f"df_context_grp_1_{language}"
    grp1_df.to_csv(os.path.join("../own_dataset", gp1_1_file_name), index=False)
    print("Group 1 context translated and new csv file created")

    grp2_df = pd.read_csv("../own_dataset/df_context_grp_2.csv")
    print("Translating group 2 context")
    grp2_df["description"] = grp2_df["description"].apply(lambda x: translate(x, language))
    gp1_2_file_name = f"df_context_grp_2_{language}"
    grp2_df.to_csv(os.path.join("../own_dataset", gp1_2_file_name), index=False)
    print("Group 2 context translated and new csv file created")


#translate_context("zh-tw")

def translate_anchor(df_train, language):
    anchor = df_train["anchor"].unique().tolist()
    anchor_translate_dict = {a: translate2(a, language) for a in anchor}
    anchor_translate_pd = pd.DataFrame.from_dict(anchor_translate_dict, orient="index").reset_index()
    anchor_translate_pd = anchor_translate_pd.rename(columns={list(anchor_translate_pd)[0]: "en", list(anchor_translate_pd)[1]: language.lower()})
    anchor_translate_pd.to_csv(f"../own_dataset/translate_anchor_{language.lower()}.csv")


#translate_anchor(df_train, "zh-TW")

def translate_target(df_train, language, starting_batch):
    target_total = df_train["target"].unique().tolist()
    num_batch = int(math.ceil(len(target_total)/500))
    for b in range(starting_batch, num_batch):
        if b != num_batch-1:
            target = target_total.copy()[b*500: (b+1)*500]
        else:
            target = target_total.copy()[b*500:]
        print("=========================================================")
        print(f"Translating batch {b} (length: {len(target)})")
        target_translate_dict = {t: translate2(t, language) for t in target}
        target_translate_pd = pd.DataFrame.from_dict(target_translate_dict, orient="index")
        target_translate_pd.to_csv(f"../own_dataset/target/translate_target_{language.lower()}_{b}.csv")
        print(f"Batch {b} translated and saved")


#translate_target(df_train, "zh-TW", 37)

def merge_target_dict(language):
    num_batch = len(os.listdir("../own_dataset/target"))
    df = pd.DataFrame()
    for d in range(num_batch):
        df_b = pd.read_csv(f"../own_dataset/target/translate_target_{language.lower()}_{d}.csv")
        df = pd.concat([df, df_b])
    df = df.reset_index().rename(columns={list(df)[0]: "en", list(df)[1]: language.lower()})
    print(df.head)
    print(f"Length of the entire target translate dictionary: {len(df)}")
    df.to_csv(f"../own_dataset/translate_target_{language.lower()}.csv", index=False)
    print("Target translation csv created")


#merge_target_dict("zh-TW")

def create_translated_train(df_train, language):

    anchor_dict_df = pd.read_csv(f"../own_dataset/translate_anchor_{language.lower()}.csv")
    anchor_dict_list1 = anchor_dict_df[list(anchor_dict_df)[1]].to_list()
    anchor_dict_list2 = anchor_dict_df[list(anchor_dict_df)[2]].to_list()
    anchor_translate_dict = {anchor_dict_list1[i]: anchor_dict_list2[i] for i in range(len(anchor_dict_df))}

    target_dict_df = pd.read_csv(f"../own_dataset/translate_target_{language.lower()}.csv")
    target_dict_list1 = target_dict_df[list(target_dict_df)[1]]
    target_dict_list2 = target_dict_df[list(target_dict_df)[2]]
    target_translate_dict = {target_dict_list1[i]: target_dict_list2[i] for i in range(len(target_dict_df))}

    print("Mapping training set to the translating language:")
    df_train_translated = df_train.copy()
    df_train_translated["anchor"] = df_train_translated["anchor"].map(anchor_translate_dict)
    df_train_translated["target"] = df_train_translated["target"].map(target_translate_dict)
    df_train_translated.to_csv(f"../kaggle/input/train_translated.csv", index=False)
    print("Translated train.csv created")

create_translated_train(df_train, "zh-TW")










