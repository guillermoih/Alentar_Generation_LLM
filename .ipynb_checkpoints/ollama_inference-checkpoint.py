# -*- coding: utf-8 -*-
# +
import re
import os
import tqdm
import json
import time
import ollama
import pandas as pd

from json import JSONDecodeError
from prompts import start_prompt, example_prompt, end_prompt
from data_load import get_cods_dict
# -

if __name__ == "__main__":
    # Load metadata from .csv
    cod_names = get_cods_dict('Informes-Urgencias/vars.csv')
    
    # Load examples .csv and filter unrevised ones
    df = pd.read_csv('Informes-Urgencias/EJEI_nocod_Clasificator_2024_07_ 04. 4.fin.csv')
    df = df[df["REVISADO"] == 1]
    
    n_responses = 10
    n_examples = 10
    # model = ['deepseek-r1:7b_es', 'gpt-oss_es', 'meditron:70b_es']
    models = ['gemma3:27b_es']

    out_file = "llm_responses.csv"

    if not os.path.exists(out_file):
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("model|code|response|prompt|response_raw|time\n")

    for model in models:
        print('-'*50)
        print(model)

        pbar = tqdm.tqdm(cod_names)
        for cod in pbar:
            pbar.set_description(cod)
            prompt = start_prompt

            df_filter = df[df[cod]=='1'] # Filter rows containing the code
            df_filter = df_filter.sample(min(n_examples, len(df_filter)), random_state=42) # Get n_samples random rows
            for i, row in df_filter.iterrows():
                cods = row.index[row == '1'].tolist()
                cods_name = [cod_names[c] for c in cods if c in cod_names.keys()] # Filter non codes keys
                example = "'" + row['EJEIDiagnóstico'] + "'"

                prompt+=example_prompt.format(codigo=cods,
                                              nombre_codigo=cods_name,
                                              ejemplo=example)
                prompt+='\n'

            prompt+=end_prompt.format(cantidad=n_responses,
                                      codigo_objetivo=cod,
                                      nombre_codigo_objetivo=cod_names[cod])

            t1 = time.time()
            res = ollama.chat(
                model=model,
                messages=
                [{
                    'role': 'user',
                    'content': prompt,
                }],
            )
            t2 = time.time()

            response_raw = res['message']['content']
            
            with open(out_file, "a", encoding="utf-8") as f:
                try:
                    prompt_text = prompt.replace('\n','\\n').replace('\r','')
                    response_raw_text = response_raw.replace('\n','\\n').replace('\r','')
                    
                    response = re.search(r'\{.*\}', response_raw, re.DOTALL).group()
                    res_dict = json.loads(response)
                    for r in res_dict:
                        for c in res_dict[r]:
                            f.write(f"{model}|{c}|{res_dict[r][c]}|{prompt_text}|{response_raw_text}|{(t2-t1):.2f}\n")

                except:
                    print(f'Error parsing {response_raw_text}')
                    f.write(f"{model}|ERROR|ERROR|{prompt_text}|{response_raw_text}|{(t2-t1):.2f}\n")
