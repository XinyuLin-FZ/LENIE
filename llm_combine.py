import configparser
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import argparse
import torch
import os

def get_prompt(dataset_name, data_path):
    if dataset_name == 'FB15k':
        with open(data_path, 'r', encoding='utf-8') as file:  
            content = file.read()
        lines = content.split('\n')  
        header = lines[0].strip().split('\t')  
        data = [line.strip().split('\t') for line in lines[1:]]

        node_id2prompt = {}
        for row in data: 
            if len(row) > 1:
                node_id = row[0] 
                mid = row[1]
                entity_name = row[2].replace('_', ' ')
                des = row[3]
                kg = row[4]

                if entity_name != 'nan':
                    prompt = "{}\n{}\nPlease generate an overview of \"{}\".".format(kg, des, entity_name)
                else:
                    prompt = None

                node_id2prompt[node_id] = prompt             
    
    elif dataset_name == 'TMDB5K' or 'MUSIC10K':
        with open(data_path, 'r', encoding='utf-8') as file:  
            content = file.read()
        lines = content.split('\n')  
        header = lines[0].strip().split('\t')  
        data = [line.strip().split('\t') for line in lines[1:]]

        node_id2prompt = {}

        for row in data: 
            if len(row) > 1:
                node_id = row[0] 
                entity_name = row[1]
                des = row[2]
                kg = row[3]
                    
                if entity_name != 'nan':
                    prompt = "{}\n{}\nPlease generate an overview of \"{}\".".format(kg, des, entity_name)
                else:
                    prompt = None

                node_id2prompt[node_id] = prompt    
                
    return node_id2prompt


def get_data_dict(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:  
        content = file.read()
    lines = content.split('\n')  
    header = lines[0].strip().split('\t')  
    data = [line.strip().split('\t') for line in lines[1:]]
    node_id2text = {}  
    for row in data:  
        if len(row) > 1:
            node_id = row[0]  
            try:
                text = row[2]  
            except:
                text = 'Null'     
            node_id2text[node_id] = text
    return node_id2text

def llm_combine(node_id2prompt, output_path,args):    
    #'''
    # Llama3.1-8b-instruct
    model_id = "./LLM/Meta-Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        )
        
    with open(output_path, 'w', encoding='utf-8') as f:  
        f.write('node_id\tllm_response\n')
        for node_id in node_id2prompt.keys():  
            print(node_id)
            prompt = node_id2prompt[node_id]
            
            message=[{"role": "user", "content": prompt}]
            
            if prompt:
                #response, history = model.chat(tokenizer, prompt, history=[])   
                response = pipeline(message, max_new_tokens=512)[0]['generated_text'][-1]['content']
                response = response.replace('\n', ' ').replace('\t', '')
            else:
                response = None
            
            f.write(f'{node_id}\t{response}\n')
    #'''
    
    '''
    # GLM4-9b-chat
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/glm-4-9b-chat",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    
    with open(output_path, 'w', encoding='utf-8') as f:  
        f.write('node_id\tllm_response\n')
        
        for node_id in node_id2prompt.keys():  
            print(node_id)
            prompt = node_id2prompt[node_id]
            
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )
            inputs = inputs.to(device)
            
            
            if prompt:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.replace('\n', ' ').replace('\t', '')
            else:
                response = None
            
            f.write(f'{node_id}\t{response}\n')
    
    '''
    
    '''
    # Qwen2-7b-instruct
    model_name = "Qwen/Qwen2-7B-Instruct"
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    with open(output_path, 'w', encoding='utf-8') as f:  
        f.write('node_id\tllm_response\n')
        
        for node_id in node_id2prompt.keys():  
            print(node_id)
            prompt = node_id2prompt[node_id]
            
            if prompt:
            
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]

                text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(device)
                
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512
                )
                
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                response = response.replace('\n', ' ').replace('\t', '')
            else:
                response = None
            
            f.write(f'{node_id}\t{response}\n')
    '''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='FB15k' )    # FB15k; TMDB5K; MUSIC10K
   
    #parser.add_argument("--gpu", type=int, default=1)   # while qwen2 : # 
    parser.add_argument("--max_triplets", type=int, default=10)     # 10; 5; 3
    parser.add_argument("--sampling", type=str, default='cluster' )   # cluster; random
    # parser.add_argument('--max_id', type=int, default=16000)
    # parser.add_argument('--min_id', type=int, default=0)

    args = parser.parse_args()
    #torch.cuda.set_device(args.gpu)    # while qwen2 : # 
    
    if args.sampling == 'cluster':
        args.data_path = './datasets/{}/mpnet/{}_merge_cluster_t{}.tsv'.format(args.dataset, (args.dataset).lower(),str(args.max_triplets))
        args.save_path = './llm_response/{}/mpnet_llama3-1/{}_llm_res_t{}_cluster.tsv'.format(args.dataset, (args.dataset).lower(),str(args.max_triplets))
    else:
        args.data_path = './datasets/{}/mpnet/{}_merge_random_t{}.tsv'.format(args.dataset, (args.dataset).lower(),str(args.max_triplets))
        args.save_path = './llm_response/{}/mpnet_llama3-1/{}_llm_res_t{}_random.tsv'.format(args.dataset, (args.dataset).lower(),str(args.max_triplets))

    print(args)

    if not os.path.exists('./llm_response/{}/mpnet_llama3-1'.format(args.dataset)):
        os.makedirs('./llm_response/{}/mpnet_llama3-1'.format(args.dataset))
    
    
    node_id2prompt = get_prompt(args.dataset,args.data_path)
    llm_combine(node_id2prompt, args.save_path, args)

    
        


     


