from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import clip
from tqdm import tqdm
from rich import print
from sentence_transformers import SentenceTransformer
from LLM import chatLLM
from utility.utils import *
import torch


class GenDesc:
    def __init__(self, dataset="AWA2", batch_size=50, model_name="gpt4o") -> None:
        self.dataset = dataset
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset2domain = {"AWA2": "animal", "CUB": "bird", "SUN": "scene"}
        self.domain = self.dataset2domain[self.dataset]
        self.batch_size = batch_size
        self.class_set, self.view_set = self.load_class_view_set(self.dataset)
        self.view_set_plus = ['global'] + self.view_set

        if len(self.class_set) > self.batch_size:
            self.batch_num = (len(self.class_set) +
                              self.batch_size-1) // self.batch_size
            self.last_batch_size = len(self.class_set) % self.batch_size
            if self.last_batch_size == 0:
                self.last_batch_size = self.batch_size
        else:
            self.batch_num = 1
            self.last_batch_size = len(self.class_set)

    def load_prompt(self):
        self.prompt_system = '\n'.join(read_lines(
            f"prompt_view/{self.dataset}/main_prompt_sys.txt"))
        self.prompt_user_view = {}
        if self.batch_num > 1:
            for k in range(self.batch_num):
                self.prompt_user_view[f'global_{k:02d}'] = '\n'.join(
                    read_lines(f"prompt_view/{self.dataset}/main_prompt_global_{k:02d}.txt"))
            for k in range(self.batch_num):
                for view in self.view_set:
                    self.prompt_user_view[f'{view}_{k:02d}'] = '\n'.join(
                        read_lines(f"prompt_view/{self.dataset}/main_prompt_{view}_{k:02d}.txt"))
        else:
            self.prompt_user_view['global'] = '\n'.join(
                read_lines(f"prompt_view/{self.dataset}/main_prompt_global.txt"))
            for view in self.view_set:
                self.prompt_user_view[view] = '\n'.join(read_lines(
                    f"prompt_view/{self.dataset}/main_prompt_{view}.txt"))

    def load_class_view_set(self, dataset):
        class_set = read_lines(
            f"prompt_view/aux_info/classnames_{dataset}.txt")
        if dataset == "CUB":
            class_set = [classname.split('.')[1] for classname in class_set]
        class_set = [classname.replace(
            '_', ' ').replace("+", " ") for classname in class_set]
        view_set = read_lines(
            f"prompt_view/aux_info/views_{dataset}.txt")
        view_set = ['. '.join(view.split('. ')[1:]) for view in view_set]
        return class_set, view_set

    def gen_multiview_desc(self, query_view='all'):
        modelname2model = {
            "gpt4o": "gpt-4o-2024-08-06",
            "gpt4omini": "gpt-4o-mini-2024-07-18",
            "llama": "llama3.1-70b-instruct",
            "llama8b": "llama3.1-8b-instruct",
            "qwen_plus": "qwen-plus-2024-11-25"
        }
        model = modelname2model[self.model_name]
        mkdirp(f"LLM/{self.dataset}/{self.model_name}_view")
        if query_view != 'all':
            print(f"Requesting {query_view}...")
            self.gpt.chat(
                self.prompt_user_view[query_view],
                model=model,
                save_path=f"LLM/{self.dataset}/{self.model_name}_view/{query_view}.txt")
        else:
            for view_name, view_prompt in self.prompt_user_view.items():
                print(f"Requesting {view_name}...")
                self.gpt.chat(
                    view_prompt,
                    model=model,
                    save_path=f"LLM/{self.dataset}/{self.model_name}_view/{view_name}.txt")

    def init_local_LLM(self):
        modelname2model = {
            "llama-local": "meta-llama/Llama-3.1-8B-Instruct",
            "qwen-local": "Qwen/Qwen2.5-7B-Instruct"
        }
        model = modelname2model[self.model_name]
        model_path = f"/data/model/{model}"
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

    def gen_multiview_desc_local(self, query_view='all'):
        mkdirp(f"LLM/{self.dataset}/{self.model_name}_view")
        prompt_user_view = self.prompt_user_view[query_view] if query_view != 'all' else self.prompt_user_view
        for view_name, view_prompt in prompt_user_view.items():
            print(f"Requesting {view_name}...")
            text = self.tokenizer.apply_chat_template(
                view_prompt, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer(
                [text], return_tensors="pt").to(self.device)

            # Generate the output sequence
            generated_ids = self.llm.generate(
                model_inputs.input_ids, max_new_tokens=5000)

            # Extract the generated portion of the output
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids)]
            answer = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)[0]
            save_path = f"LLM/{self.dataset}/{self.model_name}_view/{view_name}.txt"
            save_lines(save_path, [answer])

    def merge_multi_batch_answer(self):
        if self.batch_num == 1:
            wrong_view = []
            for view in self.view_set_plus:
                view_desc = read_lines(
                    f"LLM/{self.dataset}/{self.model_name}_view/{view}.txt")
                view_desc = [desc for desc in view_desc if desc.strip(
                ) != "" and not desc.strip().startswith("Note:")]
                view_desc = [desc for desc in view_desc if desc.strip()[
                    0].isdigit()]
                if len(view_desc) != self.batch_size:
                    print(
                        f"Warning: {view} has {len(view_desc)} lines (LLM/{self.dataset}/{self.model_name}_view/{view}.txt)")
                    wrong_view.append(view)
                else:
                    save_lines(
                        f"LLM/{self.dataset}/{self.model_name}_view/{view}.txt", view_desc)
            return
        wrong_desc = []
        for view in self.view_set_plus:
            merge_desc = []
            for k in range(self.batch_num):
                view_desc = read_lines(
                    f"LLM/{self.dataset}/{self.model_name}_view/{view}_{k:02d}.txt")
                view_desc = [desc for desc in view_desc if desc.strip(
                ) != "" and not desc.strip().startswith("Note:")]
                view_desc = [desc for desc in view_desc if desc.strip()[
                    0].isdigit()]
                if (k != self.batch_num-1 and len(view_desc) != self.batch_size) or (k == self.batch_num-1 and len(view_desc) != self.last_batch_size):
                    print(
                        f"Warning: {view}_{k:02d} has {len(view_desc)} lines (LLM/{self.dataset}/{self.model_name}_view/{view}_{k:02d}.txt)")
                    wrong_desc.append(f"{view}_{k:02d}")
                else:
                    save_lines(
                        f"LLM/{self.dataset}/{self.model_name}_view/{view}_{k:02d}.txt", view_desc)
                merge_desc.extend(view_desc)
            save_lines(
                f"LLM/{self.dataset}/{self.model_name}_view/{view}.txt", merge_desc)

    def check_desc(self):
        not_query = []
        for view in self.view_set_plus:
            if not os.path.exists(f"LLM/{self.dataset}/{self.model_name}_view/{view}.txt"):
                not_query.append(view)
        print(not_query)

    def process_desc(self):
        self.class_view_desc = [
            ['' for _ in range(len(self.view_set_plus))] for _ in range(len(self.class_set))]
        for j, view in tqdm(enumerate(self.view_set_plus)):
            view_desc = read_lines(
                f"LLM/{self.dataset}/{self.model_name}_view/{view}.txt")
            if view != 'global':
                view_desc = [desc for desc in view_desc if desc.strip() != ""]
                view_desc = [desc for desc in view_desc if desc.strip()[
                    0].isdigit()]
                view_desc = [desc.split('. ', 1)[-1]
                             for desc in view_desc]
                view_desc = [desc.split(': ', 1)[-1]
                             for desc in view_desc]
            for i, desc in enumerate(self.class_set):
                self.class_view_desc[i][j] = view_desc[i]

    def text_embedding(self, emb_modelname):
        if emb_modelname == "clip":
            emb_model, preprocess = clip.load("RN101", device=self.device)
        elif emb_modelname == "sbert":
            emb_model = SentenceTransformer(
                f'/data/model/sentence-transformers/all-mpnet-base-v2', device=self.device)
        elif emb_modelname == "qwen":
            tokenizer = AutoTokenizer.from_pretrained(
                f"/data/model/Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            emb_model = AutoModel.from_pretrained(
                f"/data/model/Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
        elif emb_modelname == "llama":
            tokenizer = AutoTokenizer.from_pretrained(
                f"/data/model/meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            emb_model = AutoModel.from_pretrained(
                f"/data/model/meta-llama/Llama-3.1-8B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown embedding model: {emb_modelname}")

        text_features = []
        for i, class_ in enumerate(tqdm(self.class_set)):
            desc = self.class_view_desc[i]
            if emb_modelname == "clip":
                tokenized_desc = clip.tokenize(
                    desc, truncate=True).to(self.device)
                with torch.no_grad():
                    desc_embedding = emb_model.encode_text(
                        tokenized_desc).cpu().numpy()
            elif emb_modelname == "sbert":
                desc_embedding = emb_model.encode(
                    desc, show_progress_bar=False)
            else:
                model_inputs = tokenizer(desc, return_tensors="pt",
                                         padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = emb_model(model_inputs.input_ids,
                                        output_hidden_states=True, return_dict=True)
                    hidden_states = outputs.hidden_states[-1].to(
                        dtype=torch.float32)
                    desc_embedding = hidden_states.mean(
                        dim=1).squeeze().cpu().numpy()
            text_features.append(desc_embedding)
        text_features = np.array(text_features)
        if emb_modelname == "clip":
            save_path = f"embeddings/view/{self.dataset}_{self.model_name}_view_clip.npy"
        elif emb_modelname == "sbert":
            save_path = f"embeddings/view/{self.dataset}_{self.model_name}_view_sbert.npy"
        elif emb_modelname == "qwen":
            save_path = f"embeddings/view/{self.dataset}_{self.model_name}_view_qwen-7b.npy"
        elif emb_modelname == "llama":
            save_path = f"embeddings/view/{self.dataset}_{self.model_name}_view_llama-8b.npy"
        np.save(save_path, text_features)
        print(f"Embedding saved at {save_path}")

    def gen_main_prompt(self):
        all_class_set = [self.class_set[i:i+self.batch_size]
                         for i in range(0, len(self.class_set), self.batch_size)]
        numbered_class_set = [[f"{i+1}. {classname}" for i,
                               classname in enumerate(sub_class_set)] for sub_class_set in all_class_set]
        numbered_class_set_str = ['\n'.join(sub_class_set)
                                  for sub_class_set in numbered_class_set]

        main_prompt_sys = '\n'.join(read_lines(
            "prompt_view/main_prompt_sys.txt"))
        main_prompt_sys = main_prompt_sys.replace("[domain]", self.domain)

        mkdirp(f"prompt_view/{self.dataset}")
        save_lines(
            f"prompt_view/{self.dataset}/main_prompt_sys.txt", [main_prompt_sys])
        for k, sub_class_set in enumerate(tqdm(numbered_class_set)):
            main_prompt_user_global = '\n'.join(
                read_lines("prompt_view/main_prompt_user_global.txt"))
            main_prompt_user_global = main_prompt_user_global.replace(
                "[domains]", f"{self.domain}s")
            main_prompt_user_global = main_prompt_user_global.replace(
                "[numbered_class_set]", numbered_class_set_str[k])
            main_prompt_user_global = main_prompt_user_global.replace(
                "[class 1]", all_class_set[k][0])
            main_prompt_user_global = main_prompt_user_global.replace(
                "[class 2]", all_class_set[k][1])
            save_lines(f"prompt_view/{self.dataset}/main_prompt_global_{k:02d}.txt",
                       [main_prompt_user_global])

            print(
                f"Finished generating main_prompt_global for {self.dataset} and saved in prompt_view/{self.dataset}.")

            for view in tqdm(self.view_set):
                main_prompt_user = '\n'.join(
                    read_lines("prompt_view/main_prompt_user_view.txt"))
                main_prompt_user = main_prompt_user.replace(
                    "[domains]", f"{self.domain}s")
                main_prompt_user = main_prompt_user.replace(
                    "[numbered_class_set]", numbered_class_set_str[k])
                main_prompt_user = main_prompt_user.replace(
                    "[view]", view.replace(']-[', ' - '))
                main_prompt_user = main_prompt_user.replace(
                    "[class 1]", all_class_set[k][0])
                main_prompt_user = main_prompt_user.replace(
                    "[class 2]", all_class_set[k][1])
                save_lines(
                    f"prompt_view/{self.dataset}/main_prompt_{view}_{k:02d}.txt", [main_prompt_user])
        print(
            f"Finished generating main_prompt_view for {self.dataset} and saved in prompt_view/{self.dataset}.")


def load_class_view_set(dataset):
    class_set = read_lines(
        f"prompt_view/aux_info/classnames_{dataset}.txt")
    if dataset == "CUB":
        class_set = [classname.split('.')[1] for classname in class_set]
    class_set = [classname.replace(
        '_', ' ').replace("+", " ") for classname in class_set]
    view_set = read_lines(
        f"prompt_view/aux_info/views_{dataset}.txt")
    view_set = ['. '.join(view.split('. ')[1:]) for view in view_set]
    return class_set, view_set


def gen_multiview_desc(gen_desc, view):
    gen_desc.gen_multiview_desc(view)


if __name__ == "__main__":
    dataset_set = ["AWA2", "CUB", "SUN"]
    model_set = ["gpt4o", "gpt4omini", "llama", "qwen_plus"]
    embedding_model_set = ["clip", "sbert", "qwen", "llama"]
    for dataset in dataset_set:
        for modelname in model_set:
            gen_desc = GenDesc(dataset, model_name=modelname)
            gen_desc.gen_main_prompt()
            gen_desc.load_prompt()
            gen_desc.check_desc()
            gen_desc.llm = chatLLM(gen_desc.prompt_system)
            gen_desc.gen_multiview_desc()
            gen_desc.init_local_LLM()
            gen_desc.gen_multiview_desc_local()
            gen_desc.merge_multi_batch_answer()
            gen_desc.process_desc()
            for embedding_model in embedding_model_set:
                gen_desc.text_embedding(embedding_model)
