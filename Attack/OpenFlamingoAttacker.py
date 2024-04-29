import torch
from copy import deepcopy
from einops import repeat
from Utils.utils import normalize, denormalize, normalize_noise, filter_special_characters

class OpenFlamingoAttacker():
    def __init__(self, model, tokenizer, config = dict(), device = 'cuda:0', verbose = False):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" 
        self.device = device
        self.verbose = verbose 

        self.H = 224
        self.W = 224

        # TODO: implement stop criteria
        self.max_new_tokens = 40

        # visual attacker config
        self.epsilon = config["epsilon"] if "epsilon" in config else 2./255.
        self.num_iter_visual = config["num_iter_visual"] if "num_iter_visual" in config else 500
        self.attack_lr = config["attack_lr"] if "attack_lr" in config else min(self.epsilon / 3, 2./255.)
        self.random_init = config["random_init"] if "random_init" in config else True
        self.targeted_threshold = 0.4
        self.early_stop = True

        # textual attacker config
        self.num_iter_textual = config["num_iter_textual"] if "num_iter_textual" in config else 10
        self.num_token_to_flip = config["num_token_to_flip"] if "num_token_to_flip" in config else 5
        self.num_token_to_insert = config["num_token_to_insert"] if "num_token_to_insert" in config else 0
        self.num_grad_candidates = config["num_grad_candidates"] if "num_grad_candidates" in config else 50
        self.num_beams = config["num_beams"] if "num_beams" in config else 20
        self._get_vocabulary()

        self.model.eval()
        self.model.requires_grad_(False)
        self.mask_ids = -100
        self.media_token_ids = self.tokenizer("<image>").input_ids[-1]

        # split prompt into sys_prompt and user_prompt through 'Question' and 'Answer' ids
        # TODO: implement prompt wrapper class
        self.question_token_ids = self.tokenizer("Question").input_ids[-1]
        self.answer_token_ids = self.tokenizer("Answer").input_ids[-1]
        self.unk_token_ids = self.tokenizer("<unk>").input_ids[-1]
        self.ignore_token_ids = [self.media_token_ids, self.question_token_ids, self.answer_token_ids]

    def _random_noise(self, x):
        return torch.zeros(x.shape, dtype=x.dtype, device=x.device).uniform_(-self.epsilon, self.epsilon)
    
    def generate(self, vision_x, prompt):
        if len(vision_x.shape) == 4:
            vision_x = repeat(vision_x, 'N c h w -> b N T c h w', b=1, T=1)
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_text = self.model.generate(
            vision_x = vision_x,
            lang_x = tokenized_prompt["input_ids"],
            attention_mask = tokenized_prompt["attention_mask"],
            max_new_tokens = self.max_new_tokens,
        )
        response = self.tokenizer.decode(generated_text[0])
        return response

    
    def get_prompt_input(self, prompt, label):
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids_length = tokenized_prompt["input_ids"].shape[1]
        label_mask = torch.full(
            size=[1, prompt_ids_length], 
            fill_value=self.mask_ids, 
            dtype=tokenized_prompt["input_ids"].dtype, 
            device=self.device
        )

        # get rid of the default <unk> in target tokenization
        target_ids = self.tokenizer(label, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

        label_ids = torch.cat([label_mask, target_ids], dim = 1).to(self.device)
        input_ids = torch.cat([tokenized_prompt["input_ids"], target_ids], dim = 1).to(self.device)
        attention_mask = torch.ones(input_ids.shape, dtype=tokenized_prompt["attention_mask"].dtype).to(self.device)

        return input_ids, attention_mask, label_ids
    
    def get_attack_loss(self, vision_x, input_ids, attention_mask, label_ids, inputs_embeds=None):
        if len(vision_x.shape) == 4:
            vision_x = repeat(vision_x, 'N c h w -> b N T c h w', b=1, T=1)

        loss = None
        if inputs_embeds != None:
            loss = self.model.forward(
                vision_x = vision_x,
                lang_x = None,
                inputs_embeds = inputs_embeds,
                media_locations = input_ids == self.media_token_ids,
                attention_mask = attention_mask,
                labels = label_ids,
            ).loss
        else:
            loss = self.model.forward(
                vision_x = vision_x,
                lang_x = input_ids,
                attention_mask = attention_mask,
                labels = label_ids,
            ).loss
        return loss
            
    # get single-token volcabulary and embedings
    # hotflip on a token-level
    def _get_vocabulary(self):
        vocab_dicts = self.tokenizer.get_vocab()
        vocabs = vocab_dicts.keys()

        single_token_vocabs = []
        single_token_vocabs_embedding = []
        # single_token_id_to_vocab = dict()
        # single_token_vocab_to_id = dict()

        cnt = 0

        for item in vocabs:
            tokens = self.tokenizer(item, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if tokens.shape[1] == 1 and filter_special_characters(item):

                single_token_vocabs.append(item)
                emb = self.model.lang_encoder.model.embed_tokens(tokens)
                single_token_vocabs_embedding.append(emb)

                # single_token_id_to_vocab[cnt] = item
                # single_token_vocab_to_id[item] = cnt

                cnt += 1

        single_token_vocabs_embedding = torch.cat(single_token_vocabs_embedding, dim=1).squeeze()

        self.vocabs = single_token_vocabs
        self.embedding_matrix = single_token_vocabs_embedding.to(self.device)
        # self.id_to_vocab = single_token_id_to_vocab
        # self.vocab_to_id = single_token_vocab_to_id

    # get 'Question' and 'Answer' ids
    def get_qa_ids(self, input_ids):
        equal_indices = torch.eq(input_ids, self.question_token_ids).squeeze()
        q_index = torch.nonzero(equal_indices, as_tuple=False)[-1].item()

        equal_indices = torch.eq(input_ids, self.answer_token_ids).squeeze()
        a_index = torch.nonzero(equal_indices, as_tuple=False)[-1].item()

        return q_index, a_index

    def insert_token(self, tensor, offset, value):
        assert tensor.shape[0] == 1

        n = self.num_token_to_insert
        expanded_tensor = torch.full([1, tensor.shape[1] + n], value, dtype=tensor.dtype, device=tensor.device)
        expanded_tensor[0][:offset] = tensor[0][:offset]
        expanded_tensor[0][offset + n:] = tensor[0][offset:]
        
        return expanded_tensor

    def flip(self, input_ids, flip_map):
        adv_input_ids = input_ids.clone().to(self.device)
        for index in flip_map:
            adv_input_ids[0][index] = flip_map[index]
        return adv_input_ids

    def generate_flip_map(self, raw_flip_map, index, candidates):
        new_flip_map_list = [raw_flip_map]
        for cand in candidates:
            new_flip_map = deepcopy(raw_flip_map)
            new_flip_map[index] = cand
            new_flip_map_list.append(new_flip_map)

        return new_flip_map_list

    def hotflip_attack(self, grad, token_ids, increase_loss=False):
        token_emb = self.embedding_matrix[token_ids]
        scores = ((self.embedding_matrix - token_emb) @ grad.T).squeeze(1)

        if not increase_loss:
            scores *= -1

        _, best_k_ids = torch.topk(scores, self.num_candidates)
        return best_k_ids.detach().cpu().numpy()
    
    
    # perturb the last image (query) of input images
    # 
    # patch: perturb only a patch that is fixed in the top-left corner.
    # images: [N, c, h, w]
    # prompt and label should all be text   
    # return the adversarial noise
    # only support prompt batch_size = 1
    def visual_attack(self, images, prompt, label, targeted=False, patch=None):

        input_ids, attention_mask, label_ids = self.get_prompt_input(prompt, label)

        vision_x_raw = denormalize(images).clone().to(self.device)
        vision_x_noise = torch.zeros(vision_x_raw.shape, dtype=vision_x_raw.dtype, device=self.device)

        if self.random_init:
            vision_x_noise = vision_x_noise + self._random_noise(vision_x_noise)

        patch_h, patch_w = self.H, self.W
        if patch is not None:
            patch_h, patch_w = patch.shape[1], patch.shape[2]
            vision_x_noise[-1][:, :patch_h, :patch_w] += patch

        vision_x_adv_noise = torch.clamp(vision_x_noise + vision_x_raw, 0, 1) - vision_x_raw

        for _ in range(self.num_iter_visual + 1):
            vision_x_adv_noise.requires_grad_(True)
            vision_x_adv = normalize(vision_x_raw + vision_x_adv_noise)
            loss = self.get_attack_loss(vision_x_adv, input_ids, attention_mask, label_ids)

            if targeted and self.early_stop and loss.item() < self.targeted_threshold:
                print(f'iter-{_}  loss: {loss.item()}')
                break
            
            loss.backward()

            # use .data in order to change the value while remaining the grad
            grad_sign = vision_x_adv_noise.grad[-1][:, :patch_h, :patch_w].sign()
            if targeted:
                vision_x_adv_noise.data[-1][:, :patch_h, :patch_w] = (vision_x_adv_noise.data[-1][:, :patch_h, :patch_w] - self.attack_lr * grad_sign).clamp(-self.epsilon, self.epsilon)
            else:
                vision_x_adv_noise.data[-1][:, :patch_h, :patch_w] = (vision_x_adv_noise.data[-1][:, :patch_h, :patch_w] + self.attack_lr * grad_sign).clamp(-self.epsilon, self.epsilon)
            vision_x_adv_noise.data[-1][:, :patch_h, :patch_w] = (vision_x_adv_noise.data[-1][:, :patch_h, :patch_w] + vision_x_raw.data[-1][:, :patch_h, :patch_w]).clamp(0, 1) - vision_x_raw.data[-1][:, :patch_h, :patch_w]
            vision_x_adv_noise.data[:-1] = 0
            vision_x_adv_noise.data[-1][:, patch_h:, patch_w:] = 0

            # it will significantly accelerate the process if we accumulate the grad
            # vision_x_adv_noise.grad.zero_()
            # self.model.zero_grad()

            # check vision_x_adv
            if self.verbose and _ % 100 == 0:
                vision_x = normalize(vision_x_raw + vision_x_adv_noise)
                response = self.generate(vision_x, prompt)
                print(f'iter-{_}  loss: {loss.item()}')
                print(f'response: {response}')

        return vision_x_adv_noise.detach()

    # insert a few tokens in different positions and perturb through beam-search
    # it's better to insert less than 5 tokens because beam-search is extremely time-consuming
    # TODO: it's hard to flip a few tokens in the question while remaining the exact same meaning

    # images: [N, c, h, w]
    # prompt and label should all be text
    # return the adversarial prompt
    # only support prompt batch_size = 1
    def textual_attack(self, images, prompt, label, targeted=False, insert='end', return_dict=False):
        
        vision_x = repeat(images, 'N c h w -> b N T c h w', b=1, T=1)
        input_ids, attention_mask, label_ids = self.get_prompt_input(prompt, label)

        input_ids, attention_mask, label_ids = self.get_prompt_input(prompt, label)
        q_index, a_index = self.get_qa_ids(input_ids)
        offset = q_index
        if insert is not None:
            if insert == 'begin':
                offset = 0
            elif insert == 'mid':
                offset = q_index
            elif insert == 'end':
                offset = a_index
            else:
                raise ValueError("Insert position should be begin, mid, end or None")

            input_ids = self.insert_token(input_ids, offset, self.unk_token_ids)
            label_ids = self.insert_token(label_ids, offset, self.mask_ids)
            attention_mask = self.insert_token(attention_mask, offset, 1)

        else:
            raise ValueError("It's hard to flip a few tokens while remaining the exact same question")

        prompt_token_length = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1] + self.num_token_to_insert
        search_length = self.num_token_to_insert if self.num_token_to_insert != 0 else prompt_token_length - offset

        # ========================================= beam search =========================================
        # beam: (flip_map, loss)
        beams = []
        beam = (dict(), self.get_attack_loss(vision_x, input_ids, attention_mask, label_ids))
        for token_to_flip_index in range(offset, offset + search_length):
            beam[0][token_to_flip_index] = input_ids[0][token_to_flip_index]
        beams.append(beam)
        
        if self.verbose:
                print(f'iter: 0  loss: {beams[0][1]:.6f}')
            
        for _ in range(self.num_iter_textual):
        
            for token_to_flip_index in range(offset, offset + search_length):
        
                new_beams = []
                for beam in beams:
                
                    adv_input_ids = self.flip(input_ids, beam[0])
                    # cur_ids = adv_input_ids[0][token_to_flip_index].item()
                    
                    inputs_embeds = self.model.lang_encoder.model.embed_tokens(adv_input_ids)
                    inputs_embeds.requires_grad_(True)
                
                    loss = self.get_attack_loss(vision_x, adv_input_ids, attention_mask, label_ids, inputs_embeds=inputs_embeds)
                    loss.backward()
                
                    tokens_grad = inputs_embeds.grad[:, token_to_flip_index, :]
                    candidates = self.hotflip_attack(tokens_grad, adv_input_ids[0][token_to_flip_index], increase_loss=not targeted)
                
                    new_flip_map_list = self.generate_flip_map(beam[0], token_to_flip_index, candidates)
                    for flip_map in new_flip_map_list:
                        adv_input_ids = self.flip(input_ids, flip_map)
                        loss = self.get_attack_loss(vision_x, adv_input_ids, attention_mask, label_ids).item()
                        new_beams.append((flip_map, loss))
                
                beams = sorted(new_beams, key=lambda x: x[1], reverse=not targeted)[:self.num_beams]

            if self.verbose:
                print(f'iter: {_ + 1}  loss: {beams[0][1]:.6f}')
        
        # =======================================================================================                     
                
        adv_input_ids = self.flip(input_ids, beams[0][0])
        adv_prompt = self.tokenizer.decode(adv_input_ids[0][1:prompt_token_length])

        print('Hotflip completed!')
        if self.verbose:
            response = self.generate(vision_x, adv_prompt)
            print(f'response: {response}')

        if return_dict:
            return adv_prompt, beams[0][0]
        else:
            return adv_prompt

