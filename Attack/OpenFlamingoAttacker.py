import torch
from einops import repeat
from Attack.utils import normalize, denormalize, normalize_noise

class OpenFlamingoAttacker():
    def __init__(self, model, tokenizer, config = dict(), device = 'cuda:0', verbose = False):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" 
        self.device = device
        self.verbose = verbose 

        self.H = 224
        self.W = 224

        # visual attacker config
        self.epsilon = config["epsilon"] if "epsilon" in config else 2./255.
        self.num_iter_visual = config["num_iter_visual"] if "num_iter_visual" in config else 1000
        self.attack_lr = config["attack_lr"] if "attack_lr" in config else min(self.epsilon / 3, 2./255.)
        self.random_init = config["random_init"] if "random_init" in config else True
        self.targeted_threshold = 0.4
        self.early_stop = True

        # textual attacker config
        self.num_iter_textual = config["num_iter_textual"] if "num_iter_textual" in config else 20
        self.trigger_token_length = config["trigger_token_length"] if "trigger_token_length" in config else 3
        self.num_candidates = config["num_candidates"] if "num_candidates" in config else 100
        self._get_vocabulary()

        self.model.eval()
        self.model.requires_grad_(False)
        self.mask_ids = -100
        self.media_token_ids = self.tokenizer("<image>").input_ids[-1]

    def _random_noise(self, x):
        return torch.zeros(x.shape, dtype=x.dtype, device=x.device).uniform_(-self.epsilon, self.epsilon)
    
    def generate(self, vision_x, prompt, max_new_tokens=30):
        if len(vision_x.shape) == 4:
            vision_x = repeat(vision_x, 'N c h w -> b N T c h w', b=1, T=1)
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_text = self.model.generate(
            vision_x = vision_x,
            lang_x = tokenized_prompt["input_ids"],
            attention_mask = tokenized_prompt["attention_mask"],
            max_new_tokens = max_new_tokens,
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
        single_token_id_to_vocab = dict()
        single_token_vocab_to_id = dict()

        cnt = 0

        for item in vocabs:
            tokens = self.tokenizer(item, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if tokens.shape[1] == 1:

                single_token_vocabs.append(item)
                emb = self.model.lang_encoder.model.embed_tokens(tokens)
                single_token_vocabs_embedding.append(emb)

                single_token_id_to_vocab[cnt] = item
                single_token_vocab_to_id[item] = cnt

                cnt+=1

        single_token_vocabs_embedding = torch.cat(single_token_vocabs_embedding, dim=1).squeeze()

        self.vocabs = single_token_vocabs
        self.embedding_matrix = single_token_vocabs_embedding.to(self.device)
        self.id_to_vocab = single_token_id_to_vocab
        self.vocab_to_id = single_token_vocab_to_id

    def hotflip_attack(self, grad, token, increase_loss=False):
        token_id = self.vocab_to_id[token]
        token_emb = self.embedding_matrix[token_id]

        scores = ((self.embedding_matrix - token_emb) @ grad.T).squeeze(1)

        if not increase_loss:
            scores *= -1

        _, best_k_ids = torch.topk(scores, self.num_candidates)
        return best_k_ids.detach().cpu().numpy()
    
    
    # perturb the last image (query) of input images
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

    # TODO: insert a few tokens in different positions and perturb
    # TODO: perturb a specific number of tokens
    # return the adversarial prompt
    # only support prompt batch_size = 1
    # images: [N, c, h, w]
    # prompt and label should all be text
    def textual_attack(self, images, prompt, label, targeted=False):
        vision_x = repeat(images, 'N c h w -> b N T c h w', b=1, T=1)
        input_ids, attention_mask, label_ids = self.get_prompt_input(prompt, label)

        for _ in range(self.num_iter_textual + 1):
            for token_to_flip_index in range(input_ids.shape[1]):

                if token_to_flip_index == self.media_token_ids:
                    continue

                input_ids, attention_mask, label_ids = self.get_prompt_input(prompt, label)

                inputs_embeds = self.model.lang_encoder.model.embed_tokens(input_ids)
                inputs_embeds.requires_grad_(True)

                loss = self.get_attack_loss(vision_x, input_ids, attention_mask, input_ids)
                loss.backward()

                tokens_grad = inputs_embeds.grad[:, token_to_flip_index, :]
                candidates = self.hotflip_attack(tokens_grad, input_ids[token_to_flip_index], increase_loss=not targeted)

                inputs_embeds.grad.zero_()
                # self.model.zero_grad()

                with torch.no_grad():
                    curr_best_loss = 999999
                    curr_best_trigger_tokens = None
                    curr_best_trigger = None

                    for token_ids in candidates:
                        input_ids[0][token_to_flip_index] = token_ids

                        flipped_loss = self.get_attack_loss(vision_x, input_ids, attention_mask, label_ids)

                        if flipped_loss < curr_best_loss:
                            curr_best_loss = flipped_loss
                            # curr_best_trigger_tokens = next_adv_prompt_tokens
                            # curr_best_trigger = next_adv_prompt

                    # Update overall best if the best current candidate is better
                    if curr_best_loss < loss:
                        adv_prompt_tokens = curr_best_trigger_tokens
                        adv_prompt = curr_best_trigger

