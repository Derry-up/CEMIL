import torch.nn as nn
import torch
import torch.nn.functional as F


class LINEAR(nn.Module):
    def __init__(self, input_dim, nclass, bias=True):
        super(LINEAR, self).__init__()
        self.fc = nn.Linear(input_dim, nclass, bias)

    def forward(self, x):
        o = self.fc(x)
        return o


class LINEAR_TO_COS_SIM(nn.Module):
    def __init__(self, weights):
        super(LINEAR, self).__init__()
        self.weights = weights
        self.cos = nn.functional.cosine_similarity(dim=1)

    def forward(self, x):
        out = []
        for sample in x:
            temp = []
            for weight in self.weights:
                temp.append(self.cos(weight, sample))
            out.append(torch.stack(temp))
        o = torch.stack(out)
        return o


class WEIGHT_PREDICTOR(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, num_layers=3):
        super(WEIGHT_PREDICTOR, self).__init__()
        assert num_layers in [1, 2, 3]
        self.num_layers = num_layers
        if num_layers == 4:
            self.fc1 = nn.Linear(input_dim, embed_dim)
            self.fc2 = nn.Linear(embed_dim, embed_dim)
            self.fc3 = nn.Linear(embed_dim, embed_dim)
            self.fc4 = nn.Linear(embed_dim, output_dim)
        if num_layers == 3:
            self.fc1 = nn.Linear(input_dim, embed_dim)
            self.fc2 = nn.Linear(embed_dim, embed_dim)
            self.fc3 = nn.Linear(embed_dim, output_dim)
        elif num_layers == 2:
            self.fc1 = nn.Linear(input_dim, embed_dim)
            self.fc2 = nn.Linear(embed_dim, output_dim)
        else:
            self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        if self.num_layers == 4:
            o = self.fc4(
                self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))
        elif self.num_layers == 3:
            o = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        elif self.num_layers == 2:
            o = self.fc2(self.relu(self.fc1(x)))
        else:
            o = self.fc1(x)
        return o


class AUTOENCODER(nn.Module):
    def __init__(self, opt, input_dim, embed_dim, output_dim=None, num_layers=3, vae=False, bias=True, class_emb=4069, wordemb_dim=512):
        super(AUTOENCODER, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim is None:
            self.output_dim = input_dim
        if vae:
            self.embed_dim = [2 * embed_dim, embed_dim]
        else:
            self.embed_dim = [embed_dim, embed_dim]
        if num_layers == 2:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], self.output_dim)
            )
        if num_layers == 3:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )
        if num_layers == 4:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim[0], self.embed_dim[0]),
                nn.ReLU(inplace=True) if not vae else nn.Identity(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class CONTEXTUAL_AUTOENCODER(nn.Module):
    def __init__(self, opt, input_dim, embed_dim, hidden_dim=512, att_dim=312, output_dim=None, num_layers=3, vae=False, bias=True, class_emb=None, wordemb_dim=512):
        super(CONTEXTUAL_AUTOENCODER, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.att_dim = att_dim
        self.embed_dim = embed_dim
        self.class_emb = class_emb
        self.wordemb_dim = wordemb_dim
        self.attention_dim = 2048
        if output_dim is None:
            self.output_dim = input_dim
        self.embed_dim = [embed_dim, embed_dim]
        self.encoder_merge = nn.Sequential(
            nn.Linear(self.att_dim+self.wordemb_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder_merge1 = nn.Sequential(
            nn.Linear(self.att_dim+self.attention_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder_att = nn.Sequential(
            nn.Linear(self.att_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder_mv = nn.Sequential(
            nn.Linear(self.wordemb_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim[1], 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.output_dim)
        )
        self.attention_fc = nn.Linear(self.wordemb_dim, 1)
        self.beta = nn.Parameter(torch.tensor(0.2))
        self.query_layer = nn.Linear(self.wordemb_dim, self.attention_dim)
        self.query_layer_att = nn.Linear(self.att_dim, self.attention_dim)
        self.key_layer = nn.Linear(self.wordemb_dim, self.attention_dim)
        self.value_layer = nn.Linear(self.wordemb_dim, self.attention_dim)

    def scfa_pooling(self, gpt_emb):
        gpt_emb = gpt_emb.permute(0, 2, 1)
        attn_weights = self.attention_fc(
            gpt_emb).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_gpt_emb = torch.sum(
            gpt_emb * attn_weights.unsqueeze(-1), dim=1)
        return weighted_gpt_emb

    def compute_cosed(self, q, k):
        cs = F.cosine_similarity(q, k, dim=-1)
        ed = torch.norm(q - k, p=2, dim=-1)
        cosed = cs * ed
        return cosed

    def encode(self, x):
        att_emb = x[:, :self.att_dim]
        desc_emb = x[:, self.att_dim:self.att_dim+self.wordemb_dim]
        gpt_emb = x[:, self.att_dim +
                    self.wordemb_dim:].view(x.shape[0], -1, self.wordemb_dim)
        
        if self.opt.refining_model == 'cf_attention' :
            desc_emb_expanded = desc_emb.unsqueeze(
                1).expand(-1, self.opt.view_num, -1)
            q = self.query_layer(desc_emb_expanded)
            k = self.key_layer(gpt_emb)
            v = self.value_layer(gpt_emb)
            attn_scores = self.compute_cosed(q, k)
            attn_weights = torch.softmax(
                attn_scores, dim=-1)
            context = torch.matmul(attn_weights, v)
            fused_context = context.mean(dim=1)
            x = torch.cat([att_emb, fused_context], dim=1)
            return self.encoder_merge1(x)

        elif self.opt.refining_model == 'scf_attention' :
            gpt_emb = gpt_emb.permute(0, 2, 1)
            gpt_emb = self.scfa_pooling(gpt_emb)
            x = torch.cat([att_emb, self.beta*desc_emb +
                          (1-self.beta)*gpt_emb], dim=1)
            return self.encoder_merge(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)



class DUAL_AUTOENCODER(nn.Module):
    def __init__(self, opt, autoencoder1, autoencoder2):
        super(DUAL_AUTOENCODER, self).__init__()
        self.ae1 = autoencoder1
        self.ae2 = autoencoder2

    def encode1(self, x):
        return self.ae1.encode(x)

    def encode2(self, x):
        return self.ae2.encode(x)

    def decode1(self, x):
        return self.ae1.decode(x)

    def decode2(self, x):
        return self.ae2.decode(x)

    def forward(self, x):
        att_in, weight_in = x

        latent_att = self.encode1(att_in)
        latent_weight = self.encode2(weight_in)

        att_from_att = self.decode1(latent_att)
        att_from_weight = self.decode1(latent_weight)

        weight_from_weight = self.decode2(latent_weight)
        weight_from_att = self.decode2(latent_att)

        return att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight

    def predict(self, x):
        latent_att = self.encode1(x)
        return self.decode2(latent_att)
