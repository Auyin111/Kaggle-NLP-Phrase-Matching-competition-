import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig

# ====================================================
# Original Model
# ====================================================

class CustomModel_Original(nn.Module):

    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.pretrained_model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.pretrained_model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.dropout_prop)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        # TODO edit the model
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

# ====================================================
# Multi Sample Dropout (MSD)
# ====================================================

class Dropout_Linear(nn.Module):
    def __init__(self, in_size, out_size, cfg, config, prob=None):
        self.config = config
        super().__init__()
        self.cfg = cfg
        if prob == None:
            self.dropout_prob = self.cfg.dropout_prop
        else:
            self.dropout_prob = prob
        self.dropout_list = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(self.cfg.dropout_sample_num)])
        self.linear_list = nn.ModuleList(nn.Linear(in_size, out_size) for _ in range(self.cfg.dropout_sample_num))
        self._init_linear_weights(self.linear_list)

    def _init_linear_weights(self, module_list):
        for linear in module_list:
            linear.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if linear.bias is not None:
                linear.bias.data.zero_()

    def forward(self, inputs):
        outputs = None
        for i, dropout in enumerate(self.dropout_list):
            linear = self.linear_list[i]
            if i == 0:
                outputs = linear(dropout(inputs))
            else:
                temp_outputs = linear(dropout(inputs))
                outputs += temp_outputs
        if self.cfg.msd_average:
            outputs = outputs / self.cfg.dropout_sample_num
        return outputs

# ====================================================
# Multi-head Self-attention
# ====================================================

class MHSA(nn.Module):
    def __init__(self, embedding_dim, cfg, config):
        super().__init__()
        self.config = config
        self.cfg = cfg
        self.Q_list = nn.ModuleList([Dropout_Linear(embedding_dim, embedding_dim, self.cfg, self.config, prob=self.cfg.self_attention_dropout_prob) for _ in range(self.cfg.self_attention_head_num)])
        self.K_list = nn.ModuleList([Dropout_Linear(embedding_dim, embedding_dim, self.cfg, self.config, prob=self.cfg.self_attention_dropout_prob) for _ in range(self.cfg.self_attention_head_num)])
        self.V_list = nn.ModuleList([Dropout_Linear(embedding_dim, embedding_dim, self.cfg, self.config, prob=self.cfg.self_attention_dropout_prob) for _ in range(self.cfg.self_attention_head_num)])

        self.W = nn.Linear(self.cfg.self_attention_head_num, 1)
        self.W.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.W.bias.data.zero_()

        self.softmax = nn.Softmax(dim=1)
        self.LayerNorm = nn.LayerNorm(embedding_dim)

        self.mixing_residual = nn.Linear(2, 1)
        self.mixing_residual.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.mixing_residual.bias.data.zero_()

    def _init_linear_weights(self, module_list):
        for linear in module_list:
            linear.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if linear.bias is not None:
                linear.bias.data.zero_()

    def forward(self, inputs):
        for h in range(self.cfg.self_attention_head_num):
            Q_CLS = torch.swapaxes(torch.unsqueeze(self.Q_list[h](inputs[:, 0, :]), dim=1), 1, 2)
            K = self.K_list[h](inputs)
            V = self.V_list[h](inputs)
            weights = torch.bmm(K, Q_CLS) / inputs.shape[2] ** 0.5
            weights_softmax = self.softmax(weights)
            if h == 0:
                CLS_embeddings = torch.unsqueeze(torch.sum(weights_softmax * V, dim=1), dim=-1)
            else:
                CLS_embedding = torch.unsqueeze(torch.sum(weights_softmax * V, dim=1), dim=-1)
                CLS_embeddings = torch.cat((CLS_embeddings, CLS_embedding), dim=-1)
        if h > 0:
            CLS_embeddings_merged = torch.squeeze(self.W(CLS_embeddings), dim=-1)
        else:
            CLS_embeddings_merged = CLS_embeddings
        pre_residual = torch.cat((CLS_embeddings_merged, torch.unsqueeze(inputs[:, 0, :], dim=-1)), dim=-1)
        residual_output = self.mixing_residual(pre_residual)
        residual_output =  self.LayerNorm(torch.squeeze(residual_output, dim=-1))
        return residual_output

# ====================================================
# Model with MSD + Mixed Hidden Layers + MH_self_Attention
# ====================================================

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.pretrained_model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.pretrained_model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        self.mixing_layers = nn.Linear(self.config.num_hidden_layers + 1, 1)
        self._init_weights(self.mixing_layers)

        self.MHSA = MHSA(self.config.hidden_size, self.cfg, self.config)
        self.final_forward = nn.Sequential(Dropout_Linear(self.config.hidden_size, 512, self.cfg, self.config),
                                           nn.Tanh(),
                                           Dropout_Linear(512, self.cfg.target_size, self.cfg, self.config))

        self.sigmoid = nn.Sigmoid()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def mixing_hidden_layers(self, inputs):
        model_list1 = ["albert-base-v2", "xlm-roberta-base"]  # Models with dim(output) == 4
        model_list2 = ["microsoft/deberta-v3-base", "microsoft/mdeberta-v3-base"]  # Models with dim(output) == 3
        #print(self.model(**inputs))
        if self.cfg.pretrained_model in model_list1:
            all_layers = [layer for layer in self.model(**inputs)[2]]
        elif self.cfg.pretrained_model in model_list2:
            all_layers = [layer for layer in self.model(**inputs)[1]]
        else:
            raise Exception("New model is being used, please confirm the model output length in 'mixing_hidden_layers'.")
        all_layers = torch.stack(all_layers, dim=-1)
        mixed_layer_embeddings = torch.squeeze(self.mixing_layers(all_layers), dim=-1)
        return mixed_layer_embeddings

    def forward(self, inputs):
        mixed_layer_embeddings = self.mixing_hidden_layers(inputs)
        final_feature = self.MHSA(mixed_layer_embeddings)
        output = self.final_forward(final_feature)
        if self.cfg.loss_fn == "MSE" or self.cfg.loss_fn == "CCC1" or self.cfg.loss_fn == "CCC2" or self.cfg.loss_fn == "PCC":
            return self.sigmoid(output)
        return output

