import torch
import torch.nn as nn

from .step2_ccVAE_utils import one_hot_encoder


class CondLayers(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cond: int,
        bias: bool,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)
            out = self.expr_L(expr) + self.cond_L(cond)
        return out

class Encoder(nn.Module):
    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 use_ccPairs: str,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: int = None):
        super().__init__()
        self.n_classes = num_classes if num_classes is not None else 0
        self.FC = None
        self.use_ccPairs = use_ccPairs
        self.encoder_architecture = "ccPairs" if self.use_ccPairs == "ccPairs" else "rest"

        if len(layer_sizes) > 1:
            print("Encoder Sub-Architecture of", self.encoder_architecture, "cells:")
            self.FC = nn.Sequential()

            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    print("\tInput Layer in, out and cond:", in_size, out_size, self.n_classes)
                    self.FC.add_module(name="L{:d}".format(i), module=CondLayers(in_size, out_size, self.n_classes, bias=True))
                else:
                    print("\tHidden Layer", i, "in/out:", in_size, out_size)
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))

                normalization_module = nn.BatchNorm1d(out_size, affine=True) if use_bn else nn.LayerNorm(out_size, elementwise_affine=False)
                self.FC.add_module("N{:d}".format(i), module=normalization_module)
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if use_dr:
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))

        print("\tMean/Var Layer in/out:", layer_sizes[-1], latent_dim)
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x_ccPairs=None, x_rest=None, batch_ccPairs=None, batch_rest=None):
        if self.use_ccPairs == "ccPairs":
            if batch_ccPairs is not None:
                batch_ccPairs = one_hot_encoder(batch_ccPairs, n_cls=self.n_classes)
                x_ccPairs = torch.cat((x_ccPairs, batch_ccPairs), dim=-1)
        
            if self.FC is not None:
                x_ccPairs = self.FC(x_ccPairs)

            means_ccPairs = self.mean_encoder(x_ccPairs)
            log_vars_ccPairs = self.log_var_encoder(x_ccPairs)
            return means_ccPairs, log_vars_ccPairs
        
        else:
            if batch_rest is not None:
                batch_rest = one_hot_encoder(batch_rest, n_cls=self.n_classes)
                x_rest = torch.cat((x_rest, batch_rest), dim=-1)

            if self.FC is not None:
                x_rest = self.FC(x_rest)

            means_rest = self.mean_encoder(x_rest)
            log_vars_rest = self.log_var_encoder(x_rest)
            return  means_rest, log_vars_rest



class Decoder(nn.Module):
    def __init__(self,
                 layer_sizes: list,
                 latent_dim: int,
                 use_ccPairs: str,
                 recon_loss: str,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: int = None):
        super().__init__()
        self.use_ccPairs = use_ccPairs
        self.use_dr = use_dr
        self.recon_loss = recon_loss
        self.n_classes = num_classes if num_classes is not None else 0
        layer_sizes = [latent_dim] + layer_sizes
        self.decoder_architecture = "ccPairs" if self.use_ccPairs == "ccPairs" else "rest"
        
        print("\nDecoder Sub-Architecture of", self.decoder_architecture, "cells:")
        # Create first Decoder layer
        self.FirstL = nn.Sequential()
        print("\tFirst Layer in, out and cond: ", layer_sizes[0], layer_sizes[1], self.n_classes)
        self.FirstL.add_module(name="L0", module=CondLayers(layer_sizes[0], layer_sizes[1], self.n_classes, bias=False))
        if use_bn:
            self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True))
        elif use_ln:
            self.FirstL.add_module("N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False))
        self.FirstL.add_module(name="A0", module=nn.ReLU())
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))

        # Create all Decoder hidden layers
        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i+3 < len(layer_sizes):
                    print("\tHidden Layer", i+1, "in/out:", in_size, out_size)
                    self.HiddenL.add_module(name="L{:d}".format(i+1), module=nn.Linear(in_size, out_size, bias=False))
                    if use_bn:
                        self.HiddenL.add_module("N{:d}".format(i+1), module=nn.BatchNorm1d(out_size, affine=True))
                    elif use_ln:
                        self.HiddenL.add_module("N{:d}".format(i + 1), module=nn.LayerNorm(out_size, elementwise_affine=False))
                    self.HiddenL.add_module(name="A{:d}".format(i+1), module=nn.ReLU())
                    if self.use_dr:
                        self.HiddenL.add_module(name="D{:d}".format(i+1), module=nn.Dropout(p=dr_rate))
        else:
            self.HiddenL = None

        # Create Output Layers
        print("\tOutput Layer in/out: ", layer_sizes[-2], layer_sizes[-1], "\n")
        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.ReLU())
        if self.recon_loss == "zinb":
            # mean gamma
            self.mean_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1))
            # dropout
            self.dropout_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        if self.recon_loss == "nb":
            # mean gamma
            self.mean_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1))

    def forward(self, z, batch_ccPairs=None, batch_rest=None, use_decoder=None):
        # Add Condition Labels to Decoder Input
        if use_decoder == 'ccPairs':
            # Add Condition Labels to Decoder Input
            if batch_ccPairs is not None:
                batch_ccPairs = one_hot_encoder(batch_ccPairs, n_cls=self.n_classes)
                z_cat = torch.cat((z, batch_ccPairs), dim=-1)
                dec_latent = self.FirstL(z_cat)
            else:
                dec_latent = self.FirstL(z)

            # Compute Hidden Output
            if self.HiddenL is not None:
                x = self.HiddenL(dec_latent)
            else:
                x = dec_latent

            # Compute Decoder Output
            if self.recon_loss == "mse":
                recon_x = self.recon_decoder(x)
                return recon_x, dec_latent
            elif self.recon_loss == "zinb":
                dec_mean_gamma = self.mean_decoder(x)
                dec_dropout = self.dropout_decoder(x)
                return dec_mean_gamma, dec_dropout, dec_latent
            elif self.recon_loss == "nb":
                dec_mean_gamma = self.mean_decoder(x)
                return dec_mean_gamma, dec_latent
        elif use_decoder == 'rest':
            # Add Condition Labels to Decoder Input
            if batch_rest is not None:
                batch_rest = one_hot_encoder(batch_rest, n_cls=self.n_classes)
                z_cat = torch.cat((z, batch_rest), dim=-1)
                dec_latent = self.FirstL(z_cat)
            else:
                dec_latent = self.FirstL(z)

            # Compute Hidden Output
            if self.HiddenL is not None:
                x = self.HiddenL(dec_latent)
            else:
                x = dec_latent

            # Compute Decoder Output
            if self.recon_loss == "mse":
                recon_x = self.recon_decoder(x)
                return recon_x, dec_latent
            elif self.recon_loss == "zinb":
                dec_mean_gamma = self.mean_decoder(x)
                dec_dropout = self.dropout_decoder(x)
                return dec_mean_gamma, dec_dropout, dec_latent
            elif self.recon_loss == "nb":
                dec_mean_gamma = self.mean_decoder(x)
                return dec_mean_gamma, dec_latent
        else:
            if batch_ccPairs is not None and batch_rest is not None:
                batch_ccPairs = one_hot_encoder(batch_ccPairs, n_cls=self.n_classes)
                batch_rest = one_hot_encoder(batch_rest, n_cls=self.n_classes)
                batch_concat = torch.concat((batch_ccPairs,batch_rest),dim=0)
                z_cat = torch.cat((z, batch_concat), dim=-1)
                dec_latent = self.FirstL(z_cat)
            else:
                dec_latent = self.FirstL(z)

            # Compute Hidden Output
            if self.HiddenL is not None:
                x = self.HiddenL(dec_latent)
            else:
                x = dec_latent

            # Compute Decoder Output
            if self.recon_loss == "mse":
                recon_x = self.recon_decoder(x)
                return recon_x, dec_latent
            elif self.recon_loss == "zinb":
                dec_mean_gamma = self.mean_decoder(x)
                dec_dropout = self.dropout_decoder(x)
                return dec_mean_gamma, dec_dropout, dec_latent
            elif self.recon_loss == "nb":
                dec_mean_gamma = self.mean_decoder(x)
                return dec_mean_gamma, dec_latent

