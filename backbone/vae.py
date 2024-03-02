# Implementation of VAE is taken & slightly modified from:
# https://github.com/GMvandeVen/class-incremental-learning

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from utils.model_utils import get_activation_from_name


def get_vae(dataset_config, args):
    if args.depth > 0:
        vae = AutoEncoder(
            args,
            dataset_config["size"],
            dataset_config["channels"],
            # -conv-layers
            conv_type=args.conv_type,
            depth=args.depth,
            start_channels=args.channels,
            reducing_layers=args.rl,
            conv_bn=(args.conv_bn == "yes"),
            conv_in=(args.conv_in == "yes"),
            conv_ln=(args.conv_ln == "yes"),
            conv_nl=args.conv_nl,
            num_blocks=args.n_blocks,
            global_pooling=args.gp,
            no_fnl=True if args.conv_type == "standard" else False,
            convE=None,
            conv_gated=False,
            # -fc-layers
            fc_layers=args.fc_lay,
            fc_units=args.fc_units,
            h_dim=args.fc_units,
            fc_drop=args.fc_drop,
            fc_bn=(args.fc_bn == "yes"),
            fc_ln=(args.fc_ln == "yes"),
            fc_nl=args.fc_nl,
            excit_buffer=True,
            fc_gated=False,
            # -prior
            z_dim=args.z_dim,
            prior=args.prior,
            n_modes=args.n_modes,
            # -decoder
            recon_loss=args.recon_loss,
            network_output="sigmoid" if args.dataset == "seq-mnist" else "none",
            deconv_type=args.deconv_type
            if hasattr(args, "deconv_type")
            else "standard",
        )
    else:
        vae = AutoEncoder(
            args,
            dataset_config["size"],
            dataset_config["channels"],
            # -conv-layers
            conv_type="standard",
            depth=0,
            start_channels=64,
            reducing_layers=3,
            conv_bn=True,
            conv_in=False,
            conv_ln=False,
            conv_nl="relu",
            num_blocks=2,
            global_pooling=False,
            no_fnl=True,
            convE=None,
            conv_gated=False,
            # -fc-layers
            fc_layers=args.fc_lay,
            fc_units=args.fc_units,
            h_dim=args.fc_units,
            fc_drop=args.fc_drop,
            fc_bn=(args.fc_bn == "yes"),
            fc_ln=(args.fc_ln == "yes"),
            fc_nl=args.fc_nl,
            excit_buffer=True,
            fc_gated=False,
            # -prior
            z_dim=args.z_dim,
            prior=args.prior,
            n_modes=args.n_modes,
            # -decoder
            recon_loss=args.recon_loss,
            network_output="sigmoid" if args.dataset == "seq-mnist" else "none",
            deconv_type=args.deconv_type
            if hasattr(args, "deconv_type")
            else "standard",
        )

    return vae


class AutoEncoder(nn.Module):
    """Class for variational auto-encoder (VAE) models."""

    def __init__(
        self,
        args,
        image_size,
        image_channels,
        # -conv-layers
        conv_type="standard",
        depth=0,
        start_channels=64,
        reducing_layers=3,
        conv_bn=True,
        conv_in=False,
        conv_ln=False,
        conv_nl="relu",
        num_blocks=2,
        global_pooling=False,
        no_fnl=True,
        convE=None,
        conv_gated=False,
        # -fc-layers
        fc_layers=3,
        fc_units=1000,
        h_dim=400,
        fc_drop=0,
        fc_bn=False,
        fc_ln=False,
        fc_nl="relu",
        excit_buffer=False,
        fc_gated=False,
        # -prior
        z_dim=20,
        prior="standard",
        n_modes=1,
        # -decoder
        recon_loss="BCE",
        network_output="sigmoid",
        deconv_type="standard",
    ):
        # Set configurations for setting up the model
        super().__init__()
        self.label = "VAE"
        self.args = args
        self.image_size = image_size
        self.image_channels = image_channels
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.depth = depth if convE is None else convE.depth
        # -type of loss to be used for reconstruction
        self.recon_loss = recon_loss  # options: BCE|MSE
        self.network_output = network_output

        # Optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []

        # Prior-related parameters (for "vamp-prior" / "GMM")
        self.prior = prior
        self.n_modes = n_modes
        # -vampprior-specific (note that these are about initializing the vamp-prior's pseudo-inputs):
        self.prior_mean = 0.25  # <-- data-specific!! TO BE CHANGED
        self.prior_sd = 0.05  # <-- data-specific!! TO BE CHANGED

        # Check whether there is at least 1 fc-layer
        if fc_layers < 1:
            raise ValueError("VAE cannot have 0 fully-connected layers!")

        ######------SPECIFY MODEL------######

        ##>----Encoder (= q[z|x])----<##
        self.convE = (
            ConvLayers(
                conv_type=conv_type,
                block_type="basic",
                num_blocks=num_blocks,
                image_channels=image_channels,
                depth=self.depth,
                start_channels=start_channels,
                reducing_layers=reducing_layers,
                batch_norm=conv_bn,
                instance_norm=conv_in,
                layer_norm=conv_ln,
                nl=conv_nl,
                output="none" if no_fnl else "normal",
                global_pooling=global_pooling,
                gated=conv_gated,
                image_size=image_size,
            )
            if (convE is None)
            else convE
        )
        self.flatten = Flatten()
        # ------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        if fc_layers < 2:
            self.fc_layer_sizes = [
                self.conv_out_units
            ]  # --> this results in self.fcE = Identity()
        elif fc_layers == 2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units] + [
                int(x) for x in np.linspace(fc_units, h_dim, num=fc_layers - 1)
            ]
        real_h_dim = h_dim if fc_layers > 1 else self.conv_out_units
        # ------------------------------------------------------------------------------------------#
        self.fcE = MLP(
            size_per_layer=self.fc_layer_sizes,
            drop=fc_drop,
            batch_norm=fc_bn,
            layer_norm=fc_ln,
            nl=fc_nl,
            excit_buffer=excit_buffer,
            gated=fc_gated,
        )
        # to z
        self.toZ = fc_layer_split(
            real_h_dim, z_dim, nl_mean="none", nl_logvar="none"
        )  # , drop=fc_drop)

        ##>----Decoder (= p[x|z])----<##
        out_nl = (
            True
            if fc_layers > 1
            else (True if (self.depth > 0 and not no_fnl) else False)
        )
        real_h_dim_down = (
            h_dim if fc_layers > 1 else self.convE.out_units(image_size, ignore_gp=True)
        )
        self.fromZ = fc_layer(
            z_dim,
            real_h_dim_down,
            batch_norm=(out_nl and fc_bn),
            layer_norm=(out_nl and fc_ln),
            nl=fc_nl if out_nl else "none",
        )
        fc_layer_sizes_down = self.fc_layer_sizes
        fc_layer_sizes_down[0] = self.convE.out_units(image_size, ignore_gp=True)
        # -> if 'gp' is used in forward pass, size of first/final hidden layer differs between forward and backward pass
        self.fcD = MLP(
            size_per_layer=[x for x in reversed(fc_layer_sizes_down)],
            drop=fc_drop,
            batch_norm=fc_bn,
            layer_norm=fc_ln,
            nl=fc_nl,
            gated=fc_gated,
            output=self.network_output if self.depth == 0 else "normal",
        )
        # to image-shape
        self.to_image = Reshape(
            image_channels=self.convE.out_channels if self.depth > 0 else image_channels
        )
        # through deconv-layers
        self.convD = DeconvLayers(
            image_channels=image_channels,
            final_channels=start_channels,
            depth=self.depth,
            reducing_layers=reducing_layers,
            batch_norm=conv_bn,
            instance_norm=conv_in,
            layer_norm=conv_ln,
            nl=conv_nl,
            gated=conv_gated,
            output=self.network_output,
            deconv_type=deconv_type,
        )

        ##>----Prior----<##
        # -if using the vamp-prior, add pseudo-inputs
        if self.prior == "vampprior":
            # -create
            self.add_pseudoinputs()
            # -initialize
            self.initialize_pseudoinputs(
                prior_mean=self.prior_mean, prior_sd=self.prior_sd
            )
        # -if using the GMM-prior, add its parameters
        if self.prior == "GMM":
            # -create
            self.z_class_means = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_logvars = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            # -initialize
            self.z_class_means.data.normal_()
            self.z_class_logvars.data.normal_()

        # Flags whether parts of the network are frozen (so they can be set to evaluation mode during training)
        self.convE.frozen = False
        self.fcE.frozen = False

    ##------ PRIOR --------##

    def add_pseudoinputs(self):
        """Create pseudo-inputs for the vamp-prior."""
        n_inputs = self.image_channels * self.image_size**2
        shape = [self.n_modes, self.image_channels, self.image_size, self.image_size]
        # define nn-object with learnable parameters, that transforms "idle-inputs" to the learnable pseudo-inputs
        self.make_pseudoinputs = nn.Sequential(
            nn.Linear(self.n_modes, n_inputs, bias=False),
            nn.Hardtanh(min_val=0.0, max_val=1.0)
            if self.network_output == "sigmoid"
            else Identity(),
            Shape(shape=shape),
        )
        # create "idle"-input
        self.idle_input = torch.eye(self.n_modes, self.n_modes)

    def initialize_pseudoinputs(self, prior_mean=0.2, prior_sd=0.05):
        """Initialize the learnable parameters of the pseudo-inputs for the vamp-prior."""
        self.make_pseudoinputs[0].weight.data.normal_(prior_mean, prior_sd)

    ##------ NAMES --------##

    def get_name(self):
        convE_label = "{}_".format(self.convE.name) if self.depth > 0 else ""
        fcE_label = (
            "{}_".format(self.fcE.name)
            if self.fc_layers > 1
            else "{}{}_".format("h" if self.depth > 0 else "i", self.conv_out_units)
        )
        z_label = "z{}{}".format(
            self.z_dim,
            ""
            if self.prior == "standard"
            else "-{}{}".format(self.prior, self.n_modes),
        )
        return "{}={}{}{}".format(self.label, convE_label, fcE_label, z_label)

    @property
    def name(self):
        return self.get_name()

    ##------ UTILITIES --------##

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    ##------ LAYERS --------##

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.toZ.list_init_layers()
        list += self.fromZ.list_init_layers()
        list += self.fcD.list_init_layers()
        list += self.convD.list_init_layers()
        return list

    def layer_info(self):
        """Return list with shape of all hidden layers."""
        # create list with hidden convolutional layers
        layer_list = self.convE.layer_info(image_size=self.image_size)
        # add output of final convolutional layer (if there was at least one conv-layer and there's fc-layers after)
        if self.fc_layers > 0 and self.depth > 0:
            layer_list.append(
                [self.conv_out_channels, self.conv_out_size, self.conv_out_size]
            )
        # add layers of the MLP
        if self.fc_layers > 1:
            for layer_id in range(1, self.fc_layers):
                layer_list.append([self.fc_layer_sizes[layer_id]])
        return layer_list

    ##------ FORWARD FUNCTIONS --------##

    def encode(self, x):
        """Pass input through feed-forward connections, to get [z_mean], [z_logvar] and [hE]."""
        # Forward-pass through conv-layers
        image_features = self.flatten(self.convE(x))
        # Forward-pass through fc-layers
        hE = self.fcE(image_features)
        # Get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)
        return z_mean, z_logvar, hE

    def reparameterize(self, mu, logvar):
        """Perform "reparametrization trick" to make these stochastic variables differentiable."""
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()  # .requires_grad_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """Decode latent variable activations.

        INPUT:  - [z]            <2D-tensor>; latent variables to be decoded

        OUTPUT: - [image_recon]  <4D-tensor>"""

        hD = self.fromZ(z)
        image_features = self.fcD(hD)
        image_recon = self.convD(self.to_image(image_features))
        return image_recon

    def forward(self, x, reparameterize=True, **kwargs):
        """Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]

        If [full] is True, output should be a <tuple> consisting of:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x] (or 2 of those: mean & logvar)
        - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
        - [logvar]      None or <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction"""

        mu, logvar, hE = self.encode(x)
        z = self.reparameterize(mu, logvar) if reparameterize else mu
        x_recon = self.decode(z)
        return (x_recon, mu, logvar, z)

    def feature_extractor(self, images):
        """Extract "final features" (i.e., after both conv- and fc-layers of forward pass) from provided images."""
        return self.fcE(self.flatten(self.convE(images)))

    ##------ SAMPLE FUNCTIONS --------##

    def sample(self, size, sample_mode=None, **kwargs):
        """Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device."""

        # set model to eval()-mode
        self.eval()

        # sample for each sample the prior-mode to be used
        if self.prior in ["vampprior", "GMM"] and sample_mode is None:
            sampled_modes = np.random.randint(0, self.n_modes, size)

        # sample z
        if self.prior in ["vampprior", "GMM"]:
            if self.prior == "vampprior":
                with torch.no_grad():
                    # -get pseudo-inputs
                    X = self.make_pseudoinputs(self.idle_input.to(self._device()))
                    # - pass pseudo-inputs through ("variational") encoder
                    prior_means, prior_logvars, _ = self.encode(X)
            else:
                prior_means = self.z_class_means
                prior_logvars = self.z_class_logvars
            # -for each sample to be generated, select the previously sampled mode
            z_means = prior_means[sampled_modes, :]
            z_logvars = prior_logvars[sampled_modes, :]
            with torch.no_grad():
                z = self.reparameterize(z_means, z_logvars)
        else:
            z = torch.randn(size, self.z_dim).to(self._device())

        # decode z into image X
        with torch.no_grad():
            X = self.decode(z)

        # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor
        return X

    ##------ LOSS FUNCTIONS --------##

    def calculate_recon_loss(self, x, x_recon, average=False):
        """Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]"""

        batch_size = x.size(0)
        if self.recon_loss == "MSE":
            # reconL = F.mse_loss(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1), reduction='none')
            # reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
            reconL = -log_Normal_standard(x=x, mean=x_recon, average=average, dim=-1)
        elif self.recon_loss == "BCE":
            reconL = F.binary_cross_entropy(
                input=x_recon.view(batch_size, -1),
                target=x.view(batch_size, -1),
                reduction="none",
            )
            reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
        else:
            raise NotImplementedError("Wrong choice for type of reconstruction-loss!")
        # --> if [average]=True, reconstruction loss is averaged over all pixels/elements (otherwise it is summed)
        #       (averaging over all elements in the batch will be done later)
        return reconL

    def calculate_log_p_z(self, z):
        """Calculate log-likelihood of sampled [z] under the prior distirbution.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")

        OUTPUT: - [log_p_z]   <1D-tensor> of length [batch_size]"""

        if self.prior == "standard":
            log_p_z = log_Normal_standard(z, average=False, dim=1)  # [batch_size]
        elif self.prior in ("vampprior", "GMM"):
            # Get [means] and [logvars] of all (possible) modes
            allowed_modes = list(range(self.n_modes))
            # -calculate/retireve the means and logvars for all modes
            if self.prior == "vampprior":
                X = self.make_pseudoinputs(
                    self.idle_input.to(self._device())
                )  # get pseudo-inputs
                prior_means, prior_logvars, _ = self.encode(
                    X[allowed_modes]
                )  # pass them through encoder
            else:
                prior_means = self.z_class_means[allowed_modes, :]
                prior_logvars = self.z_class_logvars[allowed_modes, :]
            # -rearrange / select for each batch prior-modes to be used
            z_expand = z.unsqueeze(1)  # [batch_size] x 1 x [z_dim]
            means = prior_means.unsqueeze(0)  # 1 x [n_modes] x [z_dim]
            logvars = prior_logvars.unsqueeze(0)  # 1 x [n_modes] x [z_dim]

            # Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
            n_modes = len(allowed_modes)
            a = log_Normal_diag(
                z_expand, mean=means, log_var=logvars, average=False, dim=2
            ) - math.log(n_modes)
            # --> for each element in batch, calculate log-likelihood for all modes: [batch_size] x [n_modes]
            a_max, _ = torch.max(a, dim=1)  # [batch_size]
            # --> for each element in batch, take highest log-likelihood over all modes
            #     this is calculated and used to avoid underflow in the below computation
            a_exp = torch.exp(a - a_max.unsqueeze(1))  # [batch_size] x [n_modes]
            a_logsum = torch.log(
                torch.clamp(torch.sum(a_exp, dim=1), min=1e-40)
            )  # -> sum over modes: [batch_size]
            log_p_z = a_logsum + a_max  # [batch_size]

        return log_p_z

    def calculate_variat_loss(self, z, mu, logvar):
        """Calculate reconstruction loss for each element in the batch.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")
                - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]"""

        if self.prior == "standard":
            # --> calculate analytically
            # ---- see Appendix B from: Kingma & Welling (2014) Auto-Encoding Variational Bayes, ICLR ----#
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        elif self.prior in ("vampprior", "GMM"):
            # --> calculate "empirically"

            ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on prior)
            log_p_z = self.calculate_log_p_z(z)  # [batch_size]

            ## Calculate "log_q_z" (entropy of "reparameterized" [z] given [x])
            log_q_z = log_Normal_diag(z, mean=mu, log_var=logvar, average=False, dim=1)
            # ----->  mu: [batch_size] x [z_dim]; logvar: [batch_size] x [z_dim]; z: [batch_size] x [z_dim]
            # ----->  log_q_z: [batch_size]

            ## Combine
            variatL = -(log_p_z - log_q_z)

        return variatL

    def loss_function(self, x, x_recon, mu, z, logvar, batch_weights=None):
        """Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x]           <4D-tensor> original image
                - [x_recon]     (tuple of 2x) <4D-tensor> reconstructed image in same shape as [x]
                - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                - [z]              <2D-tensor> with reparameterized [z]
                - [logvar]         <2D-tensor> with estimated log(SD^2) of [z]
                - [batch_weights]  <1D-tensor> with a weight for each batch-element (if None, normal average over batch)

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how close distribion [z] is to prior"
        """

        ###-----Reconstruction loss-----###
        batch_size = x.size(0)
        reconL = self.calculate_recon_loss(
            x=x.view(batch_size, -1), average=True, x_recon=x_recon.view(batch_size, -1)
        )  # -> average over pixels
        reconL = weighted_average(
            reconL, weights=batch_weights, dim=0
        )  # -> average over batch

        ###-----Variational loss-----###
        variatL = self.calculate_variat_loss(z=z, mu=mu, logvar=logvar)
        variatL = weighted_average(
            variatL, weights=batch_weights, dim=0
        )  # -> average over batch
        variatL /= (
            self.image_channels * self.image_size**2
        )  # -> divide by # of input-pixels

        # Return a tuple of the calculated losses
        return reconL, variatL

    def estimate_loglikelihood_single(self, x, S=5000, batch_size=128):
        """Estimate average marginal log-likelihood for [x] using [S] importance samples."""

        # Move [x]  to correct device
        # x = x.to(self._device())

        # Run forward pass of model to get [z_mu] and [z_logvar]
        with torch.no_grad():
            z_mu, z_logvar, _ = self.encode(x)

        # Importance samples will be calcualted in batches, get number of required batches
        repeats = int(np.ceil(S / batch_size))

        # For each importance sample, calculate log_likelihood
        for rep in range(repeats):
            batch_size_current = (
                ((S - 1) % batch_size + 1) if rep == (repeats - 1) else batch_size
            )

            # Reparameterize (i.e., sample z_s)
            z = self.reparameterize(
                z_mu.expand(batch_size_current, -1),
                z_logvar.expand(batch_size_current, -1),
            )

            # Calculate log_p_z
            with torch.no_grad():
                log_p_z = self.calculate_log_p_z(z)

            # Calculate log_q_z_x
            log_q_z_x = log_Normal_diag(
                z, mean=z_mu, log_var=z_logvar, average=False, dim=1
            )

            # Calcuate log_p_x_z
            # -reconstruct input
            with torch.no_grad():
                x_recon = self.decode(z)
            # -calculate p_x_z (under Gaussian observation model with unit variance)
            log_p_x_z = log_Normal_standard(x=x, mean=x_recon, average=False, dim=-1)

            # Calculate log-likelihood for each importance sample
            log_likelihoods = log_p_x_z + log_p_z - log_q_z_x

            # Concatanate the log-likelihoods of all importance samples
            all_lls = (
                torch.cat([all_lls, log_likelihoods]) if rep > 0 else log_likelihoods
            )

        # Calculate average log-likelihood over all importance samples for this test sample
        #  (for this, convert log-likelihoods back to likelihoods before summing them!)
        log_likelihood = all_lls.logsumexp(dim=0) - np.log(S)

        return log_likelihood


class MLP(nn.Module):
    """Module for a multi-layer perceptron (MLP). Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x ... x [size_per_layer[0]] tensor
    Output: (tuple of) [batch_size] x ... x [size_per_layer[-1]] tensor"""

    def __init__(
        self,
        input_size=1000,
        output_size=10,
        layers=2,
        hid_size=1000,
        hid_smooth=None,
        size_per_layer=None,
        drop=0,
        batch_norm=True,
        layer_norm=False,
        nl="relu",
        bias=True,
        excitability=False,
        excit_buffer=False,
        gated=False,
        output="normal",
    ):
        """sizes: 0th=[input], 1st=[hid_size], ..., 1st-to-last=[hid_smooth], last=[output].
        [input_size]       # of inputs
        [output_size]      # of units in final layer
        [layers]           # of layers
        [hid_size]         # of units in each hidden layer
        [hid_smooth]       if None, all hidden layers have [hid_size] units, else # of units linearly in-/decreases s.t.
                             final hidden layer has [hid_smooth] units (if only 1 hidden layer, it has [hid_size] units)
        [size_per_layer]   None or <list> with for each layer number of units (1st element = number of inputs)
                                --> overwrites [input_size], [output_size], [layers], [hid_size] and [hid_smooth]
        [drop]             % of each layer's inputs that is randomly set to zero during training
        [batch_norm]       <bool>; if True, batch-normalization is applied to each layer
        [nl]               <str>; type of non-linearity to be used (options: "relu", "leakyrelu", "none")
        [gated]            <bool>; if True, each linear layer has an additional learnable gate
                                    (whereby the gate is controlled by the same input as that goes through the gate)
        [output]           <str>; if - "normal", final layer is same as all others
                                     - "none", final layer has no non-linearity
                                     - "sigmoid", final layer has sigmoid non-linearity
        """

        super().__init__()
        self.output = output

        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                if hid_smooth is not None:
                    hidden_sizes = [
                        int(x)
                        for x in np.linspace(hid_size, hid_smooth, num=layers - 1)
                    ]
                else:
                    hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = (
                [input_size] + hidden_sizes + [output_size]
                if layers > 0
                else [input_size]
            )
        self.layers = len(size_per_layer) - 1

        # set label for this module
        # -determine "non-default options"-label
        nd_label = "{drop}{bias}{exc}{bn}{nl}{gate}".format(
            drop="" if drop == 0 else "d{}".format(drop),
            bias="" if bias else "n",
            exc="e" if excitability else "",
            bn="b" if batch_norm else "",
            nl="l" if nl == "leakyrelu" else "",
            gate="g" if gated else "",
        )
        nd_label = "{}{}".format(
            "" if nd_label == "" else "-{}".format(nd_label),
            "" if output == "normal" else "-{}".format(output),
        )
        # -set label
        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement == "" else "x", i)
        self.label = "F{}{}".format(size_statement, nd_label) if self.layers > 0 else ""

        # set layers
        for lay_id in range(1, self.layers + 1):
            # number of units of this layer's input and output
            in_size = size_per_layer[lay_id - 1]
            out_size = size_per_layer[lay_id]
            # define and set the fully connected layer
            use_bn = (
                False
                if (lay_id == self.layers and not output == "normal")
                else batch_norm
            )
            use_ln = (
                False
                if (lay_id == self.layers and not output == "normal")
                else layer_norm
            )
            layer = fc_layer(
                in_size,
                out_size,
                bias=bias,
                excitability=excitability,
                excit_buffer=excit_buffer,
                batch_norm=use_bn,
                layer_norm=use_ln,
                gated=gated,
                nl=("none" if output == "none" else nn.Sigmoid())
                if (lay_id == self.layers and not output == "normal")
                else nl,
                drop=drop if lay_id > 1 else 0.0,
            )
            setattr(self, "fcLayer{}".format(lay_id), layer)

        # if no layers, add "identity"-module to indicate in this module's representation nothing happens
        if self.layers < 1:
            self.noLayers = Identity()

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False, **kwargs):
        # Initiate <list> for keeping track of intermediate hidden-(pre)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all fc-layers
        for lay_id in range(skip_first + 1, self.layers + 1 - skip_last):
            (x, pre_act) = getattr(self, "fcLayer{}".format(lay_id))(x, return_pa=True)
            if return_lists:
                pre_act_list.append(pre_act)  # -> for each layer, store pre-activations
                if lay_id < (self.layers - skip_last):
                    hidden_act_list.append(
                        x
                    )  # -> for all but last layer, store hidden activations
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for layer_id in range(1, self.layers + 1):
            list += getattr(self, "fcLayer{}".format(layer_id)).list_init_layers()
        return list


class fc_layer(nn.Module):
    """Fully connected layer, with possibility of returning "pre-activations".

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor"""

    def __init__(
        self,
        in_size,
        out_size,
        nl=nn.ReLU(),
        drop=0.0,
        bias=True,
        excitability=False,
        excit_buffer=False,
        batch_norm=False,
        layer_norm=False,
        gated=False,
    ):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout(drop)
        self.linear = LinearExcitability(
            in_size,
            out_size,
            bias=False if batch_norm else bias,
            excitability=excitability,
            excit_buffer=excit_buffer,
        )
        assert not (batch_norm and layer_norm)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_size)
        if layer_norm:
            self.bn = nn.LayerNorm(out_size)
        if gated:
            self.gate = nn.Linear(in_size, out_size)
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == "none":
            self.nl = get_activation_from_name(nl)()

    def forward(self, x, return_pa=False, **kwargs):
        input = self.dropout(x) if hasattr(self, "dropout") else x
        pre_activ = (
            self.bn(self.linear(input)) if hasattr(self, "bn") else self.linear(input)
        )
        gate = self.sigmoid(self.gate(x)) if hasattr(self, "gate") else None
        gated_pre_activ = gate * pre_activ if hasattr(self, "gate") else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, "nl") else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.linear, self.gate] if hasattr(self, "gate") else [self.linear]


class fc_layer_split(nn.Module):
    """Fully connected layer outputting [mean] and [logvar] for each unit.

    Input:  [batch_size] x ... x [in_size] tensor
    Output: tuple with two [batch_size] x ... x [out_size] tensors"""

    def __init__(
        self,
        in_size,
        out_size,
        nl_mean=nn.Sigmoid(),
        nl_logvar=nn.Hardtanh(min_val=-4.5, max_val=0.0),
        drop=0.0,
        bias=True,
        excitability=False,
        excit_buffer=False,
        batch_norm=False,
        gated=False,
    ):
        super().__init__()

        self.mean = fc_layer(
            in_size,
            out_size,
            drop=drop,
            bias=bias,
            excitability=excitability,
            excit_buffer=excit_buffer,
            batch_norm=batch_norm,
            gated=gated,
            nl=nl_mean,
        )
        self.logvar = fc_layer(
            in_size,
            out_size,
            drop=drop,
            bias=False,
            excitability=excitability,
            excit_buffer=excit_buffer,
            batch_norm=batch_norm,
            gated=gated,
            nl=nl_logvar,
        )

    def forward(self, x):
        return (self.mean(x), self.logvar(x))

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list


class ConvLayers(nn.Module):
    """Convolutional feature extractor model for (natural) images. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x [image_channels] x [image_size] x [image_size] tensor
    Output: [batch_size] x [out_channels] x [out_size] x [out_size] tensor
                - with [out_channels] = [start_channels] x 2**[reducing_layers] x [block.expansion]
                       [out_size] = [image_size] / 2**[reducing_layers]"""

    def __init__(
        self,
        conv_type="standard",
        block_type="basic",
        num_blocks=2,
        image_channels=3,
        depth=5,
        start_channels=16,
        reducing_layers=None,
        batch_norm=True,
        nl="relu",
        output="normal",
        global_pooling=False,
        gated=False,
        instance_norm=False,
        layer_norm=False,
        image_size=32,
    ):
        """Initialize stacked convolutional layers (either "standard" or "res-net" ones--1st layer is always standard).

        [conv_type]         <str> type of conv-layers to be used: [standard|resnet]
        [block_type]        <str> block-type to be used: [basic|bottleneck] (only relevant if [type]=resNet)
        [num_blocks]        <int> or <list> (with len=[depth]-1) of # blocks in each layer
        [image_channels]    <int> # channels of input image to encode
        [depth]             <int> # layers
        [start_channels]    <int> # channels in 1st layer, doubled in every "rl" (=reducing layer)
        [reducing_layers]   <int> # layers in which image-size is halved & # channels doubled (default=[depth]-1)
                                      ("rl"'s are the last conv-layers; in 1st layer # channels cannot double)
        [batch_norm]        <bool> whether to use batch-norm after each convolution-operation
        [nl]                <str> non-linearity to be used: [relu|leakyrelu]
        [output]            <str>  if - "normal", final layer is same as all others
                                      - "none", final layer has no batchnorm or non-linearity
        [global_pooling]    <bool> whether to include global average pooling layer at very end
        [gated]             <bool> whether conv-layers should be gated (not implemented for ResNet-layers)
        """

        # Process type and number of blocks
        conv_type = "standard" if depth < 2 else conv_type
        if conv_type == "resNet":
            assert False
            # num_blocks = [num_blocks] * (depth - 1) if type(num_blocks) == int else num_blocks
            # assert len(num_blocks) == (depth - 1)
            # block = conv_layers.Bottleneck if block_type == "bottleneck" else conv_layers.BasicBlock

        # Prepare label
        type_label = (
            "C"
            if conv_type == "standard"
            else "R{}".format("b" if block_type == "bottleneck" else "")
        )
        channel_label = "{}-{}x{}".format(image_channels, depth, start_channels)
        block_label = ""
        if conv_type == "resNet" and depth > 1:
            block_label += "-"
            for block_num in num_blocks:
                block_label += "b{}".format(block_num)
        nd_label = "{bn}{nl}{gp}{gate}{out}".format(
            bn="b" if batch_norm else "",
            nl="l" if nl == "leakyrelu" else "",
            gp="p" if global_pooling else "",
            gate="g" if gated else "",
            out="n" if output == "none" else "",
        )
        nd_label = "" if nd_label == "" else "-{}".format(nd_label)

        # Set configurations
        super().__init__()
        self.depth = depth
        self.rl = (
            depth - 1
            if (reducing_layers is None)
            else (reducing_layers if (depth + 1) > reducing_layers else depth)
        )
        rl_label = "" if self.rl == (self.depth - 1) else "-rl{}".format(self.rl)
        self.label = "{}{}{}{}{}".format(
            type_label, channel_label, block_label, rl_label, nd_label
        )
        # self.block_expansion = block.expansion if conv_type == "resNet" else 1
        self.block_expansion = 1
        # -> constant by which # of output channels of each block is multiplied (if >1, it creates "bottleneck"-effect)
        double_factor = (
            self.rl if self.rl < depth else depth - 1
        )  # -> how often # start-channels is doubled
        self.out_channels = (
            (start_channels * 2**double_factor) * self.block_expansion
            if depth > 0
            else image_channels
        )
        # -> number channels in last layer (as seen from image)
        self.start_channels = start_channels  # -> number channels in 1st layer (doubled in every "reducing layer")
        self.global_pooling = global_pooling  # -> whether or not average global pooling layer should be added at end

        # Conv-layers
        output_channels = start_channels
        cur_img_size = image_size
        for layer_id in range(1, depth + 1):
            # should this layer down-sample? --> last [self.rl] layers should be down-sample layers
            reducing = True if (layer_id > (depth - self.rl)) else False
            if reducing:
                cur_img_size //= 2
            # calculate number of this layer's input and output channels
            input_channels = (
                image_channels
                if layer_id == 1
                else output_channels * self.block_expansion
            )
            output_channels = (
                output_channels * 2
                if (reducing and not layer_id == 1)
                else output_channels
            )
            # define and set the convolutional-layer
            if conv_type == "standard" or layer_id == 1:
                bnorm = False if output == "none" and layer_id == depth else batch_norm
                inorm = (
                    False if output == "none" and layer_id == depth else instance_norm
                )
                lnorm = False if output == "none" and layer_id == depth else layer_norm
                conv_layer = _conv_layer(
                    input_channels,
                    output_channels,
                    stride=2 if reducing else 1,
                    drop=0,
                    nl="no" if output == "none" and layer_id == depth else nl,
                    batch_norm=bnorm,
                    instance_norm=inorm,
                    layer_norm=lnorm,
                    gated=False if output == "none" and layer_id == depth else gated,
                    cur_img_size=cur_img_size,
                )
            else:
                assert False

            setattr(self, "convLayer{}".format(layer_id), conv_layer)
        # Perform pooling (if requested)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1)) if global_pooling else Identity()

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False):
        # Initiate <list> for keeping track of intermediate hidden (pre-)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all conv-layers
        for layer_id in range(skip_first + 1, self.depth + 1 - skip_last):
            (x, pre_act) = getattr(self, "convLayer{}".format(layer_id))(
                x, return_pa=True
            )
            if return_lists:
                pre_act_list.append(pre_act)  # -> for each layer, store pre-activations
                if layer_id < (self.depth - skip_last):
                    hidden_act_list.append(
                        x
                    )  # -> for all but last layer, store hidden activations
        # Global average pooling (if requested)
        x = self.pooling(x)
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    def out_size(self, image_size, ignore_gp=False):
        """Given [image_size] of input, return the size of the "final" image that is outputted."""
        out_size = (
            int(np.ceil(image_size / 2 ** (self.rl))) if self.depth > 0 else image_size
        )
        return 1 if (self.global_pooling and not ignore_gp) else out_size

    def out_units(self, image_size, ignore_gp=False):
        """Given [image_size] of input, return the total number of units in the output."""
        return self.out_channels * self.out_size(image_size, ignore_gp=ignore_gp) ** 2

    def layer_info(self, image_size):
        """Return list with shape of all hidden layers."""
        layer_list = []
        reduce_number = 0  # keep track how often image-size has been halved
        double_number = 0  # keep track how often channel number has been doubled
        for layer_id in range(1, self.depth):
            reducing = True if (layer_id > (self.depth - self.rl)) else False
            if reducing:
                reduce_number += 1
            if reducing and layer_id > 1:
                double_number += 1
            pooling = (
                True if self.global_pooling and layer_id == (self.depth - 1) else False
            )
            expansion = 1 if layer_id == 1 else self.block_expansion
            # add shape of this layer to list
            layer_list.append(
                [
                    (self.start_channels * 2**double_number) * expansion,
                    1 if pooling else int(np.ceil(image_size / 2**reduce_number)),
                    1 if pooling else int(np.ceil(image_size / 2**reduce_number)),
                ]
            )
        return layer_list

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for layer_id in range(1, self.depth + 1):
            list += getattr(self, "convLayer{}".format(layer_id)).list_init_layers()
        return list

    @property
    def name(self):
        return self.label

    def _device(self):
        return next(self.parameters()).device


class DeconvLayers(nn.Module):
    """ "Deconvolutional" feature decoder model for (natural) images. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [batch_size] x [in_channels] x [in_size] x [in_size] tensor
    Output: (tuple of) [batch_size] x [image_channels] x [final_size] x [final_size] tensor
                - with [final_size] = [in_size] x 2**[reducing_layers]
                       [in_channels] = [final_channels] x 2**min([reducing_layers], [depth]-1)
    """

    def __init__(
        self,
        image_channels=3,
        final_channels=16,
        depth=5,
        reducing_layers=None,
        batch_norm=True,
        instance_norm=False,
        layer_norm=False,
        nl="relu",
        gated=False,
        output="normal",
        smaller_kernel=False,
        deconv_type="standard",
    ):
        """[image_channels] # channels of image to decode
        [final_channels]    # channels in layer before output, was halved in every "rl" (=reducing layer) when moving
                                through model; corresponds to [start_channels] in "ConvLayers"-module
        [depth]             # layers (seen from the image, # channels is halved in each layer going to output image)
        [reducing_layers]   # of layers in which image-size is doubled & number of channels halved (default=[depth]-1)
                               ("rl"'s are the first conv-layers encountered--i.e., last conv-layers as seen from image)
                               (note that in the last layer # channels cannot be halved)
        [batch_norm]        <bool> whether to use batch-norm after each convolution-operation
        [nl]                <str> what non-linearity to use -- choices: [relu, leakyrelu, sigmoid, none]
        [gated]             <bool> whether deconv-layers should be gated
        [output]            <str>; if - "normal", final layer is same as all others
                                      - "none", final layer has no non-linearity
                                      - "sigmoid", final layer has sigmoid non-linearity
        [smaller_kernel]    <bool> if True, use kernel-size of 2 (instead of 4) & without padding in reducing-layers
        """

        # configurations
        super().__init__()
        self.depth = depth if depth > 0 else 0
        self.rl = (
            self.depth - 1
            if (reducing_layers is None)
            else min(self.depth, reducing_layers)
        )
        type_label = "Deconv" if deconv_type == "standard" else "DeResNet"
        nd_label = "{bn}{nl}{gate}{out}".format(
            bn="-bn" if batch_norm else "",
            nl="-lr" if nl == "leakyrelu" else "",
            gate="-gated" if gated else "",
            out="" if output == "normal" else "-{}".format(output),
        )
        self.label = "{}-ic{}-{}x{}-rl{}{}{}".format(
            type_label,
            image_channels,
            self.depth,
            final_channels,
            self.rl,
            "s" if smaller_kernel else "",
            nd_label,
        )
        self.in_channels = final_channels * 2 ** min(
            self.rl, self.depth - 1
        )  # -> input-channels for deconv
        self.final_channels = final_channels  # -> channels in layer before output
        self.image_channels = image_channels  # -> output-channels for deconv

        # "Deconv"- / "transposed conv"-layers
        if self.depth > 0:
            output_channels = self.in_channels
            for layer_id in range(1, self.depth + 1):
                # should this layer down-sample? --> first [self.rl] layers should be down-sample layers
                reducing = True if (layer_id < (self.rl + 1)) else False
                # update number of this layer's input and output channels
                input_channels = output_channels
                output_channels = (
                    int(output_channels / 2) if reducing else output_channels
                )
                # define and set the "deconvolutional"-layer
                if deconv_type == "standard":
                    use_bn = batch_norm if layer_id < self.depth else False
                    use_in = instance_norm if layer_id < self.depth else False
                    assert not layer_norm, "not implemented!"
                    new_layer = deconv_layer(
                        input_channels,
                        output_channels if layer_id < self.depth else image_channels,
                        stride=2 if reducing else 1,
                        batch_norm=use_bn,
                        instance_norm=use_in,
                        nl=nl
                        if layer_id < self.depth or output == "normal"
                        else ("none" if output == "none" else nn.Sigmoid()),
                        gated=gated,
                        smaller_kernel=smaller_kernel,
                    )
                else:
                    assert False
                    # assert not (instance_norm or layer_norm), "not implemented!"
                    # new_layer = deconv_res_layer(
                    #     input_channels, output_channels if layer_id < self.depth else image_channels,
                    #     stride=2 if reducing else 1, batch_norm=batch_norm if layer_id < self.depth else False,
                    #     nl=nl, smaller_kernel=smaller_kernel, output="normal" if layer_id < self.depth else output
                    # )
                setattr(self, "deconvLayer{}".format(layer_id), new_layer)

    def forward(self, x, skip_first=0, skip_last=0, return_lists=False):
        # Initiate <list> for keeping track of intermediate hidden (pre-)activations
        if return_lists:
            hidden_act_list = []
            pre_act_list = []
        # Sequentially pass [x] through all "deconv"-layers
        if self.depth > 0:
            for layer_id in range(skip_first + 1, self.depth + 1 - skip_last):
                (x, pre_act) = getattr(self, "deconvLayer{}".format(layer_id))(
                    x, return_pa=True
                )
                if return_lists:
                    pre_act_list.append(
                        pre_act
                    )  # -> for each layer, store pre-activations
                    if layer_id < (self.depth - skip_last):
                        hidden_act_list.append(
                            x
                        )  # -> for all but last layer, store hidden activations
        # Return final [x], if requested along with [hidden_act_list] and [pre_act_list]
        return (x, hidden_act_list, pre_act_list) if return_lists else x

    def image_size(self, in_units):
        """Given the number of units fed in, return the size of the target image."""
        input_image_size = np.sqrt(
            in_units / self.in_channels
        )  # -> size of image fed into last layer (as seen from image)
        return input_image_size * 2**self.rl

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        for layer_id in range(1, self.depth + 1):
            list += getattr(self, "deconvLayer{}".format(layer_id)).list_init_layers()
        return list

    @property
    def name(self):
        return self.label


class _conv_layer(nn.Module):
    """Standard convolutional layer. Possible to return pre-activations."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=1,
        drop=0,
        batch_norm=False,
        instance_norm=False,
        layer_norm=False,
        nl=nn.ReLU(),
        bias=True,
        gated=False,
        cur_img_size=-1,
    ):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout2d(drop)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        assert int(batch_norm) + int(instance_norm) + int(layer_norm) <= 1
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_planes)
        if instance_norm:
            self.bn = nn.InstanceNorm2d(out_planes)
        if layer_norm:
            self.bn = nn.LayerNorm([out_planes, cur_img_size, cur_img_size])
        if gated:
            self.gate = nn.Conv2d(
                in_planes,
                out_planes,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif not nl == "none":
            self.nl = get_activation_from_name(nl)()

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, "dropout") else x
        pre_activ = (
            self.bn(self.conv(input)) if hasattr(self, "bn") else self.conv(input)
        )
        gate = self.sigmoid(self.gate(x)) if hasattr(self, "gate") else None
        gated_pre_activ = gate * pre_activ if hasattr(self, "gate") else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, "nl") else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.conv]


class deconv_layer(nn.Module):
    """Standard "deconvolutional" layer. Possible to return pre-activations."""

    def __init__(
        self,
        input_channels,
        output_channels,
        stride=1,
        drop=0,
        batch_norm=True,
        instance_norm=False,
        nl="relu",
        bias=True,
        gated=False,
        smaller_kernel=False,
    ):
        super().__init__()
        if drop > 0:
            self.dropout = nn.Dropout2d(drop)
        self.deconv = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            bias=bias,
            stride=stride,
            kernel_size=(2 if smaller_kernel else 4) if stride == 2 else 3,
            padding=0 if (stride == 2 and smaller_kernel) else 1,
        )
        assert int(batch_norm) + int(instance_norm) <= 1
        if batch_norm:
            self.bn = nn.BatchNorm2d(output_channels)
        if instance_norm:
            self.bn = nn.InstanceNorm2d(output_channels)

        if gated:
            self.gate = nn.ConvTranspose2d(
                input_channels,
                output_channels,
                bias=False,
                stride=stride,
                kernel_size=(2 if smaller_kernel else 4) if stride == 2 else 3,
                padding=0 if (stride == 2 and smaller_kernel) else 1,
            )
            self.sigmoid = nn.Sigmoid()
        if isinstance(nl, nn.Module):
            self.nl = nl
        elif nl in ("sigmoid", "hardtanh"):
            self.nl = (
                nn.Sigmoid()
                if nl == "sigmoid"
                else nn.Hardtanh(min_val=-4.5, max_val=0)
            )
        elif not nl == "none":
            self.nl = (
                nn.ReLU()
                if nl == "relu"
                else (nn.LeakyReLU() if nl == "leakyrelu" else Identity())
            )

    def forward(self, x, return_pa=False):
        input = self.dropout(x) if hasattr(self, "dropout") else x
        pre_activ = (
            self.bn(self.deconv(input)) if hasattr(self, "bn") else self.deconv(input)
        )
        gate = self.sigmoid(self.gate(x)) if hasattr(self, "gate") else None
        gated_pre_activ = gate * pre_activ if hasattr(self, "gate") else pre_activ
        output = self.nl(gated_pre_activ) if hasattr(self, "nl") else gated_pre_activ
        return (output, gated_pre_activ) if return_pa else output

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        return [self.deconv]


class deconv_layer_split(nn.Module):
    """ "Deconvolutional" layer outputing [mean] and [logvar] for each unit."""

    def __init__(
        self,
        input_channels,
        output_channels,
        nl_mean="sigmoid",
        nl_logvar="hardtanh",
        stride=1,
        drop=0,
        batch_norm=True,
        bias=True,
        gated=False,
        smaller_kernel=False,
    ):
        super().__init__()
        self.mean = deconv_layer(
            input_channels,
            output_channels,
            nl=nl_mean,
            smaller_kernel=smaller_kernel,
            stride=stride,
            drop=drop,
            batch_norm=batch_norm,
            bias=bias,
            gated=gated,
        )
        self.logvar = deconv_layer(
            input_channels,
            output_channels,
            nl=nl_logvar,
            smaller_kernel=smaller_kernel,
            stride=stride,
            drop=drop,
            batch_norm=batch_norm,
            bias=False,
            gated=gated,
        )

    def forward(self, x, return_pa=False):
        mean, pre_activ = self.mean(x, return_pa=True)
        logvar = self.logvar(x)
        return ((mean, logvar), pre_activ) if return_pa else (mean, logvar)

    def list_init_layers(self):
        """Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers)."""
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list


def linearExcitability(input, weight, excitability=None, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = c(xA^T) + b`.

    Shape:
        - input:        :math:`(N, *, in\_features)`
        - weight:       :math:`(out\_features, in\_features)`
        - excitability: :math:`(out\_features)`
        - bias:         :math:`(out\_features)`
        - output:       :math:`(N, *, out\_features)`
    (NOTE: `*` means any number of additional dimensions)
    """
    if excitability is not None:
        output = input.matmul(weight.t()) * excitability
    else:
        output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output


class LinearExcitability(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = c(Ax) + b`

    Args:
        in_features:    size of each input sample
        out_features:   size of each output sample
        bias:           if 'False', layer will not learn an additive bias-parameter (DEFAULT=True)
        excitability:   if 'False', layer will not learn a multiplicative excitability-parameter (DEFAULT=True)

    Shape:
        - input:    :math:`(N, *, in\_features)` where `*` means any number of additional dimensions
        - output:   :math:`(N, *, out\_features)` where all but the last dimension are the same shape as the input.

    Attributes:
        weight:         the learnable weights of the module of shape (out_features x in_features)
        excitability:   the learnable multiplication terms (out_features)
        bias:           the learnable bias of the module of shape (out_features)
        excit_buffer:   fixed multiplication variable (out_features)

    Examples::

        >>> m = LinearExcitability(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        excitability=False,
        excit_buffer=False,
    ):
        super(LinearExcitability, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if excitability:
            self.excitability = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("excitability", None)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        if excit_buffer:
            buffer = torch.Tensor(out_features).uniform_(1, 1)
            self.register_buffer("excit_buffer", buffer)
        else:
            self.register_buffer("excit_buffer", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Modifies the parameters "in-place" to reset them at appropriate initialization values"""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.excitability is not None:
            self.excitability.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Running this model's forward step requires/returns:
        INPUT: -[input]: [batch_size]x[...]x[in_features]
        OUTPUT: -[output]: [batch_size]x[...]x[hidden_features]"""
        if self.excit_buffer is None:
            excitability = self.excitability
        elif self.excitability is None:
            excitability = self.excit_buffer
        else:
            excitability = self.excitability * self.excit_buffer
        return linearExcitability(input, self.weight, excitability, self.bias)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ")"
        )


def weighted_average(tensor, weights=None, dim=0):
    """Computes weighted average of [tensor] over dimension [dim]."""
    if weights is None:
        mean = torch.mean(tensor, dim=dim)
    else:
        batch_size = tensor.size(dim) if len(tensor.size()) > 0 else 1
        assert len(weights) == batch_size
        # sum_weights = sum(weights)
        # norm_weights = torch.Tensor([weight/sum_weights for weight in weights]).to(tensor.device)
        norm_weights = torch.tensor([weight for weight in weights]).to(tensor.device)
        mean = torch.mean(norm_weights * tensor, dim=dim)
    return mean


def log_Normal_standard(x, mean=0, average=False, dim=None):
    """Calculate log-likelihood of sample [x] under Gaussian distribution(s) with mu=[mean], diag_var=I.
    NOTES: [dim]=-1    summing / averaging over all but the first dimension
           [dim]=None  summing / averaging is done over all dimensions"""
    log_normal = -0.5 * torch.pow(x - mean, 2)
    if dim is not None and dim == -1:
        log_normal = log_normal.view(log_normal.size(0), -1)
        dim = 1
    if average:
        return (
            torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
        )
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    """Calculate log-likelihood of sample [x] under Gaussian distribution(s) with mu=[mean], diag_var=exp[log_var].
    NOTES: [dim]=-1    summing / averaging over all but the first dimension
           [dim]=None  summing / averaging is done over all dimensions"""
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if dim is not None and dim == -1:
        log_normal = log_normal.view(log_normal.size(0), -1)
        dim = 1
    if average:
        return (
            torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
        )
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)


class Identity(nn.Module):
    """A nn-module to simply pass on the input data."""

    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "()"
        return tmpstr


class Shape(nn.Module):
    """A nn-module to shape a tensor of shape [shape]."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.dim = len(shape)

    def forward(self, x):
        return x.view(*self.shape)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "(shape = {})".format(self.shape)
        return tmpstr


class Reshape(nn.Module):
    """A nn-module to reshape a tensor(-tuple) to a 4-dim "image"-tensor(-tuple) with [image_channels] channels."""

    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x):
        if type(x) == tuple:
            batch_size = x[0].size(0)  # first dimenstion should be batch-dimension.
            image_size = int(
                np.sqrt(x[0].nelement() / (batch_size * self.image_channels))
            )
            return (
                x_item.view(batch_size, self.image_channels, image_size, image_size)
                for x_item in x
            )
        else:
            batch_size = x.size(0)  # first dimenstion should be batch-dimension.
            image_size = int(np.sqrt(x.nelement() / (batch_size * self.image_channels)))
            return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "(channels = {})".format(self.image_channels)
        return tmpstr


class Flatten(nn.Module):
    """A nn-module to flatten a multi-dimensional tensor to 2-dim tensor."""

    def forward(self, x):
        batch_size = x.size(0)  # first dimenstion should be batch-dimension.
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "()"
        return tmpstr
