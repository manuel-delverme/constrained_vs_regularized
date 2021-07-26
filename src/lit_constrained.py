import torch_constrained

import numpy as np
import torch
import pytorch_lightning as pl
#from torch.optim.lr_scheduler import StepLR

def tensor2dict(tensor, const_names=None):
    # Converts 1d tensor to dictionary. Used for logging
    if const_names is None:
        const_names = ["aux_objective" + str(i+1) for i,_ in enumerate(tensor)]
    return {name: val for name,val in zip(const_names,tensor)}

def multipliers2dict(mult_list, mult_name):
    # Converts 1d tensor to dictionary. Used for logging
    values = {mult_name + '_' + str(i+1): val.weight for i,val in enumerate(mult_list)}
    grads = {mult_name + '_grad_' + str(i+1): - val.weight.grad for i,val in enumerate(mult_list)}
    return {**values, **grads}

def reduce(tensor):
    # Produces the mean of a logs of tensor, ignoring zero-valued elements and nans
    tensor = tensor[tensor>0]
    tensor = tensor.detach().cpu().numpy()
    return np.nanmean(tensor)

class LitConstrainedModel(pl.LightningModule):
    """
    This class implements a pytorch lightning module with the possibility
    of training under a torch_constrained framework.
    """
    def __init__(
        self, optimizer_class, optimizer_kwargs={}, le_levels=None, eq_levels=None,
        le_names=None, eq_names=None, model_lr=1e-3, dual_lr=0., gamma=1,
        log_constraints=False, metrics = False):
        """
        Args:
            optimizer_class (torch optimizer): an optimizer class for the model's
                parameters. If constraints are provided, the optimizer must have
                an extrapolation method.
            optimizer_kwargs (dict, optional): keyword arguments for the initialization.
                of the optimizer. Defaults to {}.
            le_levels (float list, optional): Levels for less or equal
                constraints on the different objectives. Defaults to None,
                where no le constraint is considered.
            eq_levels (float list, optional): Levels for equality
                constraints on the different objectives. Defaults to None,
                where no le constraint is considered.
            le_names (str list, optional): names for the different constraints.
                By default, generic names are generated.
            eq_names (str list, optional): names for the different constraints.
                By default, generic names are generated.
            model_lr (float, optional): learning rate for the model's
                parameters. Defaults to 1e-3.
            dual_lr (float, optional): learning rate for the dual
                parameters. Defaults to 0.
            gamma (int, optional): unused. Defaults to 1.
            log_constraints (bool, optional): whether to log the values of
                the constrained objectives and the dual variables during
                training. Defaults to False.
            metrics (bool, optional): whether to log user-defined metrics
                during training. Defaults to False.
        """
        super().__init__()

        # Auto optimization is 'disabled' so we can do manual optimizer steps
        self.automatic_optimization = False

        ## Constrained Optimization parameters
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.model_lr = model_lr
        self.dual_lr = dual_lr
        #self.le_damp = le_damp

        # Constraints
        self.le_levels = None if le_levels is None else torch.tensor(le_levels)
        self.le_names = None if le_names is None else le_names
        self.eq_levels = None if eq_levels is None else torch.tensor(eq_levels)
        self.eq_names = None if eq_names is None else eq_names

        # Are we solving a constrained optimization problem?
        self.is_constrained = le_levels is not None or eq_levels is not None

        # Logging
        self.log_constraints = log_constraints
        self.metrics = metrics

        # The optimizer is mostly hard coded
        self.gamma = gamma # lr scheduling

    def configure_optimizers(self):
        """
        Defines the optimizer for model training. If constrains are set on
        the training optimization, a ConstrainedOptimizer from
        torch_constrained is returned. Otherwise, a torch Optimizer is returned.

        Returns:
            torch optimizer: optimizer for the class' parameters. This may
                include dual variables associated with the constraints.
        """
        if self.is_constrained:
            # Then its a constrained optimization problem
            optimizer = torch_constrained.ConstrainedOptimizer(
                self.optimizer_class, # let the user decide
                torch_constrained.ExtraSGD, # dual
                lr_x=self.model_lr,
                lr_y=self.dual_lr,
                primal_parameters=list(self.parameters()),
            )
        else:
            # it is a scalar optimization problem
            optimizer = self.optimizer_class(
                self.parameters(),
                lr=self.model_lr
                )

        # Scheduler - mostly hard-coded
        #self.scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)

        return optimizer

    def eval_gaps(self, vals, const_type):
        """
        Calculate the gap between the values of the constrained objectives
        and their respective levels.

        Args:
            vals (1D tensor): the evaluated values of objectives.
            const_type (str): either "equality" or "inequality".
        Returns:
            1D tensor: feasibility gaps of the constraints
        """
        levels = self.le_levels if const_type == "inequality" else self.eq_levels
        gaps = vals - levels.to(self.device)
        gaps = [gaps.reshape(1, -1), ] # format for optimizer. Necessary?
        return gaps

    def training_step(self, batch, batch_idx):
        """
        Implements a training step for the model.

        Args:
            batch (tensor or structure of tensors): a training batch.
        """
        # For logging - need to PR torch_constrained
        loss, eq_vals, le_vals = self.eval_losses(batch)

        if self.is_constrained:
            def closure():
                """
                Evaluates the different objectives on the current state of the
                model and an infered batch. Feasibility gaps are estimated for constrained
                objectives. Depends on user-defined function "eval_losses".

                Returns:
                    tensors: the main objective, the objectives constrained with
                    equalities and those constrained with inequalities.
                """
                loss, eq_vals, le_vals = self.eval_losses(batch)
                eq_gaps = self.eval_gaps(eq_vals, "equality") if self.eq_levels is not None else None
                le_gaps = self.eval_gaps(le_vals, "inequality") if self.le_levels is not None else None
                return loss, eq_gaps, le_gaps

            # JC ToDo: self.optimizers() can return a list.
            # torch_constrained handles most of the heavy lifting.
            lagrangian = self.optimizers().optimizer.step(closure=closure)
        else:
            # Only primal params are optimized, on a vanilla Pytorch way.
            self.optimizers().optimizer.zero_grad() #JC ToDo
            loss.backward()
            self.optimizers().optimizer.step() # JC ToDo
            lagrangian = loss

        self.log_paths(lagrangian, loss, eq_vals, le_vals, batch)

    def validation_step(self, batch, batch_idx):
        """
        Implements a validation step for the model. Objectives are evaluated
        for their subsequent logging alongside optional, user defined
        performance metrics.

        Args:
            batch (tensor or structure of tensors): a validation batch.
        Returns:
            float: the lagrangian of the optimization problem. Returned for
            early stopping purposes.
        """
        # Losses
        loss, eq_vals, le_vals = self.eval_losses(batch)
        eq_gaps = self.eval_gaps(eq_vals, "equality") if eq_vals is not None else None
        le_gaps = self.eval_gaps(le_vals, "inequality") if le_vals is not None else None

        # Lagrangian
        if self.is_constrained:
            constraints = self.optimizers().optimizer.weighted_constraint(eq_gaps, le_gaps) #JC ToDo
            lagrangian = loss + sum(constraints)
        else:
            lagrangian = loss # there is no lagrangian

        self.log_paths(lagrangian, loss, eq_vals, le_vals, batch)

    def log_paths(self, aug_lag, loss, eq_vals, le_vals, batch):
        """
        Logs various values relevant to the torch_constrained framework:
        the values of the main objective, constrained objectives, dual
        variables and Lagrangian.

        Args:
            aug_lag (float): the Lagrangian of the optimization problem-
            loss (float): main objective of the constrained formulation
            le_vals (1D tensor): the evaluated values of objectives associated
                with less or equal constraints.
            eq_vals (1D tensor): the evaluated values of objectives associated
                with equality constraints.
            batch (4D tensor): a data batch for metric evaluation.
        """
        if self.metrics is not None:
            self.log_metrics(batch)

        self.log_dict({"Lagrangian": aug_lag}, prog_bar=True)
        self.log_dict({"ERM": loss}, prog_bar=True)

        if self.log_constraints:
            if self.is_constrained:
                nus = self.optimizers().optimizer.equality_multipliers #JC ToDo
                if len(nus)>0: self.log_dict(multipliers2dict(nus, "nu"))
                lambdas = self.optimizers().optimizer.inequality_multipliers # JC ToDo
                if len(lambdas)>0: self.log_dict(multipliers2dict(lambdas, "lambda"))
            if eq_vals is not None:
                self.log_dict(tensor2dict(eq_vals, self.eq_names), prog_bar=True, reduce_fx=reduce)
            if le_vals is not None:
                self.log_dict(tensor2dict(le_vals, self.le_names), prog_bar=True, reduce_fx=reduce)

    def get_const_levels(self, loader, loss_type, loss_idx, epochs,
                        optimizer_class, optimizer_kwargs):
        """
        This function trains on one specific objective only, in order
        to provide a principled estimate for its bound.

        Args:
            loader (Dataloader): dataloader to use during training.
            loss_type (str): one of "le" or "eq", indicating if the objective
                is from the inequality or the equality constrained ones.
            loss_idx (scalar int): index of the objective in the tensor
                returned by eval_losses.
            epochs (scalar int): number of epochs to train.
            optimizer_class (torch Optimizer): optimizer class.
            optimizer_kwargs (dict): keyword arguments for the optimizer constructor.

        Returns:
            float: the achieved value of the objective after training.
        """

        # Imbalanced: Loader has very few samples which are used for this
        # procedure. Should reformulate.

        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        for e in range(epochs):
            rl = 0
            for batch in loader:

                optimizer.zero_grad()

                _, eq_vals, le_vals = self.eval_losses(batch)
                # Filter the desired loss
                this_loss = le_vals if loss_type == "le" else eq_vals
                this_loss = this_loss[loss_idx]

                if this_loss.requires_grad == False:
                    continue # the loss was not evaluated for lack of samples

                this_loss.backward()
                optimizer.step()
                rl += this_loss.item()

            rl = rl / len(loader) # normalize

        print("""The model has been pre-trained on a particular loss.
            Consider re-initializing the model for future training""")

        return rl

    def forward(self, input):
        """
        What the model does with a batch. This is application specific and
        hence must be implemented by the user.

        Args:
            input (tensor): inpuit data for the model.
        """
        raise NotImplementedError

    def eval_losses(self, batch):
        """
        Evaluates the different objectives for a given batch, according
        to the state of the model. This is application specific and
        hence must be implemented by the user.
        Args:
            batch (tensor or structure of tensors): a batch.
        """
        raise NotImplementedError

    def log_metrics(self, batch):
        """
        Logs some desired performance metrics. This is application specific
        and must be implemented by the user. This function is optional.
        Args:
            batch (tensor or structure of tensors): a batch.
        """
        raise NotImplementedError