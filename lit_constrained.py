import utils
import torch_constrained

from functools import partial
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR

class LitConstrained(pl.LightningModule):
    """
    This class implements a pytorch lightning module with the possibility 
    of training under a torch_constrained framework. 
    """
    def __init__(self, le_levels=None, eq_levels=None, 
                model_lr=1e-3, dual_lr=0., gamma=1, 
                log_constraints=False, metrics = False):
        """
        Args:
            le_levels (float list, optional): Levels for less or equal 
                constraints on the different objectives. Defaults to None,
                where no le constraint is considered.
            eq_levels (float list, optional): Levels for equality 
                constraints on the different objectives. Defaults to None,
                where no le constraint is considered.
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
        self.model_lr = model_lr
        self.dual_lr = dual_lr
        #self.le_damp = le_damp
        #self.eq_damp = le_damp # Should we join both damps?
        
        # Initialize objects for inequality (Less than or Equal) constraints
        if le_levels is not None: self.le_levels = torch.tensor(le_levels)
        else: self.le_levels = None
        
        # Initialize objects for Equality constraints
        if eq_levels is not None: self.eq_levels = torch.tensor(eq_levels)
        else: self.eq_levels = None
        
        # Is this a constrained optimization problem?
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
                torch_constrained.ExtraAdagrad, # let the user decide
                torch_constrained.ExtraSGD, 
                lr_x=self.model_lr,
                lr_y=self.dual_lr,
                primal_parameters=list(self.parameters()),
            )
        else: 
            # it is a scalar optimization problem
            optimizer = torch.optim.Adagrad(
                self.parameters(), 
                lr=self.model_lr
                )
        
        # Scheduler - mostly hard-coded
        #self.scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        
        return optimizer
    
    def eval_gaps(self, le_vals, eq_vals):
        """
        Calculate the gap between the values of the constrained objectives 
        and their respective levels. 
        
        Args:
            le_vals (1D tensor): the evaluated values of objectives associated
                with less or equal constraints.
            eq_vals (1D tensor): the evaluated values of objectives associated
                with equality constraints.
        Returns:
            1D tensors: feasibility gaps of the constraints
        """
        
        if self.le_levels is not None:
            le_gaps = le_vals - self.le_levels.to(self.device) 
            le_gaps = [le_gaps.reshape(1, -1), ] # format for optimizer
        else:
            le_gaps = None
        
        if self.eq_levels is not None:
            eq_gaps = eq_vals - self.eq_levels.to(self.device) 
            eq_gaps = [eq_gaps.reshape(1, -1), ] # format for optimizer
        else:
            eq_gaps = None
        
        return le_gaps, eq_gaps
    
    def training_step(self, batch, batch_idx):
        """
        Implements a training step for the model.
        
        Args:
            batch (tensor or structure of tensors): a training batch.
        """
        def closure():
            """
            Evaluates the different objectives on the current state of the 
            model and batch. Feasibility gaps are estimated for constrained
            objectives. Depends on user-defined function "eval_losses".
            
            Returns:
                tensors: the main objective, the objectives constrained with
                inequalities and those constrained with equalities.
            """
            loss, le_vals, eq_vals = self.eval_losses(batch)
            le_gaps, eq_gaps = self.eval_gaps(le_vals, eq_vals)
            return loss, le_gaps, eq_gaps
        
        if self.is_constrained:
            # torch_constrained handles most work. 
            self.optimizers().step(closure=closure)
        
        else:
            # Only primal params are optimized, on a vanilla Pytorch way.
            self.optimizers().zero_grad()
            loss, _, _ = closure() # constrained objectives are irrelevant
            loss.backward()
            self.optimizers().step()
        
        # self.scheduler.step()
    
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
        loss, le_vals, eq_vals = self.eval_losses(batch)
        le_gaps, eq_gaps = self.eval_gaps(le_vals, eq_vals)
        
        # Augmented lagrangian
        if self.is_constrained:
            constraints = self.optimizers().weighted_constraint(eq_gaps, le_gaps)
            aug_lag = loss + sum(constraints)
        else: 
            aug_lag = loss # there is no lagrangian
        
        # Log
        self.log_paths(aug_lag, loss, le_vals, eq_vals)
        if self.metrics is not None:
            self.log_metrics(batch)
        
        return aug_lag
    
    def log_paths(self, aug_lag, loss, le_vals, eq_vals):
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
        """
        log_fn = partial(self.log_dict, prog_bar=True, on_step=False, on_epoch=True)
        
        log_fn({"aug_lag": aug_lag})
        log_fn({"log_loss": utils.safe_log10(loss)})
        
        if self.log_constraints:
            if self.is_constrained:
                lambdas = self.optimizers().inequality_multipliers
                log_fn(utils.tensor2dict(lambdas, "lambda"))
            if le_vals is not None:
                log_fn(utils.tensor2dict(le_vals, "le"), 
                    reduce_fx = utils.reduce_fx)
            if self.is_constrained:
                nus = self.optimizers().equality_multipliers
                log_fn(utils.tensor2dict(nus, "nu"))
            if eq_vals is not None:
                log_fn(utils.tensor2dict(eq_vals, "eq"), 
                    reduce_fx = utils.reduce_fx)
    
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
                
                _, le_vals, eq_vals = self.eval_losses(batch)
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