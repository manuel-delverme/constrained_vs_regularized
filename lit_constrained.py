import utils
import torch_constrained

from functools import partial
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR

class LitConstrained(pl.LightningModule):

    def __init__(self, le_levels=None, eq_levels=None, model_lr=1e-3, 
                dual_lr=0., le_damp = 0., train_method='ERM', gamma=1, 
                log_constraints=False, metrics = False):
        
        super().__init__()
        
        # Auto optimization is 'disabled' so we can do manual optimizer steps
        self.automatic_optimization = False
        
        ## Constrained Optimization parameters
        self.train_method = train_method
        self.model_lr = model_lr
        self.dual_lr = dual_lr
        self.le_damp = le_damp
        self.eq_damp = le_damp # Should we join both damps?
        
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
        
        # Add optimizer stuff
        self.gamma = gamma # lr scheduling
    
    def configure_optimizers(self):
        
        if self.is_constrained:
            # Then its a constrained optimization problem
            optimizer = torch_constrained.ConstrainedOptimizer(
                torch_constrained.ExtraAdagrad, # let the user decide
                torch_constrained.ExtraSGD,
                lr_x=self.model_lr,
                lr_y=self.dual_lr,
                primal_parameters=list(self.model.parameters()),
            )
        else: 
            # it is a scalar optimization problem
            optimizer = torch.optim.Adagrad(
                self.model.parameters(), 
                lr=self.model_lr
                )
        
        # Scheduler - mostly hard-coded
        #self.scheduler = StepLR(optimizer, step_size=1, gamma=self.gamma)
        
        return optimizer
    
    def eval_gaps(self, le_vals, eq_vals):
        
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
        # add support for non constrained approaches
        def closure():
            loss, le_vals, eq_vals = self.eval_losses(batch)
            le_gaps, eq_gaps = self.eval_gaps(le_vals, eq_vals)
            return loss, le_gaps, eq_gaps
        
        if self.is_constrained:
            self.optimizers().step(closure=closure)
        
        else:
            # This is done inside the torch_constrained optimizer
            self.optimizers().zero_grad()
            loss, _, _ = closure()
            loss.backward()
            self.optimizers().step()
        
        # self.scheduler.step()
    
    def validation_step(self, batch, batch_idx):
        
        # Losses
        loss, le_vals, eq_vals = self.eval_losses(batch)
        le_gaps, eq_gaps = self.eval_gaps(le_vals, eq_vals)
        
        # Augmented lagrangian
        if self.is_constrained:
            constraints = self.optimizers().weighted_constraint(eq_gaps, le_gaps)
            aug_lag = loss + sum(constraints)
        else: 
            aug_lag = loss # there is no lagrangian
        
        self.log_paths(aug_lag, loss, le_vals, eq_vals)
        
        if self.metrics is not None:
            self.log_metrics(batch)
        
        return aug_lag
    
    def log_paths(self, aug_lag, loss, le_vals, eq_vals):
        # Add comments
        
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
    
    def get_const_levels(self, loader, loss_idx, loss_type, 
                        epochs, optimizer_class, optimizer_kwargs):
        
        # Currently the loader has very few samples which are used for this
        # training procedure. Should reformulate the loader
        
        # Configure optimizer
        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        
        for e in range(epochs):
            rl = 0
            for batch in loader:
                
                optimizer.zero_grad()
                
                _, le_vals, eq_vals = self.eval_losses(batch)
                # Filter the desired loss
                this_loss = le_vals if loss_type == "leq" else eq_vals
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
    
    def forward(self, batch):
        raise NotImplementedError
    
    def eval_losses(self, batch):
        raise NotImplementedError
    
    def log_metrics(self, batch):
        raise NotImplementedError
    
    """            
    def augmented_lagrangian(self, loss, le_gaps=None, eq_gaps=None):
        
        lambdas = self.optimizers().inequality_multipliers
        nus = self.optimizers().equality_multipliers
        
        if le_gaps is None:
            le_contrib = 0.
        else:
            le_gaps = le_gaps[0]
            
            # Avoid ALM computations when damp = 0
            le_gaps = torch.nan_to_num(le_gaps, nan=0.0) # nans dont contribute
            
            # ALM auxiliary term
            le_active = ((le_gaps >= 0) + (lambdas > 0)).float()
            aux_term = self.le_damp * le_active * le_gaps
            
            # Contrinbution to the lagrangian
            le_contrib = torch.dot(aux_term + lambdas, le_gaps)
        
        if eq_gaps is None:
            eq_contrib = 0.
        else:
            eq_gaps = eq_gaps[0]
            eq_gaps = torch.nan_to_num(eq_gaps, nan=0.0)
            eq_contrib = torch.dot(self.eq_damp * eq_gaps + nus, eq_gaps)
        
        return loss + le_contrib + eq_contrib
    """
