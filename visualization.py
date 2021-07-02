    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:13:15 2021

Used to plot the behaviour of various multi-objective optimization techniques

@author: juan
"""

import utils

from matplotlib import pyplot as plt

import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import plotly.express.colors as c
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"

##################################
# Toy 2d data
#################################

def plot_2d_predictions(X, y, title, alpha=0.8, colorbar=True):
    plt.figure(figsize=(4.5,4))
    plt.scatter(*X.T, c=y, s=70, edgecolors='k', alpha=alpha, 
                vmin=0, vmax=1, cmap=plt.get_cmap('bwr'))
    if colorbar:
        plt.colorbar()
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    
####################################3 
# Static optimization paths
##################################3
# Make a class to incorporate these plots


def add_const(fig,typ,level,horizontal=True,row=1,col=1):
        
    gen_line = fig.add_hline if horizontal else fig.add_vline
    const_type = " = " if typ=="eq" else " < " if typ=="le" else "??? "
    annotation_text="<b>objective</b>"+const_type+str(np.round(level.item(),2))
    
    fig = gen_line(
        level,
        annotation_text=annotation_text,
        annotation_font_size=8,
        row=row,
        col=col
        )
    
    return fig

def get_1v1_path(df,name,col,row,loss_keys,colour):
    
    if row==0 and col=="epoch":
        # This is added to add legend and colorbar only once
        showlegend=True
        colorbar=dict(title="Epochs")
    else:
        showlegend=False
        colorbar={}
    
    x = df["epoch"] if col=="epoch" else df[loss_keys[col]] 
    y = df[loss_keys[row]]
    
    trace = go.Scatter(
        name=name,
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            color=df["epoch"],
            colorscale=colour,
            colorbar=colorbar,
            reversescale=False,
            line=dict(color=colour[-1],width=1),
            size=5
            ),
        showlegend=showlegend,
        )
    
    return trace

def loss_plot(paths_dict, loss_keys=None, lam_keys=None, loss_names=None, lam_names = None,
                loss_levels=None, log_levels=True, title="Optimization path of losses", 
                colours=None, height=400, width=950):
    
    if loss_levels is not None and log_levels: 
        loss_levels = utils.safe_log10(torch.tensor(loss_levels))
    
    # Hard coded. Should not depend on plotly seq. At least move helper elsewere
    if colours is None: 
        colours = [c.sequential.Blues, c.sequential.Oranges, c.sequential.Greens, 
                   c.sequential.Purp, c.sequential.Reds]
        colours = [c[int(0.25*len(c)):] for c in colours]
        
    if loss_names is None: loss_names = loss_keys
                
    fig = make_subplots(rows=2, cols=len(loss_keys))
    
    for n_key,key in enumerate(loss_keys): 
    
        # Add loss vs epoch
        for i, (name, df) in enumerate(paths_dict.items()):
            trace = get_1v1_path(df,name,"epoch",n_key,loss_keys,colours[i])
            fig.add_trace(
                trace,
                row=1, 
                col=n_key+1 # render indexing starts at 1, not 0
                )
            
        # x and y labels
        fig.update_xaxes(title_text="Epoch", row=1, col=n_key+1)
        fig.update_yaxes(title_text=loss_names[n_key], row=1, col=n_key+1)    
        
        # Do not hard code le. Make code nicer
        if loss_levels is not None and not torch.isnan(loss_levels[n_key]):
            fig = add_const(fig,"le",loss_levels[n_key],horizontal=True,
                            row=1,col=n_key+1)

        if n_key > 0:
            # Add lam vs epoch
            for i, (name, df) in enumerate(paths_dict.items()):
                if name != "ERM":
                    trace = get_1v1_path(df,name,"epoch",n_key,lam_keys,colours[i])
                    fig.add_trace(
                        trace,
                        row=2, 
                        col=n_key+1 # render indexing starts at 1, not 0
                        )
                
            # x and y labels
            fig.update_xaxes(title_text="Epoch", row=2, col=n_key+1)
            fig.update_yaxes(title_text=lam_names[n_key], row=2, col=n_key+1)
        
    fig.update_layout(
        height=height, 
        width=width, 
        title_text=title,
        legend=dict(
            x=-0.25
            ),
        )
    
    return fig


def pairwise_plot(paths_dict, loss_keys=["loss","le_1","le_2"], loss_names=None,
                  loss_levels=None, log_levels=True, title="Optimization path of losses", 
                  colours=None, height=850, width=850):
    
    if loss_levels is not None and log_levels: 
        loss_levels = utils.safe_log10(torch.tensor(loss_levels))
    
    # Hard coded. Should generalize and not depend on color sequences
    if colours is None: 
        colours = [c.sequential.Blues, c.sequential.Oranges, c.sequential.Greens, 
                   c.sequential.Purp, c.sequential.Reds]
        colours = [c[int(0.25*len(c)):] for c in colours]
        
    if loss_names is None: loss_names = loss_keys
                
    grid_dim = len(loss_keys)
    fig = make_subplots(rows=grid_dim, cols=grid_dim)
    
    for row in range(grid_dim): 
    
        # Add loss vs epoch in diagonal
        for i, (name, df) in enumerate(paths_dict.items()):
            trace = get_1v1_path(df,name,"epoch",row,loss_keys,colours[i])
            fig.add_trace(
                trace,
                row=row+1, # render indexing starts at 1, not 0
                col=row+1 # this is the diagonal
                )
            
        # x and y labels
        fig.update_xaxes(title_text="Epoch", row=row+1, col=row+1)
        fig.update_yaxes(title_text=loss_names[row], row=row+1, col=row+1)    
        
        # Do not hard code le. Make code nicer
        if loss_levels is not None and not torch.isnan(loss_levels[row]):
            fig = add_const(fig,"le",loss_levels[row],horizontal=True,
                            row=row+1,col=row+1)
        
        for col in range(row+1,grid_dim): 

            for i, (name, df) in enumerate(paths_dict.items()):
                
                trace = get_1v1_path(df,name,col,row,loss_keys,colours[i])
                fig.add_trace(
                    trace,
                    row=row+1, 
                    col=col+1
                    )
                
            # Do not hard code le. Make code nicer
            if loss_levels is not None and not torch.isnan(loss_levels[row]):
                fig = add_const(fig,"le",loss_levels[row],horizontal=True,
                                row=row+1,col=col+1)
                
            if loss_levels is not None and not torch.isnan(loss_levels[col]):
                fig = add_const(fig,"le",loss_levels[col],horizontal=False,
                                row=row+1,col=col+1)
            
            # x and y labels
            fig.update_xaxes(title_text=loss_names[col], row=row+1, col=col+1)
            fig.update_yaxes(title_text=loss_names[row], row=row+1, col=col+1)                
                
    fig.update_layout(
        height=height, 
        width=width, 
        title_text=title,
        legend=dict(
            x=-0.25
            ),
        )
    """
    # Make colorbar a scale of greys where opacity denotes iteration
    fig.update_coloraxes(
        colorscale=c.sequential.Greys,
        reversescale=True,
        colorbar=dict(
            title="Epoch",
            ),
        )
    """
    return fig
    

def plot_opt_path():
    plt.scatter(x_vals, y_vals , s=10, c=range(len(df[x_axis].values)), alpha=0.3)
    plt.scatter(x_vals[0], y_vals[0], marker='s', c='black', s=100, alpha=0.5) # Start
    plt.scatter(x_vals[-1], y_vals[-1], marker='*', c='red', s=100, alpha=0.5) # End
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.colorbar()
    plt.axis("equal")


####################################3 
# Animated optimization paths
##################################3
# Make class to incorporate animations

def plot_trajectories(params,paths,f=None,g=None,xmin=None,xmax=None,
                    title="",xlab="",ylab="",do_pareto=True,to_annotate=None):    
    """
     Main function
    
    """    
    n_paths = params.shape[0]
    
    # Sample colours
    colours = px.colors.sequential.Turbo_r
    colours = [colours[int(i*len(colours)/(n_paths+1))] for i in range(n_paths+1)]
    
    
    annotations,lam_title = gen_annotations(to_annotate,params.columns) # training hyper_param annot
    buttons = gen_buttons() # play and pause

    # Frames
    trajectories,anim_frames = training_frames(params,paths,n_paths,
                                               colours,annotations,
                                               lam_title
                                       ) 
        
    if do_pareto is True:
        # Pareto front
        pareto = trace_pareto(f,g,xmin,xmax)
        trajectories = trajectories+[pareto]
        
    # Plot animation
    range_x=[0,max(paths["f"])]
    range_y=[0,max(paths["g"])]
    fig = go.Figure(data=trajectories,
                    layout=go.Layout(
                        title_text=title, 
                        hovermode="closest",
                        updatemenus=[buttons],
                        xaxis=dict(range=range_x, autorange=True),
                        yaxis=dict(range=range_y, autorange=True),
                        xaxis_title=xlab,
                        yaxis_title=ylab,
                        width=600,
                        height=600
        ),
    frames=anim_frames
    )
    
    fig.add_annotation(annotations)
    fig.add_annotation(lam_title)
    
    # Add hlines for the restriction
    if to_annotate is not None and "const_value" in to_annotate.keys():
        fig = add_rest_lines(fig, to_annotate["const_type"], to_annotate["const_value"])
        
    return fig
  
    
def training_frames(params,paths,n_paths,colours,annotations,lam_title):
    n_steps = paths[paths["number"]==0].count()[0]
    
    param_annot = go.layout.Annotation(annotations)
    lam_title = go.layout.Annotation(lam_title)
        
    by_path = [paths.loc["number"==path] for path in range(n_paths)]
    by_params = [params.iloc[path] for path in range(n_paths)]
    
    # Names for legend    
    names = [str(key)+": "+str(np.round(value,2))+"<br>" for p in by_params 
             for key,value in p.items()]
    names = ["".join(name) for name in names]
        
    # Full trajectories (enables labelling) (hidden when the animation starts)
    trajectories = [(
        go.Scatter(name=names[path],
                   x=by_path[path]["f"],
                   y=by_path[path]["g"],
                   marker=dict(color=colours[path+1])
                   )
        ) for path in range(n_paths)]
    
    frames = [
        go.Frame(
            data=[go.Scatter(x=by_path[path].iloc[max(0,step-10):step]["f"],
                             y=by_path[path].iloc[max(0,step-10):step]["g"],
                             mode="lines+markers", 
                             marker=dict(color=colours[path+1])
                            )
                  for path in range(n_paths)],
            layout=go.Layout(
                        annotations= [param_annot] + [go.layout.Annotation(
                            text=str(by_path[path].iloc[step]["lam"]),
                            align='left',
                            showarrow=False,
                            xref='paper',
                            yref='paper',
                            font=dict(color=colours[path+1]),
                            x=1.15,y=0.2 - 0.025 * path,
                            ) for path in range(n_paths)] + [lam_title]
                        ) 
            )
        for step in range(n_steps)
    ]

    return trajectories,frames



    
def plot_feas(f,g,xmin,xmax,
              title="",xlab="",ylab="",domain_str=""):
    """
    Scatter plot values of f vs g for a grid of points (x,y). Pareto-optimal points
    among the ones of the grid are highlited.
    
    Parameters
    ----------
    
    f: function
        a function of two variables (x,y)
    g: function
        a function of two variables (x,y)
    xmin: int list
        minimum values of the variable x
    xmax: int list
        maximum values of the variable x
    title: string, optional
        title for the plot
    xlab: string, optional
        identifier of function f
    ylab: string, optional
        identifier of function g
    
        
    """

    # Acheivable points
    f_x,g_x=acheivable_set(f,g,xmin,xmax)
    acheivable = go.Scatter(name="Acheivable Region",x=f_x,y=g_x)
    
    # Pareto front
    pareto = trace_pareto(f,g,xmin,xmax)

    fig = go.Figure([acheivable,pareto])

    fig.update_layout(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        legend_title_text=domain_str,
        width = 600,
        height = 600
    )
        
    return fig
    
def trace_pareto(f,g,xmin,xmax):

    # Acheivable set
    f_x,g_x=acheivable_set(f,g,xmin,xmax)
    costs = np.transpose(np.array([f_x,g_x]))
    costs_pareto = pareto(costs)
        
    # Plot 
    trace = go.Scatter(
        name="<b>Pareto Front</b>",
        x=costs_pareto[:,0],
        y=costs_pareto[:,1], 
        hoverinfo='skip',
        marker=dict(
            color='lightsalmon'
        )
    )
        
    return trace

def gen_buttons():
    b=dict(type="buttons",
         buttons=[
             dict(label="Play",
                   method="animate",
                   args=[None, {
                       "frame": {"duration": 25},
                       "fromcurrent": True, 
                       "transition": {"duration": 50,
                                      "easing": "quadratic-in-out"}}
                        ]
                  ),
              dict(label="Pause",
                   method="animate",
                   args=[[None], {
                       "frame": {"duration": 0},
                       "mode": "immediate",
                       "transition": {"duration": 0}}
                        ]
                  )
                ]
             )
    return b

def gen_annotations(to_annotate,lab_names):
    
    if to_annotate is None:
        return None
    
    text="<b>Hyper-params</b> <br>"
    
    for name,value in to_annotate.items():
                
        if not isinstance(value, str):
            value=str(np.round(value,2))
        
        if name != "const_value" and name not in lab_names:
            text += name + ": " + value + "<br>"
    
    a=dict(
        text=text,align='left',showarrow=False,
        xref='paper',yref='paper',x=1.3,y=0.5,
        bordercolor='white',borderwidth=1
    )
    
    # An annotation to Indicate lam variations
    lam_title = dict(text="Lambda:",align='left',showarrow=False,
                    xref='paper',yref='paper',x=1.12,y=0.24,
                    )
    return a, lam_title


def pareto(costs):
    """
    Find pareto-efficient points
    
    Parameters
    ----------
    
    costs: An (n_points, n_costs) array
        costs associated with various data points
    
    Returns
    -------
    
    pareto_set: array
        set of costs amongst the provided ones which are pareto optimal
        
    """
    # Aviod numerical uncertainties:
    costs = np.round(costs,5)
    
    original_costs=costs
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    
    pareto_set = original_costs[is_efficient,]
    
    return pareto_set
    
    
def acheivable_set(f,g,xmin=[0,0],xmax=[1,1]):
    """
    Produce the costs f(x) and g(x) for a grid of points of x
    
    Parameters
    ----------
    
    f: function
        a function of two variables (x,y)
    g: function
        a function of two variables (x,y)
    xmin: int list
        minimum value of the variables x
    xmax: int list
        maximum value of the variables x
    
    
    Returns
    -------
    
    f_x: list 
        values of f for a grid of points
    
    g_x: list 
        values of f for a grid of points
        
    """
    x_set = [jnp.linspace(lb,ub) for lb,ub in zip(xmin,xmax)] 
    grid = list(itertools.product(*x_set))
      
    f_x = [f(point) for point in grid]
    g_x = [g(point) for point in grid]
    
    return f_x,g_x
    

def lam_v_g(var_name,var_values,lam_paths,f_paths,g_paths,
            to_annotate,title=""):
    
    n_iter = len(lam_paths[0])
    names = ["Lambda              ","f","g"]
    colors = ["blue","red","yellow"]
    
    paths = list(zip(lam_paths,f_paths,g_paths))[0] # first one
    traces = [go.Scatter(name=name,
                         x=list(range(n_iter)),
                         y=path,
                         marker=dict(color=c)
                         ) 
              #for paths in zip(lam_paths,f_paths,g_paths)
              for name,c,path in zip(names,colors,paths)
              ]
    
    fig = go.Figure(data=traces,
                    layout=go.Layout(
                        title_text=title, 
                        hovermode="closest",
                        xaxis_title="Iteration",
                        yaxis_title="",
                        width=600,
                        height=600
        )
    )
    
    fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            xanchor = 'left',
            yanchor = 'top',
            showactive=True,
            buttons=list(
                [
                    dict(
                        label=var_name +": " + str(np.round(value,2)),
                        method="update",
                        args=[{"y": [lam_paths[i],f_paths[i],g_paths[i]]}],
                    )
                for i,value in enumerate(var_values)]
            ),
        )
        ]
    )
    
    
    annotations,_ = gen_annotations(to_annotate,var_name) # training hyper_param annot
    fig.add_annotation(annotations)    
    
    return fig
