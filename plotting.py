
import matplotlib.pyplot as plt

import re
import os
import importlib.util
import numpy as np

WANDB_AVAILABLE = importlib.util.find_spec("wandb") is not None
if WANDB_AVAILABLE:
    import wandb
else:
    print("wandb is not installed. Not used here.")


COLORS = ['#D81B60', '#1E88E5', '#e69f00','#f0e441','#029e72', '#FFC107', '#004D40', '#57b4e8', '#b4a7d6', '#b1bab2']


# Added by Myra Z.
def log_plot_predictions(y_true, y_hat, tensorboard_writer, use_wandb=False, output_dir='', split=''):
    plt.scatter(y_true, y_hat)
    plt.xlabel('true score')
    plt.ylabel('predictions')
    plt.title(f'Scatterplot of the true labels and the predictions of {split}.')
    if tensorboard_writer is not None:
        tensorboard_writer.add_figure('Scatter_Predictions', plt.gcf())
    if use_wandb:
        title = split + '_predictions'
        wandb.log({title: wandb.Image(plt)})
    if os.path.exists(output_dir) and output_dir != '':
        plt.savefig(output_dir + '/' + split + '_predictions.pdf', bbox_inches='tight')
    plt.close()


def log_plot_gradients(model, tensorboard_writer, use_wandb=False, output_dir=''):
    # plot gating of the gradients of the last state of the model
    # investigate gating
    grad_by_layer = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'gat' in name and not 'bias' in name:
            gate_grad = param.data.detach().squeeze().cpu().numpy()
            # get the number of the layer using regex
            layer = re.search('layer\.[0-9]{1,}', name)
            if layer is None: continue  # just go to next step and do nothing
            layer_num = re.search('[0-9]{1,}', layer.group(0)).group(0)
            # make dictionary for each layer with the different elements as touples:
            # (dict(str:(str, np.array)), e.g. {'1': ('layer1.lora', np.array([1,2]))}))
            if layer_num not in grad_by_layer:
                grad_by_layer[layer_num] = [(name, gate_grad)]
            else:
                grad_by_layer[layer_num].append((name, gate_grad))

    # create figure from the gating layers
    sub_fig_y = len(grad_by_layer.keys())
    sub_fig_x = max([len(grad_by_layer[key]) for key in grad_by_layer.keys()])

    fig, axs = plt.subplots(sub_fig_y, sub_fig_x, sharey=True, sharex=True, figsize=(sub_fig_x*5, sub_fig_y*5))

    for i, (key, val) in enumerate(sorted(grad_by_layer.items(), key=lambda x: int(x[0]))):
        key_num = int(key)
        for j, (name, grads) in enumerate(val):
            axs[i, j].hist(grads, orientation="horizontal")
            axs[i,j].set_title(name)
    
    if tensorboard_writer is not None:
        tensorboard_writer.add_figure('Gating gradients per layer', plt.gcf())
    if use_wandb:
        wandb.log({'Gating gradients per layer': wandb.Image(plt)})
    if os.path.exists(output_dir) and output_dir != '':
        plt.savefig(output_dir + '/' + 'gradient_gating_per_layer.pdf', bbox_inches='tight')

    plt.close()

    fig, axs = plt.subplots(sub_fig_y, 1, sharey=True, sharex=False, figsize=(sub_fig_x*5/2, sub_fig_y*5))

    for i, (key, val) in enumerate(sorted(grad_by_layer.items(), key=lambda x: int(x[0]))):
        key_num = int(key)
        gradients = [grads for (n, grads) in val]
        names = [n for (n, grads) in val]
        axs[i].boxplot(gradients, labels=names)
    
    if tensorboard_writer is not None:
        tensorboard_writer.add_figure('Gating gradients per layer', plt.gcf())
    if use_wandb:
        wandb.log({'Gating gradients per layer: Boxplot': wandb.Image(plt)})
    plt.close()


def log_plot_gates(model, tensorboard_writer, use_wandb=False, output_dir=''):
    gates = model.bert.gates
    if gates.empty:
        return
    encoder_layers = sorted(set(gates['encoder_layer']))
    # get eval and train of last epoch while in train
    last_train = gates[(gates['split'] == 'train_evaluation') & (gates['epoch'] == max(gates['epoch'])) & (gates['is_in_train'] == True)].reset_index()
    after_train_eval = gates[(gates['split'] == 'eval') & (gates['is_in_train'] == False)].reset_index()
    after_train_test = gates[(gates['split'] == 'test') & (gates['is_in_train'] == False)].reset_index()

    show_plot_crit = lambda key: len(gate_per_set[key]) > 0 # criterion to not show the plot for the data set, here: if dataset not used / df is empty
    gate_per_set = {'train':last_train, 'eval':after_train_eval, 'test':after_train_test}
    count_data_available = sum([1 for key in gate_per_set.keys() if show_plot_crit(key)])

    for layer in encoder_layers:
        fig, axs = plt.subplots(count_data_available, sharey=True, sharex=False, constrained_layout=True)
        idx = 0
        for key in gate_per_set.keys():
            columns = ['gate_prefix', 'gate_lora_value', 'gate_lora_query', 'gate_adapters']
            if show_plot_crit(key):
                #print(gate_per_set[key]) 
                dataset = gate_per_set[key]
                dataset = dataset[dataset['encoder_layer'] == layer].reset_index()
                dataset.dropna(axis=1, inplace=True)
               
                if count_data_available == 1:
                    for i, col in enumerate(columns):
                        if col not in dataset.columns:
                            continue
                        axs.plot(dataset[col].to_numpy(), label=col[5:], c=COLORS[i])
                    axs.set_ylabel('gating value')
                    axs.set_title(f'{key} data set')
                else:
                    for i, col in enumerate(columns):
                        if col not in dataset.columns:
                            continue
                        axs[idx].plot(dataset[col].to_numpy(), label=col[5:], c=COLORS[i])
                    #axs[idx].plot(dataset[['gate_prefix', 'gate_lora_value', 'gate_lora_query', 'gate_adapters']], label=['gate_prefix', 'gate_lora_value', 'gate_lora_query', 'gate_adapters'])
                    axs[idx].set_ylabel('gating value')
                    axs[idx].set_title(f'{key} data set')
                idx += 1

        title = f'{key}/gating/layer{int(layer) + 1}'
        fig.suptitle(title)
        plt.legend()
        plt.xlabel('data sample')
        plt.ylim(0,1)
        if tensorboard_writer is not None:
            tensorboard_writer.add_figure(title, plt.gcf())
        if use_wandb:
            wandb.log({title: wandb.Image(plt)})
        if os.path.exists(output_dir) and output_dir != '':
            plt.savefig(output_dir + '/' + title.replace('/', '_') + '.pdf', bbox_inches='tight')
        plt.close()


def log_plot_gates_per_layer(model, tensorboard_writer, use_wandb, output_dir=''):
    gates = model.bert.gates
    if gates.empty:
        return
    encoder_layers = sorted(set(gates['encoder_layer']))
    # get eval and train of last epoch while in train
    last_train = gates[(gates['split'] == 'train') & (gates['epoch'] == max(gates['epoch'])) & (gates['is_in_train'] == True)].reset_index()
    after_train_eval = gates[(gates['split'] == 'eval') & (gates['is_in_train'] == False)].reset_index()
    after_train_test = gates[(gates['split'] == 'test') & (gates['is_in_train'] == False)].reset_index()

    show_plot_crit = lambda key: len(gate_per_set[key]) > 0 # criterion to not show the plot for the data set, here: if dataset not used / df is empty
    gate_per_set = {'train':last_train, 'eval':after_train_eval, 'test':after_train_test}
    count_data_available = sum([1 for key in gate_per_set.keys() if show_plot_crit(key)])

    idx = 0
    for key in gate_per_set.keys():
        if show_plot_crit(key):
            fig, axs = plt.subplots()
            dataset = gate_per_set[key]
            #dataset = dataset[dataset['encoder_layer'] == layer].reset_index()

            gating_cols = [col for col in dataset.columns if 'gate' in col]
            grouped_mean = dataset.groupby(['encoder_layer']).agg({col: 'mean' for col in gating_cols})
            grouped_std = dataset.groupby(['encoder_layer']).agg({col: 'std' for col in gating_cols})

            bar_width = 2
            width = bar_width * (len(gating_cols)) + bar_width
            x = grouped_mean.index.to_numpy() * width
            this_colors = COLORS
            if len(this_colors) < len(gating_cols):  # if not enough colors in the list
                this_colors = [COLORS[i] if i < len(COLORS) else '#000000' for i in range(len(gating_cols))]

            fig, axs = plt.subplots()
            fig.set_figwidth(len(grouped_mean))
            for idx, col in enumerate(gating_cols):
                y_pos = x + idx * bar_width
                color_i = this_colors[idx]
                label_i = col[5:].replace('-', ' ').replace('_', ' ')
                axs.bar(x=y_pos, yerr=grouped_std[col], height=grouped_mean[col], width=bar_width, label=label_i, color=color_i)#, 'gate_lora_value', 'gate_lora_query', 'gate_adapters']], label=['gate_prefix', 'gate_lora_value', 'gate_lora_query', 'gate_adapters'])
            
            axs.set_xlabel('Encoder Layer')
            axs.set_xticklabels(grouped_mean.index.to_numpy())
            axs.set_xticks(x + ((len(gating_cols)-1) * bar_width)/2)
            axs.set_title(f'{key} data set')
            axs.set_xlim(x[0]-bar_width/2, x[-1] + ((len(gating_cols)-1) * bar_width) + bar_width/2)
            axs.set_ylim(0,1)
            axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            fig.suptitle('Mean Gating Values for all Encoder Layers')
            fig.tight_layout()
            title = f'{key}/gating_layers'
            plt.xlabel('Mean gating value')
            if tensorboard_writer is not None:
                tensorboard_writer.add_figure(title, plt.gcf())
            if use_wandb:
                wandb.log({title: wandb.Image(plt)})
            if os.path.exists(output_dir) and output_dir != '':
                plt.savefig(output_dir + '/' + title.replace('/', '_') + '.pdf', bbox_inches='tight')
            plt.close()


def log_plot_gates_per_epoch(model, tensorboard_writer=None, use_wandb=False, output_dir=''):
    gates = model.bert.gates
    if gates.empty:
        return
    encoder_layers = sorted(set(gates['encoder_layer']))

    last_train = gates[(gates['split'] == 'train') & (gates['is_in_train'] == True)].reset_index()
    after_train_eval = gates[(gates['split'] == 'eval') & (gates['is_in_train'] == True)].reset_index()
    after_train_test = gates[(gates['split'] == 'test') & (gates['is_in_train'] == False)].reset_index()

    show_plot_crit = lambda key: len(gate_per_set[key]) > 0 if key in gate_per_set.keys() else False # criterion to not show the plot for the data set, here: if dataset not used / df is empty
    gate_per_set = {'train':last_train, 'eval':after_train_eval, 'test':after_train_test}
    
    idx = 0
    for key in gate_per_set.keys():
        if show_plot_crit(key):
            dataset = gate_per_set[key]
            dataset.dropna(axis=1, inplace=True)
            #available_columns = [c for c in columns if c in dataset.columns]
            #if len(available_columns) < 2:  # No column available -> no plot
            #    continue
            gating_cols = [col for col in dataset.columns if 'gate' in col]
            grouped_mean = dataset.groupby(['encoder_layer', 'epoch']).agg({col: 'mean' for col in gating_cols})
            grouped_std = dataset.groupby(['encoder_layer', 'epoch']).agg({col: 'std' for col in gating_cols})
            #grouped_mean = dataset.groupby(['encoder_layer', 'epoch']).agg({c: 'mean' for c in available_columns})
            #grouped_std = dataset.groupby(['encoder_layer', 'epoch']).agg({c: 'std' for c in available_columns})
            for layer in encoder_layers:
                layer_mean = grouped_mean.loc[layer]
                layer_std = grouped_std.loc[layer]
                fig, axs = plt.subplots()
                bar_width = 2
                width = bar_width*4 + bar_width*1.5
                
                for i, col in enumerate(gating_cols):
                    if col not in gating_cols:
                        continue

                    label_i = col[5:].replace('-', ' ').replace('_', ' ')
                    x = np.array(range(len(layer_mean[col].to_numpy())))
                    axs.plot(x, layer_mean[col].to_numpy(), c=COLORS[i], label=label_i)
                    axs.fill_between(x, layer_mean[col].to_numpy() + layer_std[col].to_numpy(), layer_mean[col].to_numpy() - layer_std[col].to_numpy(), color=COLORS[i], alpha=0.5)
                if len(x) <= 1:
                    plt.close()
                    continue
                axs.set_xlabel('epochs')
                axs.set_ylabel('gating value')
                axs.set_ylim(0,1)
                axs.set_xticks(x)
                axs.set_xticklabels([str(item + 1) for item in x])
                axs.set_title(f'Layer {layer +1}: Mean of the Gating Values with Std \n {key} data')
                axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                idx += 1
                fig.tight_layout()
                
                plt.show()
                title = f'{key}/gating_plot/layer{layer + 1}'
                if tensorboard_writer is not None:
                    tensorboard_writer.add_figure(title, plt.gcf())
                if use_wandb:
                    wandb.log({title: wandb.Image(plt)})
                if os.path.exists(output_dir) and output_dir != '':
                    plt.savefig(output_dir + '/' + title.replace('/', '_') + '.pdf', bbox_inches='tight')
                plt.close()
