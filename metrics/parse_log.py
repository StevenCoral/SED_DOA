import os
import pickle

trial_type = 'single'  # single or series
loss_type = 'DOA'  # 'SED' or 'DOA'

training_params = list()
loss_results = list()

dest_folder = '/home/username/results_pickles/'

if trial_type == 'single':
    if loss_type != 'DOA' and loss_type != 'SED':
        print('Error: unknown loss type.')
        quit()

    log_file = 'doa_0000.log'
    with open(os.path.join(dest_folder, log_file), 'r') as file:
        content = file.read()

    hyper_results = content.split('Training mode')  # 0 is hyperparams, 1 is results
    train_key = 'Train ' + loss_type + ' loss'
    valid_key = 'Valid ' + loss_type + ' loss'
    results_dict = {train_key: None, valid_key: None}
    per_evaluation = hyper_results[1].split('Train SELD loss')  # Split results per evaluation
    train_list = list()
    valid_list = list()
    for curr_valid in per_evaluation[1:]:
        items = curr_valid.split(', ')  # Separate the loss values in the log
        # Populate lists only with the desired values:
        for curr_item in items:
            if train_key in curr_item:
                train_list.append(float(curr_item[-5:]))
            if valid_key in curr_item:
                valid_list.append(float(curr_item[-5:]))

    # Save in dictionary and main list:
    results_dict[train_key] = train_list
    results_dict[valid_key] = valid_list
    loss_results.append(results_dict)

    with open(os.path.join(dest_folder, 'losses_' + loss_type + '.pickle'), 'wb') as handle:
        pickle.dump(loss_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif trial_type == 'series':

    if loss_type == 'SED':
        log_file = '0001.log'  # For SED, we need to copy output from the terminal window into a single file.
        with open(os.path.join(dest_folder, log_file), 'r') as file:
            content = file.read()
        per_training = content.split('lr: ')
        for curr_training in per_training[1:]:
            hyper_results = curr_training.split('Training mode')  # 0 is hyperparams, 1 is results
            # Create a dictionary of hyperparams for every train ##########
            hyperparams_dict = dict()
            hyperparams_list = hyper_results[0].split('\n')[0].split(', ')
            hyperparams_dict['lr'] = float(hyperparams_list[0])
            for curr_param in hyperparams_list[1:]:
                key_value = curr_param.split(': ')  # 0 is key, 1 is value
                # Enter values into the dictionary, convert only nheads to int:
                hyperparams_dict[key_value[0]] = int(key_value[1]) if key_value[0] == 'nhead' else key_value[1]
            training_params.append(hyperparams_dict)

            # Create a dictionary of loss results for every train ##########
            train_key = 'Train ' + loss_type + ' loss'
            valid_key = 'Valid ' + loss_type + ' loss'
            results_dict = {train_key: None, valid_key: None}
            per_evaluation = hyper_results[1].split('Train SELD loss')  # Split results per evaluation
            train_list = list()
            valid_list = list()
            for curr_valid in per_evaluation[1:]:
                items = curr_valid.split(', ')  # Separate the loss values in the log
                # Populate lists only with the desired values:
                for curr_item in items:
                    if train_key in curr_item:
                        train_list.append(float(curr_item[-5:]))
                    if valid_key in curr_item:
                        valid_list.append(float(curr_item[-5:]))

            # Save in dictionary and main list:
            results_dict[train_key] = train_list
            results_dict[valid_key] = valid_list
            loss_results.append(results_dict)

        # Save files params to disk:
        with open(os.path.join(dest_folder, 'hyperparams_' + loss_type + '.pickle'), 'wb') as handle:
            pickle.dump(training_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(dest_folder, 'losses_' + loss_type + '.pickle'), 'wb') as handle:
            pickle.dump(loss_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif loss_type == 'DOA':
        file_list = os.listdir(dest_folder)
        for log_file in file_list:
            if log_file.endswith('.log'):
                with open(os.path.join(dest_folder, log_file), 'r') as file:
                    content = file.read()

                hyper_results = content.split('Training mode')  # 0 is hyperparams, 1 is results
                # We take the hyper-parameters from the SED file ###############
                # Create a dictionary of loss results for every train ##########
                train_key = 'Train ' + loss_type + ' loss'
                valid_key = 'Valid ' + loss_type + ' loss'
                results_dict = {train_key: None, valid_key: None}
                per_evaluation = hyper_results[1].split('Train SELD loss')  # Split results per evaluation
                train_list = list()
                valid_list = list()
                for curr_valid in per_evaluation[1:]:
                    items = curr_valid.split(', ')  # Separate the loss values in the log
                    # Populate lists only with the desired values:
                    for curr_item in items:
                        if train_key in curr_item:
                            train_list.append(float(curr_item[-5:]))
                        if valid_key in curr_item:
                            valid_list.append(float(curr_item[-5:]))

                # Save in dictionary and main list:
                results_dict[train_key] = train_list
                results_dict[valid_key] = valid_list
                loss_results.append(results_dict)

        with open(os.path.join(dest_folder, 'losses_' + loss_type + '.pickle'), 'wb') as handle:
            pickle.dump(loss_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print('Error: unknown loss type.')

else:
    print('Error: unknown trial type.')