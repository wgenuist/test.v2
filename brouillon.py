from data_nilm import load_nilm_mapping
from data_nilm_import import load_nilm_mapping

from utilities import normalize_per_series
from neural_model import seq2point_reduced
from utilities import train, ts2windows, unscalerise, nilm_f1score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

from data_format import window_format_val, data_event_format
from nilmtk.electric import get_activations

seqlen_washing_machine = 102  # 171
seqlen_kettle = 85
seqlen_dish_washer = 171  # 106

sample_s = 7
seqlens = {'washing machine': seqlen_washing_machine, 'kettle': seqlen_kettle, 'dish washer': seqlen_dish_washer}
batchsz = 512
epochs = 20
patience = 6
APPLIANCES = ['kettle', 'dish washer', 'washing machine', ]
ALL_APPLIANCES = ['mains'].append(APPLIANCES)
mode = 'train'
folder_location = '/Users/wilfriedgenuist/PycharmProjects/desagregation/project/MODELS_window_size_corrected/'

import pandas as pd


def initialisation():
    app_out_appliances = {}
    home_dict = load_nilm_mapping(sample_s=sample_s)
    normalized_all, scalers = normalize_per_series(home_dict)

    test, normalized = {}, {}  # extraction de la maison test
    for i, elt in enumerate(normalized_all.keys()):
        if i == len(normalized_all.keys()) - 1:
            test[elt] = normalized_all[elt]
            break
        normalized[elt] = normalized_all[elt]

    for i, app_name in enumerate(APPLIANCES):
        app_in = {k: v['mains'] for k, v in normalized.items()}
        app_out = {k: v[app_name] for k, v in normalized.items()}
        app_out_appliances[app_name] = app_out

    data = {'app_in': app_in, 'app_out': app_out, 'app_out_appliances': app_out_appliances, 'home_dict': home_dict,
            'scalers': scalers, 'test': test}
    train_kwargs = {'batchsz': batchsz, 'epochs': epochs, 'patience': patience, }
    return train_kwargs, data


train_kwargs, data = initialisation()
data['APPLIANCES'] = APPLIANCES
maes, f1s, m = np.zeros([3]), np.zeros([3]), 0

for appliance in APPLIANCES:
    assert mode in ['test', 'train']  # mode
    seqlen = seqlens[appliance]
    if mode == 'train':
        model = seq2point_reduced(seqlen)
    if mode == 'test':
        model = keras.models.load_model(folder_location + appliance)

    ids = list(data['app_in'].keys())  # REFIT-2, REFIT-3, REFIT-5...

    # choice of appropriate data
    data['app_out'] = data['app_out_appliances'][appliance]
    data_format = window_format_val(data, seqlen)

    thr, padding = 5, seqlen // 2
    #for i in ids:
    #    data['app_in'][i], data['app_out'][i] = data_event_format(data['app_in'][i], data['app_out'][i], thr, padding, sample_s, 'perso')
    #    # aussi sur in_val/out_val ???

    ids = ['REFIT-3']

    for house in ids:  # Series of activations + app_in activations for each house
        activation = get_activations(data['app_out'][house], seqlen, seqlen // 2, seqlen // 2 * sample_s, thr)
        timestamps = [activation[k].index for k in range(len(activation))]
        data['app_in_activated'][house] = pd.concat([data['app_in'][house].loc[elt[0]: elt[-1]] for elt in timestamps])
        data['app_out_activated'][house] = pd.concat(activation)

    # train key arguments
    train_kwargs['in_data'] = np.vstack([data_format['app_in_activated'][i] for i in ids])
    train_kwargs['out_data'] = np.vstack([data_format['app_out_activated'][i] for i in ids])
    train_kwargs['in_val'] = np.vstack([data_format['in_val'][i] for i in ids])
    train_kwargs['out_val'] = np.vstack([data_format['out_val'][i] for i in ids])
    train_kwargs['model'] = model

    # save model
    if mode == 'train':
        history = train(**train_kwargs)
        model.save(folder_location + appliance)

    house_test = 'REFIT-3'

    # testing on an already known house
    scalers = data['scalers']

    output = model.predict(ts2windows(data['app_in'][house_test]['mains'].values, seqlen, padding='output'))

    # scaling back values
    unscaled_values = unscalerise(data, data[house_test][appliance], house_test, APPLIANCES, appliance, 'known',
                                  scalers)
    unscaled_values_output = unscalerise(data, output, house_test, APPLIANCES, appliance, 'known', scalers)

    # graph for visuals
    t, delta_t = 4700 + 250 + 2000, 7000

    plt.plot(np.array(unscaled_values['mains'])[t: t + delta_t], 'b')
    plt.plot(np.array(unscaled_values[appliance])[t: t + delta_t], 'g')
    #    plt.plot(np.array(unscaled_values_output[appliance])[t: t+delta_t], 'r')
    plt.legend(['Main consumption', 'Reference for ' + appliance, 'Algorithm output for ' + appliance])
    plt.title('Test for ' + house_test)
    plt.show()

    # score calculation
    f1s[m] = nilm_f1score(unscaled_values[appliance], unscaled_values_output[appliance])
    maes[m] = mean_absolute_error(unscaled_values[appliance], unscaled_values_output[appliance])
    m = m + 1

print('---end---')


########################################################################################################################
# choice of appropriate data
data['app_out'] = data['app_out_appliances'][appliance]
data_format = window_format_val(data, seqlen)

ids = list(data['app_in'].keys())  # REFIT-2, REFIT-3, REFIT-5...

thr_act = 5
for house in ids:  # Series of activations + app_in activations for each house
    activation = get_activations(data['app_out'][house], seqlen, seqlen // 2, seqlen // 2 * sample_s, thr_act)
    timestamps = [activation[k].index for k in range(len(activation))]
    data['app_in_activated'][house] = pd.concat([data['app_in'][house].loc[elt[0]: elt[-1]] for elt in timestamps])
    data['app_out_activated'][house] = pd.concat(activation)

# train key arguments
train_kwargs['in_data'] = np.vstack([data_format['app_in'][i] for i in ids])
train_kwargs['out_data'] = np.vstack([data_format['app_out'][i] for i in ids])
train_kwargs['in_val'] = np.vstack([data_format['in_val'][i] for i in ids])
train_kwargs['out_val'] = np.vstack([data_format['out_val'][i] for i in ids])
train_kwargs['model'] = model
