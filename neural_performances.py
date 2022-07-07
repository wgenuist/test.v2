from data_nilm import load_nilm_mapping
from data_nilm_import import load_nilm_mapping
from data_format import window_format_val
from nilmtk.electric import get_activations
from utilities import normalize_per_series, barchart
from neural_model import seq2point_reduced, network_seq2point
from utilities import train, ts2windows, unscalerise, nilm_f1score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

seqlen_washing_machine = 102  # 171
seqlen_kettle = 85
seqlen_dish_washer = 171  # 106

# nilmtk.electric.get_activations
# 50 x samples_s, on rajoute en border la moitié de l'event

# objectf avoir un settup rapide et représentatif (taille fenêtre (20 min max), passer à deux semaines de données)
# augmenter le nombre de maisons (tenter REFIT-3 comme maison inconnue)
# faire une semaine, puis une semaine en ayant extrait les events

sample_s = 7
seqlens = {'washing machine': seqlen_washing_machine, 'kettle': seqlen_kettle, 'dish washer': seqlen_dish_washer}
batchsz = 512
epochs = 5
patience = 6
APPLIANCES = ['kettle', 'dish washer', 'washing machine', ]
ALL_APPLIANCES = ['mains'].append(APPLIANCES)
mode = 'train'  # train or test
save = 'ON'  # ON for saving graph
experience_title = 'full_model+trained_on_activations'  # no "/"
folder_location = '/Users/wilfriedgenuist/PycharmProjects/desagregation/project/MODELS/'  # neural model
graph_location = '/Users/wilfriedgenuist/PycharmProjects/desagregation/project/output_results/'


# dataset refit au complet (1 semaine), quels sont les appareils dispo par maison
# faire un entrainement par appareils => metrics par appareils
# liste différente de maisons en fonction des appliances
# charger toutes les donnnées en memoire (1 semaine), on change appliances (dict aves le nom de l'appareil et les clé des maisons qui contiennent l'appareil)
#
# avancer rapport

def initialisation():
    app_out_appliances = {}
    home_dict = load_nilm_mapping(sample_s=sample_s)
    normalized_all, scalers = normalize_per_series(home_dict)

    test, normalized = {}, {}  # extraction de la maison test
    for i, elt in enumerate(normalized_all.keys()):
        if i == len(normalized_all.keys())-1:
            test[elt] = normalized_all[elt]
            break
        normalized[elt] = normalized_all[elt]

    for i, app_name in enumerate(APPLIANCES):
        app_in = {k: v['mains'] for k, v in normalized.items()}
        app_out = {k: v[app_name] for k, v in normalized.items()}
        app_out_appliances[app_name] = app_out

    data = {'app_in': app_in, 'app_out': app_out, 'app_out_appliances': app_out_appliances, 'home_dict': home_dict,
            'scalers': scalers, 'test': test, }
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
#        model = network_seq2point(seqlen)
    if mode == 'test':
        model = keras.models.load_model(folder_location + appliance)

    # choice of appropriate data
    data['app_out'] = data['app_out_appliances'][appliance]
    data_format = window_format_val(data, seqlen)

    ids = list(data['app_in'].keys())  # REFIT-2, REFIT-3, REFIT-5...

    thr_act = 5
    data['app_in_activated'], data['app_out_activated'] = {}, {}
    data['app_in_val_activated'], data['app_out_val_activated'] = {}, {}
    for house in ids:  # Series of activations + app_in activations for each house
        #dat = data['app_out'][house][0:80*len(data['app_out'][house])//100+1]  # 80% of data is trained
        dat = data['app_out'][house]
        # seqlen, seqlen//2, seqlen * sample_s
        activation = get_activations(dat, seqlen * sample_s, 2 * sample_s, seqlen * 20, thr_act)
        timestamps = [activation[k].index for k in range(len(activation))]

        #data['app_in_activated'][house] = pd.concat([data['app_in'][house].loc[elt[0]: elt[-1]] for elt in timestamps])
        #data['app_out_activated'][house] = pd.concat(activation)

        A = pd.concat([data['app_in'][house].loc[elt[0]: elt[-1]] for elt in timestamps])
        B = pd.concat(activation)

        lA = len(A)
        lB = len(B)

        data['app_in_activated'][house] = A[0:80 * lA // 100+1]
        data['app_out_activated'][house] = B[0:80 * lB // 100+1]
        data['app_in_val_activated'][house] = A[80 * lA // 100+1:]
        data['app_out_val_activated'][house] = B[80 * lB // 100+1:]


    # train key arguments
    train_kwargs['in_data'] = np.vstack([ts2windows(data['app_in_activated'][house].values, seqlen) for house in ids])
    train_kwargs['out_data'] = np.vstack([ts2windows(data['app_out_activated'][house].values, seqlen) for house in ids])
    #train_kwargs['in_val'] = np.vstack([data_format['in_val'][i] for i in ids])
    #train_kwargs['out_val'] = np.vstack([data_format['out_val'][i] for i in ids])

    train_kwargs['in_val'] = np.vstack([ts2windows(data['app_in_val_activated'][house].values, seqlen) for house in ids])
    train_kwargs['out_val'] = np.vstack([ts2windows(data['app_out_val_activated'][house].values, seqlen) for house in ids])

    train_kwargs['model'] = model

    # save model
    if mode == 'train':
        history = train(**train_kwargs)
        model.save(folder_location + experience_title + '/' + appliance)

    # testing on an unknown house
    scalers = data['scalers']
    test = data['test']
    house_test = list(test.keys())[0]
    output = model.predict(ts2windows(test[house_test]['mains'].values, seqlen, padding='output'))

    # scaling back values
    unscaled_values = unscalerise(test, test[house_test][appliance], house_test, APPLIANCES, appliance, 'output',
                                  scalers)
    unscaled_values_output = unscalerise(test, output, house_test, APPLIANCES, appliance, 'output', scalers)

    # graph for visuals
    t, delta_t = 4700+250+2000, 3600

    plt.plot(np.array(unscaled_values['mains'])[t: t+delta_t], 'b')
    plt.plot(np.array(unscaled_values[appliance])[t: t + delta_t], 'g')
    plt.plot(np.array(unscaled_values_output[appliance])[t: t+delta_t], 'r')
    plt.legend(['Main consumption', 'Reference for ' + appliance, 'Algorithm output for ' + appliance])
    plt.title('Test for ' + house_test)

    if save == 'ON':
        plt.savefig(graph_location + experience_title + '/' + appliance + '.png')
    plt.show()

    # score calculation
    f1s[m] = nilm_f1score(unscaled_values[appliance], unscaled_values_output[appliance])
    maes[m] = mean_absolute_error(unscaled_values[appliance], unscaled_values_output[appliance])
    m = m+1

barchart(f1s, maes, APPLIANCES, graph_location, experience_title, save)

print('---end---')
