import matplotlib.pyplot as plt
import nilmtk
import pandas as pd

plt.style.use("tableau-colorblind10")
nilmtk.Appliance.allow_synonyms = False

APPLIANCES = ['kettle', 'dish washer', 'washing machine', ]
DATASETS = {

    "REFIT": {
        "path": "/Users/wilfriedgenuist/PycharmProjects/desagregation/datasets/REFIT/REFIT.h5",
        "buildings": {
            2: {"start_time": "2014-03-07", "end_time": "2014-03-14"},
            3: {"start_time": "2014-09-08", "end_time": "2014-09-15"},
            5: {"start_time": "2013-11-15", "end_time": "2013-11-22"},
            6: {"start_time": "2014-11-12", "end_time": "2014-11-19"},
            7: {"start_time": "2014-05-07", "end_time": "2014-05-14"},
#            9: { "start_time": "2014-05-03", "end_time": "2014-05-10" },
#            13: { "start_time": "2014-03-08", "end_time": "2014-03-15" },
#            19: { "start_time": "2014-06-01", "end_time": "2014-06-08" },
        }
    },
}


def clip_values(series, pmin, pmax):
    series[series <= pmin] = pmin
    series[series >= pmax] = pmax


def load_dataset(sample_s, appliances, path, buildings={}, physical=('power','active'), pmin=0, pmax=12000):
    load_kwargs = {
        'physical_quantity': physical[0],
        'ac_type': physical[1],
        'sample_period': sample_s,
    }
    ds = nilmtk.DataSet(path)
    for b_id, tframe in buildings.items():
        print("Loading {}, building {}".format(ds.metadata['name'], b_id))
        ds.set_window(tframe["start_time"], tframe["end_time"])
        elecmeter = ds.buildings[b_id].elec
        maingen = elecmeter.mains().load(**load_kwargs)
        mains_data = next(maingen)
        clip_values(mains_data, pmin, pmax)
        mains_data.columns = ['mains', ]
        all_data = []
        for app_name in appliances:
            appgen = elecmeter[app_name].load(**load_kwargs)
            app_data = next(appgen)
            clip_values(app_data, pmin, pmax)
            all_data.append(app_data)

        all_data = pd.concat(all_data, axis=1)
        all_data.columns = appliances
        idx = mains_data.index.intersection(all_data.index)
        all_data = all_data.loc[idx].copy().fillna(0)
        mains_data = mains_data.loc[idx].copy().fillna(0)
        aggregate = pd.DataFrame(all_data.sum(axis=1), columns=['mains',])
        mains_data[mains_data < aggregate] = aggregate[mains_data < aggregate]
        yield b_id, mains_data.join(all_data)
    ds.store.close()


def load_nilm_mapping(sample_s=7, appliances=APPLIANCES, datasets=DATASETS):
    all_data = {}
    for dsname, dsdict in datasets.items():
        datagen = load_dataset(sample_s, appliances, dsdict['path'], dsdict['buildings'])
        for b_id, b_data in datagen:
            all_data['{}-{}'.format(dsname, b_id)] = b_data

    return all_data


if __name__ == '__main__':
    total_rows = 0
    for title, data in load_nilm_mapping().items():
        print(data.shape)
        print(data.max())
        total_rows += data.shape[0]
        data.plot()
        plt.title(title)
        plt.show()

    print("Total number of points per channel: {}".format(total_rows))
    print("Total number of points: {}".format(total_rows * data.shape[1]))




