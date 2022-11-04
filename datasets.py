import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NCMAPSS:
    def __init__(self, path):
        self.train, self.test = self.load_data(path)

    def load_data(self, path):
        with h5py.File(path, 'r') as hdf:
            # Development set
            W_dev = np.array(hdf.get('W_dev'))  # W
            X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
            X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
            T_dev = np.array(hdf.get('T_dev'))  # T
            Y_dev = np.array(hdf.get('Y_dev'))  # RUL
            A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

            # Test set
            W_test = np.array(hdf.get('W_test'))  # W
            X_s_test = np.array(hdf.get('X_s_test'))  # X_s
            X_v_test = np.array(hdf.get('X_v_test'))  # X_v
            T_test = np.array(hdf.get('T_test'))  # T
            Y_test = np.array(hdf.get('Y_test'))  # RUL
            A_test = np.array(hdf.get('A_test'))  # Auxiliary

            # Varnams
            W_var = np.array(hdf.get('W_var'))
            X_s_var = np.array(hdf.get('X_s_var'))
            X_v_var = np.array(hdf.get('X_v_var'))
            T_var = np.array(hdf.get('T_var'))
            A_var = np.array(hdf.get('A_var'))

            # from np.array to list dtype U4/U5
            self.W_var = list(np.array(W_var, dtype='U20'))
            self.X_s_var = list(np.array(X_s_var, dtype='U20'))
            self.X_v_var = list(np.array(X_v_var, dtype='U20'))
            self.T_var = list(np.array(T_var, dtype='U20'))
            self.A_var = list(np.array(A_var, dtype='U20'))

        train = pd.concat([pd.DataFrame(W_dev, columns=self.W_var),
                           pd.DataFrame(X_s_dev, columns=self.X_s_var),
                           pd.DataFrame(X_v_dev, columns=self.X_v_var),
                           pd.DataFrame(T_dev, columns=self.T_var),
                           pd.DataFrame(A_dev, columns=self.A_var),
                           pd.DataFrame(Y_dev, columns=['RUL'])],
                          axis=1)

        test = pd.concat([pd.DataFrame(W_test, columns=self.W_var),
                          pd.DataFrame(X_s_test, columns=self.X_s_var),
                          pd.DataFrame(X_v_test, columns=self.X_v_var),
                          pd.DataFrame(T_test, columns=self.T_var),
                          pd.DataFrame(A_test, columns=self.A_var),
                          pd.DataFrame(Y_test, columns=['RUL'])],
                         axis=1)
        self.train_units = train['unit'].unique().tolist()
        self.test_units = test['unit'].unique().tolist()
        print('train_units', self.train_units)
        print('test_units', self.test_units)

        return train, test

    def keep_regime(self, health_state):
        '''
        Keep the normal degradation regime points if health_state == 1,
        keep the abnormal degradation regime if the health_state == 0.
        '''
        if health_state == 0:
            self.train = self.train[self.train['hs'] == 0.0]
            self.test = self.test[self.test['hs'] == 0.0]
        elif health_state == 1:
            self.train = self.train[self.train['hs'] == 0.0]
            self.test = self.test[self.test['hs'] == 0.0]
        return None

    def keep_train_units(self, units):
        self.train = self.train[self.train['unit'].isin(units)]
        self.train_units = self.train['unit'].unique().tolist()
        return None

    def keep_test_units(self, units):
        self.test = self.test[self.test['unit'].isin(units)]
        self.test_units = self.test['unit'].unique().tolist()
        return None

    def subsample_cycles(self, frac=0.1):
        self.train = self.__subsample_cycles(self.train, frac)
        self.test = self.__subsample_cycles(self.test, frac)
        return None

    def __subsample_cycles(self, data, frac):
        dfs = []
        for unit in data['unit'].unique():
            unit_df = data[data['unit'] == unit]
            cycles = unit_df['cycle'].unique()
            cycles = np.random.choice(cycles, size=int(frac * len(cycles)), replace=False)
            unit_df = unit_df[unit_df['cycle'].isin(cycles)]
            dfs.append(unit_df)
        result = pd.concat(dfs)
        result = result.reset_index(drop=True)
        return result

    def __shift_df(self, df, suffix='_k+1'):
        '''
        Add columns that are shifted one row back with a given suffix.
        '''

        unit_dfs = []

        for unit in df['unit'].unique():
            unit_df = df[df['unit'] == unit]
            for cycle in unit_df['cycle'].unique():
                cycle_df = unit_df[unit_df['cycle'] == cycle]
                shifted = cycle_df.shift(-1)
                merged = pd.concat([cycle_df, shifted.add_suffix(suffix)], axis=1)
                merged = merged.dropna()
                unit_dfs.append(merged)
        result = pd.concat(unit_dfs)
        result = result.reset_index(drop=True)
        return result

    def shift_data(self, suffix='_k+1'):
        self.shift_suffix = suffix
        self.train = self.__shift_df(self.train, suffix=suffix)
        self.test = self.__shift_df(self.test, suffix=suffix)
        return None

    def scale(self, columns):
        scaler = StandardScaler()
        self.train[columns] = scaler.fit_transform(self.train[columns])
        self.test[columns] = scaler.transform(self.test[columns])
        return scaler

    def get_data(self, columns):
        return self.train[columns], self.test[columns]

    def get_eval_data(self, unit, cycle):
        c1 = self.test['unit'] == unit
        c2 = self.test['cycle'] == cycle
        return self.test[c1 & c2]
