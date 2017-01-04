from __future__ import unicode_literals
from os.path import join as os_join ,\
                    exists as os_exists
from os import mkdir as os_mkdir
import sys
import timeit
from sys import exc_info
from traceback import format_exception

from pandas import read_csv, to_datetime, DataFrame, Series, \
                   date_range, to_numeric, read_pickle
from numpy import intersect1d, vectorize, logical_not, isnan, \
                  where, linspace, logical_or, full, \
                  linalg, matmul, any as np_any, ediff1d, \
                  round as np_round, repeat, tile, nan, var, \
                  abs as np_abs, logical_and, isfinite, \
                  all as np_all, append, delete, set_printoptions, \
                  get_include, mgrid, flipud
from pathos.multiprocessing import ProcessPool as mp_pool
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text

import pyximport
pyximport.install(setup_args={"include_dirs":get_include()})

from normcop_cyftns import get_corrcoeff, get_dist, fill_correl_mat, \
                           norm_cdf_py, norm_pdf_py, \
                           norm_ppf_py, norm_ppf_py_arr, \
                           bi_var_copula, bivar_gau_cop_arr

plt.ioff()
set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})


class NormCopulaInfill(object):
    '''Implementation of Bardossy & Pegram 2014 with some bells and jingles

    Description:
    -----------
        To infill missing time series data of a given station(s) using
        neighboring stations and the multivariate normal distribution.
        Can be used for infilling time series data that acts like stream
        discharge or precipitation.

    Parameters:
    ----------
        in_var_file: unicode, DataFrame_like
            Location of the file that holds the time series input data.
            The file should have its first column as time. The header
            should be the names of the stations. Any valid separator is
            allowed.
        in_coords_file: unicode, DataFrame_like
            Location of the file that has stations' coordinates.
            The names of the stations should be the first row. The header
            should have the \'X\' for eastings, \'Y\' for northings. Rest
            is ignored. Any valid separator is allowed.
        out_dir: unicode
            Location of the output directory. Will be created if it does
            not exist. All the ouputs are stored inside this directory.
        infill_stns: list_like
            Names of the stations that should be infilled. These names
            should be in the in_var_file header and the index of
            in_coords_file.
        min_valid_vals: integer
            The minimum number of the union of one station with others
            so that all have valid values. This is different in different
            cases. e.g. for calculating the long term correlation only
            two stations are used but during infilling all stations
            should satisfy this condition with respect to each other. i.e.
            only those days are used to make the correlation matrix on
            which all the neighboring stations have valid
            values at every step.
        infill_interval_type: unicode
            A string that is either: \'slice\', \'indiv\', \'all\'.
            slice: all steps in between infill_dates_list are
            considered for infilling.
            indiv: only fill on these dates.
            all: fill wherever there is a missing value.
        infill_dates_list: list_like, date_like
            A list containing the dates on which infilling is done.
            The way this list is used depends on the value of
            infill_interval_type.
            if infill_interval_type is \'slice\' and infill_dates_list
            is \'[date_1, date_2]\' then all steps in between and
            including date_1 and date_2 are infilled provided that
            neighboring stations have enough valid values at every
            step.
        n_nrst_stns_min: integer
            Number of minimum neighbors to use for infilling.
        n_nrst_stns_max: integer
            Number of maximum stations to use for infilling.
            Normally n_nrst_stns_min stations are used but it could be
            that at a given step one of the neighbors is missing a
            value in that case we have enough stations to choose from.
            This number depends on the given dataset. It cannot be
            less than n_nrst_stns_min.
        ncpus: integer
            Number of processes to initiate. Ideally equal to the
            number of cores available.
        skip_stns: list_like
            The names of the stations that should not be used while
            infilling. Normally, after calling the
            \'cmpt_plot_rank_corr_stns\', one can see the stations
            that do not correlate well with the infill_stns.
        sep: unicode
            The type of separator used in in_var_file and in_coords_file.
            Both of them should be similar. The output will have this as
            the separator as well.
        time_fmt: unicode
            Format of the time in the in_var_file. This is required to
            convert the string time to a datetime object that allows
            better indexing. Any valid time format from the datetime
            module can be used.
        freq: unicode or pandas datetime offset
            The type of interval used in the in_var_file. e.g. for
            days it is \'D\'. It is better to take a look at the
            pandas datetime offsets.
        verbose: bool
            if True, print activity messages
        force_call: bool
            if True then call functions even if they were called before.
            This can be useful sometimes. By default it is False and
            calling some functions will raise an error if called again.
            This is controlled because calling the same function twice
            might make undesirable changes.

    Sequence of the program:
    -----------------------
        After passing the required arguments successfully, calling the
        do_infill method will start performing infilling as desired.
        However, it is better to explain what happens during the
        execution.

        Initiating the class reads all the data and performs some checks
        on it. Failing any of the checks stops execution. Checks include
        shape, type and value checking.

        Certain variables that will be needed later on are also
        initialized.

        Calling the do_infill method calls the cmpt_plot_nrst_stns first
        to get the neighboring stations based on their proximity
        to each infill station. The station names are sorted based on their
        distance from a given infill station. This is done for two reasons
        first to get rid of stations that are too far and are likely to
        contribute less towards the infilling, in this way we don\'t need
        to go through each station while the cmpt_plot_rank_corr_stns method
        is invoked. The second reason is to use stations based on their
        proximity rather than correlation, this is true if
        nrst_stns_type == \'dist\'. At the end it reduces the input data
        to have only those stations that are in the proximity of the
        neighboring stations along with the infill_stns.

        If nrst_stns_type == \'rank\', which is the default, the method
        cmpt_plot_rank_corr_stns is called afterwards and the neighboring
        stations are sorted based on their linear correlation with
        infill_stns.

        Then cmpt_conf_ser method is called. This method checks if the
        probabilities in _conf_probs are ascending (which they should be)
        and that they are within adj_probs_bounds. To see what these
        variables mean, check them in the __init__ and cmpt_conf_ser
        methods.

        If flag_susp_flag is True then checks are made for the probabilities
        in flag_probs. These probabilities should be in _conf_probs and
        ascending order.

        The output DataFrame is also defined. This DataFrame will have the
        same shape as the original in_var_file.

        Then we loop through each infill_stn, output directories are
        created for each infill_stn, other directories are also created
        based on flags such as plot_diag_flag, plot_step_cdf_pdf_flag.
        To see what these flags do, go through the class.

        If ncpus is 1 or debug_mode is True then infilling is done within
        the same process. Only those values are infilled that are missing
        but if the compare_infill_flag is True then all values are
        infilled. Although, values that were present originally are used
        in the final output DataFrame.

        The process of infilling is explained shortly.

        If ncpus >= 2 and debug_mode is False, then infill_dates are
        divided based on the number of ncpus and available records so
        that each cpus is kept busy for an equal amount of time and then
        infilling is carried out. It is done by calling the _infill
        method.

        The _infill function is the actual implementation of
        Bardossy & Pegram 2014 along with provisions for discharge type
        data because the original was for precipitation. Infilling of
        discharge data is easier than precipitation because it does not
        include zeros. In _infill we loop through each data and search
        if enough data is available for infilling based on given criteria,
        then infilling is done otherwise the rest is skipped and next
        step is processed. If infill_type == \'precipitation\' and all
        stations on that step have values equal to var_le_trs then the
        value var_le_trs is also inserted in the output DataFrame.
        The paramter var_le_trs stands for variable less than equality
        threshold i.e. zero by default. It can be changed before
        calling the do_infill method along with var_ge_trs (variable
        greater than equality threshold) i.e. 1.0 by default. These
        are described in the paper.

        Values of the neighboring stations are collected at a given
        step, missing values are dropped. Union of steps at which all
        neighbors have valid values along with infill_stn are selected
        to make the correlation matrix (later). In case of
        precipitation, probabilities of var_le_trs and not exceeding
        var_ge_trs are calculated same as described in the paper.
        Rest of the data is ranked and empirical probabilites are
        calculated. Using these probabilities, standard normal values
        are computed which are then used as input for the normal copula.
        A piece-wise interpolation function is also defined for the
        infill_stn (it is needed later). This function returns the
        probability of a given value of the variable. If plot_diag_flag
        is True, then cummulative distrubtion functions (CDFs) are
        also plotted and saved.

        The correlation matirx is calculated given the standard normal
        values, these are the Pearson correlations. This concludes the
        input data preparation for the normal copula. Putting these
        values into the formulas and using the piece-wise interpolation
        function defined earlier, a CDF is calculated for the values
        at that given step and infill_stn. The procedure differs based
        on \'infill_type\'.

        After each loop, the neighboring stations are checked. If they are
        same as the previous step, then data from the previous is
        used.

        Finally, using this CDF, values at given confidence intervals are
        also calculated and written to a file that is saved in a directory
        with the name of the station inside the out_dir for each step.
        If some plotting flags are True, then plots are also made.

        After infilling, in each process, the data is sent back to do_infill
        where it is combined to form one continuous DataFrame. Figures
        are also plotted for the given steps. All outputs are saved in
        out_dir. Some outputs such as neighboring stations plots are saved
        in separate directories. Outputs related to a single infill_stn
        are saved in a directory that has the name of the infill_stn.

        Some final notes: Given the methods used here, it is impossible to
        estimate a value that is outside the range of
        infill_stn time series. This can be handled to some extent through
        Kernel Density Estimation. Many things are not mentioned in
        this documentation. The user is encouraged to go through the
        implementation and explore :)

    Returns
    -------
        Actually, the do_infill method returns None but important outputs
        saved during the execution are described here.

        out_infilled_var_df.csv: DataFrame
            The infilled DataFrame. Same as in_var_file but with
            infilled values at 50% confidence limits by default.
            All original values are kept intact even for the infill_stns.
            Saved in out_dir.
        out_infilled_flag_var_df.csv: DataFrame
            A DataFrame that holds flags that tell if the actual data is
            out of interpolated bounds. Only for infill_stns though.
            By default these bounds are infill values at 0.05 and 0.95
            probabilities. -1 means below bounds, 0 means within
            bounds and +1 means greater than bounds. Set the
            flag_susp to True before calling the do_infill method.
            This sets the compare_infill flag to True
            when do_infill is called. Saved in out_dir.
        flag_infill_STATION_NAME.png: figure
            For each infill_stn, the plot of out_infilled_flag_var_df.csv.
            Saved inside a directory with the name of the station.
        stn_infill_cdfs and stn_infill_pdfs: directory
            These hold the computed CDFs and probabilitiy density
            functions (PDFs) plots if the plot_step_cdf_pdf_flag
            or plot_diag_flag are True.
            Saved inside a directory with the name of the station.
        stn_step_cdfs and stn_step_corrs: directory
            These hold the CDFs of each station and the correlation
            matrix used for the normal copula that was used during
            the infilling process of each step respectively. Created
            if plot_diag_flag is True.
            Saved inside a directory with the name of the station.
        stn_STATION_NAME_infill_conf_vals_df.csv: DataFrame
            For each infill_stn, the infill values at given confidence
            intervals. By default these are the 0.05, 0.25, 0.5, 0.75, 0.95
            confidence limit infill values.
            Saved inside a directory with the name of the station.
        missing_infill_STATION_NAME.png: figure
            A figure showing the final infilled time series of a given
            infill_stn along with values at confidence intervals.
            Saved inside a directory with the name of the station.
        compare_infill_STATION_NAME.png: figure
            If the flag compare_infill_flag is True. Then a figure comparing
            the original variable values against infilled values is
            plotted. This is done only for those values that were not
            missing.
            Saved inside a directory with the name of the station.

        Other methods also create outputs as well but they are not
        explained here.

    '''

    def __init__(self,
                 in_var_file,
                 in_coords_file,
                 out_dir,
                 infill_stns,
                 min_valid_vals,
                 infill_interval_type,
                 infill_type=None,
                 infill_dates_list=None,
                 n_nrst_stns_min=1,
                 n_nrst_stns_max=1,
                 ncpus=1,
                 skip_stns=None,
                 sep=';',
                 time_fmt='%Y-%m-%d',
                 freq='D',
                 verbose=True,
                 force_call=False):

        self.verbose = verbose
        self.in_var_file = in_var_file
        self.out_dir = out_dir
        self.infill_stns = infill_stns
        self.infill_interval_type = infill_interval_type
        self.infill_type = infill_type
        self.infill_dates_list = infill_dates_list
        self.min_valid_vals = min_valid_vals
        self.in_coords_file = in_coords_file
        self.n_nrst_stns_min = n_nrst_stns_min
        self.n_nrst_stns_max = n_nrst_stns_max
        self.ncpus = ncpus
        self.skip_stns = skip_stns
        self.sep = sep
        self.time_fmt = time_fmt
        self.freq = freq
        self.force_call = force_call

        self.in_var_df = read_csv(self.in_var_file, sep=self.sep,
                                  index_col=0, encoding='utf-8')
        self.in_var_df.index = to_datetime(self.in_var_df.index,
                                           format=self.time_fmt)

        assert self.in_var_df.shape[0] > 0, '\'in_var_df\' has no records!'
        assert self.in_var_df.shape[1] > 1, '\'in_var_df\' has < 2 fields!'
        self.in_var_df_orig = self.in_var_df.copy()

        self.in_var_df.columns = map(unicode, self.in_var_df.columns)

        if not os_exists(self.out_dir):
            os_mkdir(self.out_dir)

        if self.verbose:
            print 'INFO: \'in_var_df\' original shape:', self.in_var_df.shape

        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        if self.verbose:
            print 'INFO: \'in_var_df\' shape after dropping NaN steps:', \
                   self.in_var_df.shape

        self.in_coords_df = read_csv(self.in_coords_file, sep=sep,
                                     index_col=0, encoding='utf-8')
        assert self.in_coords_df.shape[0] > 0, '\'in_coords_df\' has no records!'
        assert self.in_coords_df.shape[1] > 1, '\'in_coords_df\' has < 2 fields!'

        self.in_coords_df.index = map(unicode, self.in_coords_df.index)
        self.in_coords_df = self.in_coords_df[~self.in_coords_df.index.duplicated(keep='last')]

        assert u'X' in self.in_coords_df.columns, \
                        'Column \'X\' not in \'in_coords_df\'!'
        assert u'Y' in self.in_coords_df.columns, \
                        'Column \'Y\' not in \'in_coords_df\'!'

        if self.verbose:
            print 'INFO: \'in_coords_df\' original shape:', self.in_coords_df.shape

        if hasattr(self.skip_stns, '__iter__'):
            if len(self.skip_stns) > 0:
                self.in_var_df.drop(labels=self.skip_stns, axis=1,
                                    inplace=True, errors='ignore')
                self.in_coords_df.drop(labels=self.skip_stns, axis=1,
                                       inplace=True, errors='ignore')

        assert isinstance(self.min_valid_vals, int), \
               '\'min_valid_vals\' not an integer!'
        assert self.min_valid_vals >= 1, '\'min_valid_vals\' cannot be less than one!'

        assert isinstance(self.n_nrst_stns_min, int), \
               '\'n_nrst_stns_min\' not an integer!'
        assert isinstance(self.n_nrst_stns_max, int), \
               '\'n_nrst_stns_max\' not an integer!'

        assert self.n_nrst_stns_min >= 1, '\'n_nrst_stns_min\' cannot be < one!'
        assert self.n_nrst_stns_max <= self.in_var_df.shape[1], \
               '\'n_nrst_stns_max\' > total number of stations!'
        assert self.n_nrst_stns_min <= self.n_nrst_stns_max, \
               '\'n_nrst_stns_min\' > \'n_nrst_stns_max\'!'

        assert (self.infill_type == u'discharge') or (self.infill_type == u'precipitation'), \
            '\'infill_type\' can either be \'discharge\' or \'precipitation\'!'

        if self.n_nrst_stns_max + 1 > self.in_var_df.shape[1]:
            self.n_nrst_stns_max = self.in_var_df.shape[1] - len(self.infill_stns)
            print 'WARNING: \'n_nrst_stns_max\' reduced to %d' % self.n_nrst_stns_max

        assert isinstance(self.ncpus, int), '\'ncpus\' not an integer!'
        assert self.ncpus >= 1, '\'ncpus\' cannot be less than one!'

        if (self.infill_interval_type == 'slice') or \
           (self.infill_interval_type == 'indiv'):
            assert hasattr(self.infill_dates_list, '__iter__'), \
               '\'infill_dates_list\' not an iterable!'

        if self.infill_interval_type == 'slice':
            assert len(self.infill_dates_list) == 2, \
            'For infill_interval_type \'slice\' only two objects inside \'infill_dates_list\' are allowed!'
            self.infill_dates_list = to_datetime(self.infill_dates_list,
                                                 format=self.time_fmt)
            assert self.infill_dates_list[1] > self.infill_dates_list[0], \
                   'First infill date < the last!'
            self.infill_dates = date_range(start=self.infill_dates_list[0],
                                           end=self.infill_dates_list[-1],
                                           format=self.time_fmt,
                                           freq=self.freq)

        elif self.infill_interval_type == 'all':
            self.infill_dates_list = None
            self.infill_dates = self.in_var_df.index
        elif self.infill_interval_type == 'indiv':
            assert len(self.infill_dates_list) > 0, \
                   '\'infill_dates_list\' is empty!'
            self.infill_dates = to_datetime(self.infill_dates_list, format=self.time_fmt)
        else:
            assert False, \
            '\'infill_interval_type\' can only be \'slice\', \'all\', or \'indiv\'!'

        self.insuff_val_cols = self.in_var_df.columns[self.in_var_df.count() < self.min_valid_vals]

        if len(self.insuff_val_cols) > 0:
            self.in_var_df.drop(labels=self.insuff_val_cols, axis=1, inplace=True)
            if self.verbose:
                print 'INFO: The following stations (n=%d) are with insufficient values:\n' % \
                      self.insuff_val_cols.shape[0], self.insuff_val_cols

        self.avail_nrst_stns_ser = self.in_var_df.count(axis=1)
        self.insuff_nrst_stns_days = \
        self.in_var_df.loc[self.avail_nrst_stns_ser < self.n_nrst_stns_min]
        self.in_var_df.drop(labels=self.insuff_nrst_stns_days, axis=0,
                            inplace=True, errors='ignore')
        if self.verbose:
            print 'INFO: %d steps were with too few neighboring values' % self.insuff_nrst_stns_days.shape[0]
        self.in_var_df.dropna(axis=(0, 1), how='all', inplace=True)

        if self.verbose:
            print 'INFO: \'in_var_df\' shape after dropping values less than \'min_valid_vals\' and \'n_nrst_stns\':', \
            self.in_var_df.shape
        assert self.min_valid_vals <= self.in_var_df.shape[0], \
        'Number of stations in \'in_var_df\' less than \'min_valid_vals\' after dropping days with insufficient records!'

        self.commn_stns = intersect1d(self.in_var_df.columns,
                                      self.in_coords_df.index)
        self.in_var_df = self.in_var_df[self.commn_stns]

        self.in_coords_df = self.in_coords_df.loc[self.commn_stns]
        self.xs = self.in_coords_df['X'].values
        self.ys = self.in_coords_df['Y'].values

        if self.infill_stns == 'all':
            self.infill_stns = self.in_var_df.columns
        else:
            assert self.n_nrst_stns_max + len(self.infill_stns) <= self.in_var_df.shape[1], \
            'Number of stations in \'in_var_df\' less than \'n_nrst_stns_max\' after dropping stations with insufficient stations!'

        if self.verbose:
            print 'INFO: \'in_var_df\' shape after station name intersection with \'in_coords_df\':', \
            self.in_var_df.shape
            print 'INFO: \'in_coords_df\' shape after station name intersection with \'in_var_df\':', \
            self.in_coords_df.shape

        assert self.n_nrst_stns_max <= self.in_var_df.shape[1], \
        'Number of stations in \'in_var_df\' less than \'n_nrst_stns_max\' after intersecting stations names!'

        for infill_stn in self.infill_stns:
            assert infill_stn in self.in_var_df.columns, \
            'station %s not in input variable dataframe anymore!' % infill_stn

        # check if atleast one infill date is in the in_var_df
        date_in_dates = False
        full_dates = self.in_var_df.index
        for infill_date in self.infill_dates:
            if infill_date in full_dates:
                date_in_dates = True
                break

        assert date_in_dates, \
        'Some or all infill dates do not exist in \'in_var_df\' after dropping stations and records with insufficient information!'

        self.nrst_stns_list = []
        self.nrst_stns_dict = {}

        self.rank_corr_stns_list = []
        self.rank_corr_stns_dict = {}

        self.nrst_stns_type = 'rank' # can be rank or dist

        self.n_discret = 300

        self.fig_size_long = (20, 7)
        self.out_fig_dpi = 80
        self.out_fig_fmt = 'png'

        self._conf_heads = ['var_0.05', 'var_0.25', 'var_0.5', 'var_0.75',
                            'var_0.95']
        self._conf_probs = [0.05, 0.25, 0.5, 0.75, 0.95]
        self._fin_conf_head = self._conf_heads[2]
        self._adj_prob_bounds = [0.005, 0.995]
        self._flag_probs = [0.05, 0.95]
        self.n_round = 3
        self.cop_bins = 20

        self._norm_cop_pool = None

        self._infilled = False
        self._dist_cmptd = False
        self._rank_corr_cmptd = False
        self._conf_ser_cmptd = False

        self.debug_mode_flag = False
        self.plot_diag_flag = False
        self.plot_step_cdf_pdf_flag = False
        self.compare_infill_flag = False
        self.flag_susp_flag = False
        self.force_infill_flag = False  # force infill if avail_cols < n_nrst_stns_min
        self.plot_neighbors_flag = False
        self.take_min_stns_flag = True # to take n_nrst_stns_min stns or all available
        self.overwrite_flag = False
        self.read_pickles_flag = False

        self.out_var_file = os_join(self.out_dir, 'out_infilled_var_df.csv')
        self.out_flag_file = os_join(self.out_dir, 'out_infilled_flag_var_df.csv')
        self.out_stns_avail_file = os_join(self.out_dir, 'out_n_avail_stns_df.csv')
        self.out_stns_avail_fig = os_join(self.out_dir, 'out_n_avail_stns_compare.png')

        self.out_rank_corrs_pkl_file = os_join(self.out_dir, 'out_rank_corrs_mat.pkl')
        self.out_rank_corrs_ctr_pkl_file = os_join(self.out_dir, 'out_rank_corrs_ctr_mat.pkl')

        if self.infill_type == u'precipitation':
            self.var_le_trs = 0.0
            self.var_ge_trs = 1.0
            self.ge_le_trs_n = 1

        return

    def _get_ncpus(self):
        '''
        Set the number of processes to be used

        call it in the first line of the function that has mp in it
        '''
        if self.debug_mode_flag:
            self.ncpus = 1
        elif not hasattr(self._norm_cop_pool, '_id'):
            self._norm_cop_pool = mp_pool(nodes=self.ncpus)
        return

    def cmpt_conf_ser(self):
        '''Check if all the variables for the calculation of confidence intervals
           are correct
        '''
        assert np_any(where(ediff1d(self._conf_probs) > 0, 1, 0)), \
               '\'_conf_probs\' not ascending (%s)!' % repr(self._conf_probs)
        assert len(self._adj_prob_bounds) == 2, \
        '\'adj_bounds_probs\' are not two values (%s)!' % repr(self._adj_prob_bounds)

        assert self._adj_prob_bounds[0] < self._adj_prob_bounds[1], \
        '\'adj_bounds_probs\' not ascending (%s)!' % repr(self._adj_prob_bounds)

        assert max(self._conf_probs) <= self._adj_prob_bounds[1], \
        'max \'adj_prob_bounds\' < max \'_conf_probs\' (%s)!' % repr(self._adj_prob_bounds)
        assert min(self._conf_probs) > self._adj_prob_bounds[0], \
        'min \'adj_prob_bounds\' > min \'_conf_probs\' (%s)!' % repr(self._adj_prob_bounds)

        assert self._fin_conf_head in self._conf_heads, \
        '\'out_conf_prob\': %s not in \'_conf_heads\': %s' % \
        (self._fin_conf_head, self._conf_heads)

        self.conf_ser = Series(index=self._conf_heads, data=self._conf_probs)
        self._conf_ser_cmptd = True
        return

    def cmpt_plot_nrst_stns(self):
        '''Plot nearest stations around each infill station
        '''
        if self.verbose:
            print 'INFO: Computing and plotting nearest stations...'

        if not self.force_call:
            assert not self._dist_cmptd, 'Nearest stations computed already!'

        for infill_stn in self.infill_stns:
            # get the x and y coordinates of the infill_stn
            infill_x, infill_y = \
                      self.in_coords_df[['X', 'Y']].loc[infill_stn].values

            # calculate distances of all stations from the infill_stn
            dists = vectorize(get_dist)(infill_x, infill_y, self.xs, self.ys)
            dists_df = DataFrame(index=self.in_var_df.columns, data=dists,
                                 columns=['dists'])
            dists_df.sort_values('dists', axis=0, inplace=True)

            # take the nearest n_nrn stations to the infill_stn
            for nrst_stn in dists_df.iloc[:self.n_nrst_stns_max + 1].index:
                if nrst_stn not in self.nrst_stns_list:
                    self.nrst_stns_list.append(nrst_stn)

            # put the neighboring stations in a dictionary for each infill_stn
            self.nrst_stns_dict[infill_stn] = \
                            dists_df.iloc[1:self.n_nrst_stns_max + 1].index
            assert len(self.nrst_stns_dict[infill_stn]) >= self.n_nrst_stns_min, \
            'Neighboring stations less than \'n_nrst_stns_min\' for station: %s' % infill_stn

        # have the nrst_stns_list in the in_var_df only
        self.in_var_df = self.in_var_df[self.nrst_stns_list]
        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        for infill_stn in self.infill_stns:
            assert infill_stn in self.in_var_df.columns, \
            'station %s not in input variable dataframe anymore!' % infill_stn

        # check if atleast one infill date is in the in_var_df
        date_in_dates = False
        full_dates = self.in_var_df.index
        for infill_date in self.infill_dates:
            if infill_date in full_dates:
                date_in_dates = True
                break

        assert date_in_dates, \
        'None of the infill dates exist in \'in_var_df\' after dropping stations and records with insufficient information!'

        if self.verbose:
            print 'INFO: \'in_var_df\' shape after calling \'cmpt_plot_nrst_stns\':', \
                  self.in_var_df.shape

        if self.plot_neighbors_flag:
            self.nebor_plots_dir = os_join(self.out_dir, 'neighbor_stns_plots')
            if not os_exists(self.nebor_plots_dir):
                os_mkdir(self.nebor_plots_dir)

            tick_font_size = 5
            for infill_stn in self.infill_stns:
                # get the x and y coordinates of the infill_stn
                infill_x, infill_y = \
                            self.in_coords_df[['X', 'Y']].loc[infill_stn].values

                nrst_stns_ax = plt.subplot(111)
                nrst_stns_ax.scatter(infill_x, infill_y, c='r', label='infill_stn')
                nrst_stns_ax.scatter(self.in_coords_df['X'].loc[self.nrst_stns_dict[infill_stn]],
                                     self.in_coords_df['Y'].loc[self.nrst_stns_dict[infill_stn]],
                                     alpha=0.75, c='c', label='neibor_stn (%d)' % \
                                     self.nrst_stns_dict[infill_stn].shape[0])
                plt_texts = []
                plt_texts.append(nrst_stns_ax.text(infill_x, infill_y, infill_stn,
                                                   va='top', ha='left',
                                                   fontsize=tick_font_size))
                for stn in self.nrst_stns_dict[infill_stn]:
                    plt_texts.append(nrst_stns_ax.text(self.in_coords_df['X'].loc[stn],
                                                       self.in_coords_df['Y'].loc[stn], stn,
                                                       va='top', ha='left', fontsize=5))

                adjust_text(plt_texts, only_move={'points':'y', 'text':'y'})
                nrst_stns_ax.grid()
                nrst_stns_ax.set_xlabel('Eastings', size=tick_font_size)
                nrst_stns_ax.set_ylabel('Northings', size=tick_font_size)
                nrst_stns_ax.legend(framealpha=0.5, loc=0)
                plt.setp(nrst_stns_ax.get_xticklabels(), size=tick_font_size)
                plt.setp(nrst_stns_ax.get_yticklabels(), size=tick_font_size)
                plt.savefig(os_join(self.nebor_plots_dir, '%s_neibor_stns.png' % \
                                    infill_stn), bbox='tight', dpi=300)
                plt.clf()
            plt.close()
        self._dist_cmptd = True
        return

    def cmpt_plot_rank_corr_stns(self):
        '''
        Plot stations around infill stations based on highest pearson rank correlations
        '''
        if self.verbose:
            print 'INFO: Computing highest correlation stations...'

        if not self.force_call:
            assert not self._rank_corr_cmptd, 'Rank correlation stations computed already!'

        recalc_rank_corrs = False
        if os_exists(self.out_rank_corrs_pkl_file) and self.read_pickles_flag:
            if self.verbose:
                print 'INFO: Loading rank correlations pickle...'

            self.rank_corrs_df = read_pickle(self.out_rank_corrs_pkl_file)
            self.rank_corrs_df = self.rank_corrs_df.apply(lambda x: to_numeric(x))

            n_stns = self.rank_corrs_df.shape[1]

            rank_corr_stns = self.rank_corrs_df.columns
            for infill_stn in self.infill_stns:
                if infill_stn not in rank_corr_stns:
                    recalc_rank_corrs = True
                    if self.verbose:
                        print 'INFO: Rank corrlations pickle is not up-to-date!'
                    break

            if not recalc_rank_corrs:
                self.rank_corr_vals_ctr_df = read_pickle(self.out_rank_corrs_ctr_pkl_file)
                self.rank_corr_vals_ctr_df = self.rank_corr_vals_ctr_df.apply(lambda x: to_numeric(x))

                assert self.rank_corrs_df.shape[1] == n_stns, 'Incorrect number of stations in rank correlations pickle!'

                for i in range(n_stns):
                    for j in range(n_stns):
                        if i > j:
                            correl = self.rank_corrs_df.iloc[i, j]
                            if round(correl, 3) == 1.0:
                                print '\a GOOD NEWS!'
                                print 'Stations %s and %s have a correlation of %0.6f!' % \
                                (self.in_var_df.columns[i], self.in_var_df.columns[j], correl)
                                print 'The missing values of one are substituted by the other and no infilling is done for those!'
                                ser_i_full = self.in_var_df.iloc[:, i].copy()
                                ser_j_full = self.in_var_df.iloc[:, j].copy()
                                ser_i_fill = where(isnan(ser_i_full.values), ser_j_full.values, ser_i_full.values)
                                ser_j_fill = where(isnan(ser_j_full.values), ser_i_fill, ser_j_full.values)
                                self.in_var_df.iloc[:, i][:] = ser_i_fill
                                self.in_var_df.iloc[:, j][:] = ser_j_fill
                                i_shape = self.in_var_df.iloc[:, i].dropna().shape[0]
                                j_shape = self.in_var_df.iloc[:, j].dropna().shape[0]
                                assert i_shape == j_shape, 'Unequal shapes. Values not substituted properly!'

                self._rank_corr_cmptd = True
        else:
            recalc_rank_corrs = True

        if recalc_rank_corrs:
            n_stns = self.in_var_df.shape[1]
            self.rank_corrs_df = DataFrame(index=self.in_var_df.columns,
                                           columns=self.in_var_df.columns)

            self.rank_corrs_df = self.rank_corrs_df.apply(lambda x: to_numeric(x))
            self.rank_corr_vals_ctr_df = self.rank_corrs_df.copy()

            tot_corrs_written = 0
            for i in range(n_stns):
                ser_i = self.in_var_df.iloc[:, i].dropna().copy()
                ser_i_index = ser_i.index
                for j in range(n_stns):
                    if i > j:
                        ser_j = self.in_var_df.iloc[:, j].dropna().copy()
                        index_ij = ser_i_index.intersection(ser_j.index)
                        if index_ij.shape[0] > self.min_valid_vals:
                            new_ser_i = ser_i.loc[index_ij].copy()
                            new_ser_j = ser_j.loc[index_ij].copy()
                            rank_ser_i = new_ser_i.rank()
                            rank_ser_j = new_ser_j.rank()
                            correl = get_corrcoeff(rank_ser_i.values, rank_ser_j.values)
                            if round(correl, 3) == 1.0:
                                print 'Stations %s and %s have a correlation of 1!' % \
                                (self.in_var_df.columns[i], self.in_var_df.columns[j])
                                print 'The missing values of one are substituted by the other one and no infilling is done for those!'
                                ser_i_full = self.in_var_df.iloc[:, i].copy()
                                ser_j_full = self.in_var_df.iloc[:, j].copy()
                                ser_i_fill = where(isnan(ser_i_full.values), ser_j_full.values, ser_i_full.values)
                                ser_j_fill = where(isnan(ser_j_full.values), ser_i_fill, ser_j_full.values)
                                self.in_var_df.iloc[:, i][:] = ser_i_fill
                                self.in_var_df.iloc[:, j][:] = ser_j_fill
                                i_shape = self.in_var_df.iloc[:, i].dropna().shape[0]
                                j_shape = self.in_var_df.iloc[:, j].dropna().shape[0]
                                assert i_shape == j_shape, 'Unequal shapes. Values not substituted properly!'

                            self.rank_corrs_df.iloc[i, j] = correl
                            self.rank_corr_vals_ctr_df.iloc[i, j] = new_ser_i.shape[0]
                            tot_corrs_written += 2
                    elif i == j:
                        self.rank_corrs_df.iloc[i, j] = 1.0
                        self.rank_corr_vals_ctr_df.iloc[i, j] = ser_i.shape[0]
                        tot_corrs_written += 1

            for i in range(n_stns):
                for j in range(n_stns):
                    if i < j:
                        self.rank_corrs_df.iloc[i, j] = self.rank_corrs_df.iloc[j, i]
                        self.rank_corr_vals_ctr_df.iloc[i, j] = \
                                self.rank_corr_vals_ctr_df.iloc[j, i]

            if self.verbose:
                print 'INFO: %d out of possible %d correlations written' % \
                      (tot_corrs_written, (n_stns**2.))

        for infill_stn in self.infill_stns:
            stn_correl_ser = self.rank_corrs_df[infill_stn].copy()
            stn_correl_ser[:] = np_abs(stn_correl_ser[:])
            stn_correl_ser.sort_values(axis=0, ascending=False, inplace=True)
            stn_correl_ser.values[1:][np_round(stn_correl_ser.values, 3)[1:] == 1.0] = nan

            # take the nearest n_nrn stations to the infill_stn
            for rank_corr_stn in stn_correl_ser.iloc[:self.n_nrst_stns_max + 1].index:
                if rank_corr_stn not in self.rank_corr_stns_list:
                    self.rank_corr_stns_list.append(rank_corr_stn)

            # put the neighboring stations in a dictionary for each infill_stn
            self.rank_corr_stns_dict[infill_stn] = \
                            stn_correl_ser.iloc[1:self.n_nrst_stns_max + 1].index

            assert len(self.rank_corr_stns_dict[infill_stn]) >= self.n_nrst_stns_min, \
            'Rank correlation stations less than \'n_nrst_stns_min\' for station: %s' % infill_stn

        # have the rank_corr_stns_list in the in_var_df only
        self.in_var_df = self.in_var_df[self.rank_corr_stns_list]
        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        for infill_stn in self.infill_stns:
            assert infill_stn in self.in_var_df.columns, \
            'station %s not in input variable dataframe anymore!' % infill_stn

        # check if atleast one infill date is in the in_var_df
        date_in_dates = False
        full_dates = self.in_var_df.index
        for i, infill_date in enumerate(self.infill_dates):
            if infill_date in full_dates:
                date_in_dates = True
                break

        assert date_in_dates, \
        'None of the infill dates exist in \'in_var_df\' after dropping stations and records with insufficient information!'

        if self.verbose:
            print 'INFO: \'in_var_df\' shape after calling \'cmpt_plot_rank_corr_stns\':', \
                  self.in_var_df.shape

        if self.plot_neighbors_flag:
            self.rank_corr_plots_dir = os_join(self.out_dir, 'rank_corr_stns_plots')
            if not os_exists(self.rank_corr_plots_dir):
                os_mkdir(self.rank_corr_plots_dir)

            tick_font_size = 5
            for infill_stn in self.infill_stns:
                # get the x and y coordinates of the infill_stn
                infill_x, infill_y = \
                            self.in_coords_df[['X', 'Y']].loc[infill_stn].values

                hi_corr_stns_ax = plt.subplot(111)
                hi_corr_stns_ax.scatter(infill_x, infill_y, c='r', label='infill_stn')
                hi_corr_stns_ax.scatter(self.in_coords_df['X'].loc[self.rank_corr_stns_dict[infill_stn]],
                                        self.in_coords_df['Y'].loc[self.rank_corr_stns_dict[infill_stn]],
                                        alpha=0.75, c='c', label='hi_corr_stn (%d)' % \
                                        self.rank_corr_stns_dict[infill_stn].shape[0])
                plt_texts = []
                plt_texts.append(hi_corr_stns_ax.text(infill_x, infill_y, infill_stn,
                                                      va='top', ha='left',
                                                      fontsize=tick_font_size))
                for stn in self.rank_corr_stns_dict[infill_stn]:
                    if not infill_stn == stn:
                        plt_texts.append(hi_corr_stns_ax.text(self.in_coords_df['X'].loc[stn],
                                                              self.in_coords_df['Y'].loc[stn], '%s\n(%0.2f)' % \
                                                              (stn, self.rank_corrs_df[stn].loc[infill_stn]),
                                                              va='top', ha='left', fontsize=5))

                adjust_text(plt_texts, only_move={'points':'y', 'text':'y'})
                hi_corr_stns_ax.grid()
                hi_corr_stns_ax.set_xlabel('Eastings', size=tick_font_size)
                hi_corr_stns_ax.set_ylabel('Northings', size=tick_font_size)
                hi_corr_stns_ax.legend(framealpha=0.5, loc=0)
                plt.setp(hi_corr_stns_ax.get_xticklabels(), size=tick_font_size)
                plt.setp(hi_corr_stns_ax.get_yticklabels(), size=tick_font_size)
                plt.savefig(os_join(self.rank_corr_plots_dir, 'rank_corr_stn_%s.png' % \
                                    infill_stn), bbox='tight', dpi=300)
                plt.clf()

        tick_font_size = 3
        corrs_arr = self.rank_corrs_df.values
        corrs_ctr_arr = self.rank_corr_vals_ctr_df.values
        corrs_ctr_arr[isnan(corrs_ctr_arr)] = 0
        n_stns = corrs_arr.shape[0]

        fig, corrs_ax = plt.subplots(1, 1, figsize=(0.4 * n_stns, 0.4 * n_stns))
        corrs_ax.matshow(corrs_arr, vmin=0, vmax=1, cmap=cmaps.Blues, origin='lower')
        for s in zip(repeat(range(n_stns), n_stns), tile(range(n_stns), n_stns)):
            corrs_ax.text(s[1], s[0], '%0.2f\n(%d)' % (corrs_arr[s[0], s[1]],
                                                       int(corrs_ctr_arr[s[0], s[1]])),
                          va='center', ha='center', fontsize=tick_font_size)

        corrs_ax.set_xticks(range(0, n_stns))
        corrs_ax.set_xticklabels(self.rank_corrs_df.columns)
        corrs_ax.set_yticks(range(0, n_stns))
        corrs_ax.set_yticklabels(self.rank_corrs_df.columns)

        corrs_ax.spines['left'].set_position(('outward', 10))
        corrs_ax.spines['right'].set_position(('outward', 10))
        corrs_ax.spines['top'].set_position(('outward', 10))
        corrs_ax.spines['bottom'].set_position(('outward', 10))

        corrs_ax.tick_params(labelleft=True,
                             labelbottom=True,
                             labeltop=True,
                             labelright=True)

        plt.setp(corrs_ax.get_xticklabels(), size=tick_font_size, rotation=45)
        plt.setp(corrs_ax.get_yticklabels(), size=tick_font_size)

        plt.savefig(os_join(self.out_dir, 'long_term_stn_rank_corrs.png'), dpi=300, bbox='tight')
        plt.close()

        self.rank_corrs_df.to_pickle(self.out_rank_corrs_pkl_file)
        self.rank_corr_vals_ctr_df.to_pickle(self.out_rank_corrs_ctr_pkl_file)
        self._rank_corr_cmptd = True
        return

    def cmpt_plot_stats(self):
        '''Compute and plot statistics of each station
        '''
        if self.verbose:
            print 'INFO: Computing and plotting input variable statistics...'

        stats_cols = ['min', 'max', 'mean', 'stdev',
                      'CoV', 'skew', 'count']
        self.stats_df = DataFrame(index=self.in_var_df.columns, columns=stats_cols)

        for i, stn in enumerate(self.stats_df.index):
            curr_ser = self.in_var_df[stn].dropna().copy()
            curr_min = curr_ser.min()
            curr_max = curr_ser.max()
            curr_mean = curr_ser.mean()
            curr_stdev = curr_ser.std()
            curr_coovar = curr_stdev / curr_mean
            curr_skew = curr_ser.skew()
            curr_count = curr_ser.count()
            self.stats_df.iloc[i] = [curr_min, curr_max, curr_mean,
                                     curr_stdev, curr_coovar, curr_skew,
                                     curr_count]

        self.stats_df = self.stats_df.apply(lambda x: to_numeric(x))

        tick_font_size = 5
        stats_arr = self.stats_df.values
        n_stns = stats_arr.shape[0]
        n_cols = stats_arr.shape[1]

        fig, stats_ax = plt.subplots(1, 1, figsize=(0.45 * n_stns, 0.8 * n_cols))
        stats_ax.matshow(stats_arr.T, cmap=cmaps.Blues, vmin=0, vmax=0,
                         origin='upper')

        for s in zip(repeat(range(n_stns), n_cols), tile(range(n_cols), n_stns)):
            stats_ax.text(s[0], s[1], ('%0.2f' % stats_arr[s[0], s[1]]).rstrip('0'),
                          va='center', ha='center', fontsize=tick_font_size)

        stats_ax.set_xticks(range(0, n_stns))
        stats_ax.set_xticklabels(self.stats_df.index)
        stats_ax.set_yticks(range(0, n_cols))
        stats_ax.set_yticklabels(self.stats_df.columns)

        stats_ax.spines['left'].set_position(('outward', 10))
        stats_ax.spines['right'].set_position(('outward', 10))
        stats_ax.spines['top'].set_position(('outward', 10))
        stats_ax.spines['bottom'].set_position(('outward', 10))

        stats_ax.set_xlabel('Stations', size=tick_font_size)
        stats_ax.set_ylabel('Statistics', size=tick_font_size)

        stats_ax.tick_params(labelleft=True,
                             labelbottom=True,
                             labeltop=True,
                             labelright=True)

        plt.setp(stats_ax.get_xticklabels(), size=tick_font_size, rotation=45)
        plt.setp(stats_ax.get_yticklabels(), size=tick_font_size)

        plt.savefig(os_join(self.out_dir, 'var_statistics.png'), dpi=300, bbox='tight')
        plt.close()
        return

    def plot_ecops(self):
        '''Plot empirical copulas of each station against all other
        '''
        if self.verbose:
            print 'INFO: Plotting empirical copulas of infill stations against others...'

        self.ecops_dir = os_join(self.out_dir, 'empirical_copula_plots')
        if not os_exists(self.ecops_dir):
            os_mkdir(self.ecops_dir)

        self._get_ncpus()

        if (self.ncpus == 1) or self.debug_mode_flag:
            self._plot_ecops(self.in_var_df.columns)
        else:
            idxs = linspace(0, len(self.in_var_df.columns), self.ncpus + 1,
                            endpoint=True, dtype='int64')
            sub_cols = []
            for idx in range(self.ncpus):
                sub_cols.append(self.in_var_df.columns[idxs[idx]:idxs[idx + 1]])

            try:
                list(self._norm_cop_pool.uimap(self._plot_ecops, sub_cols))
                self._norm_cop_pool.clear()
            except:
                self._norm_cop_pool.close()
                self._norm_cop_pool.join()
                return
        return

    def _plot_ecops(self, columns):
        try:
            n_ticks = 6
            x_mesh, y_mesh = mgrid[0:self.cop_bins + 1, 0:self.cop_bins + 1]
            cop_ax_ticks = linspace(0, self.cop_bins, n_ticks)
            cop_ax_labs = np_round(linspace(0, 1., n_ticks, dtype='float'), 1)

            n_rows, n_cols = 1, 15 + 1

            plt.figure(figsize=(19, 6))
            ecop_raw_ax = plt.subplot2grid((n_rows, n_cols), (0, 0), rowspan=1, colspan=5)
            ecop_grid_ax = plt.subplot2grid((n_rows, n_cols), (0, 5), rowspan=1, colspan=5)
            gau_cop_ax = plt.subplot2grid((n_rows, n_cols), (0, 10), rowspan=1, colspan=5)
            leg_ax = plt.subplot2grid((n_rows, n_cols), (0, 15), rowspan=1, colspan=1)
            
            divider = make_axes_locatable(leg_ax)
            cax = divider.append_axes("left", size="100%", pad=0.0)
            
            for infill_stn in self.infill_stns:
                ser_i = self.in_var_df.loc[:, infill_stn].dropna().copy()
                ser_i_index = ser_i.index
                for other_stn in columns:
                    if infill_stn != other_stn:
                        ser_j = self.in_var_df.loc[:, other_stn].dropna().copy()
                        index_ij = ser_i_index.intersection(ser_j.index)
                        if index_ij.shape[0] > self.min_valid_vals:
                            new_ser_i = ser_i.loc[index_ij].copy()
                            new_ser_j = ser_j.loc[index_ij].copy()
                            prob_i = new_ser_i.rank().values / (new_ser_i.shape[0] + 1.)
                            prob_j = new_ser_j.rank().values / (new_ser_j.shape[0] + 1.)

                            # plot the empirical copula
                            if prob_i.min() < 0 or prob_i.max() > 1:
                                raise Exception('\'prob_i\' values out of bounds!')
                            if prob_j.min() < 0 or prob_j.max() > 1:
                                raise Exception('\'prob_j\' values out of bounds!')

                            correl = get_corrcoeff(prob_i, prob_j)

                            # Empirical copula - scatter
                            ecop_raw_ax.scatter(prob_i, prob_j, alpha=0.9,
                                                color='b', s=0.5)
                            ecop_raw_ax.set_xlabel('infill station: %s' % infill_stn)
                            ecop_raw_ax.set_ylabel('other station: %s' % other_stn)
                            ecop_raw_ax.set_xlim(0, 1)
                            ecop_raw_ax.set_ylim(0, 1)
                            ecop_raw_ax.grid()
                            ecop_raw_ax.set_title('Empirical Copula - Scatter')

                            # Empirical copula - gridded
                            cop_dict = bi_var_copula(prob_i, prob_j, self.cop_bins)
                            emp_dens_arr = cop_dict['emp_dens_arr']
                            ecop_grid_ax.pcolormesh(x_mesh, y_mesh, emp_dens_arr, cmap=cmaps.Blues,
                                                    vmin=0, vmax=emp_dens_arr.max())
                            ecop_grid_ax.set_xlabel('infill station: %s' % infill_stn)
                            ecop_grid_ax.set_xticks(cop_ax_ticks)
                            ecop_grid_ax.set_xticklabels(cop_ax_labs)
                            ecop_grid_ax.set_yticks(cop_ax_ticks)
                            ecop_grid_ax.set_yticklabels([])
                            ecop_grid_ax.set_xlim(0, self.cop_bins)
                            ecop_grid_ax.set_ylim(0, self.cop_bins)
                            ecop_grid_ax.set_title('Empirical copula - Gridded')

                            # Corresponding gaussian grid
                            gau_cop_arr = bivar_gau_cop_arr(correl, self.cop_bins)
                            _cb = gau_cop_ax.pcolormesh(x_mesh, y_mesh, gau_cop_arr, cmap=cmaps.Blues,
                                                        vmin=0, vmax=emp_dens_arr.max())
                            gau_cop_ax.set_xlabel('infill station: %s' % infill_stn)
                            gau_cop_ax.set_xticks(cop_ax_ticks)
                            gau_cop_ax.set_xticklabels(cop_ax_labs)
                            gau_cop_ax.set_yticks(cop_ax_ticks)
                            gau_cop_ax.set_yticklabels(cop_ax_labs)
                            gau_cop_ax.tick_params(labelleft=False,
                                                   labelbottom=True,
                                                   labeltop=False,
                                                   labelright=False)
                            gau_cop_ax.set_xlim(0, self.cop_bins)
                            gau_cop_ax.set_ylim(0, self.cop_bins)
                            gau_cop_ax.set_title('Gaussian copula')

                            leg_ax.set_axis_off()
                            cb = plt.colorbar(_cb, cax=cax)
                            cb.set_label('copula density')
                            cb.set_ticks(linspace(0, emp_dens_arr.max(), 5))
                            cb.set_ticklabels(np_round(linspace(0, emp_dens_arr.max(), 5), 1))
                            #cb.ax.tick_params(labelsize=tick_font_size)

                            plt.suptitle('Copula densities of stations: %s and %s\n (n = %d, corr = %0.3f)' % \
                                         (infill_stn, other_stn, prob_i.shape[0], correl))

                            plt.subplots_adjust(hspace=0.15, wspace=1, top=0.8)
                            out_ecop_fig_loc = os_join(self.ecops_dir,
                                                            'ecop_%s_vs_%s.%s' % (infill_stn, other_stn,
                                                                                  self.out_fig_fmt))
                            plt.savefig(out_ecop_fig_loc, dpi=self.out_fig_dpi, bbox='tight')
                            ecop_raw_ax.cla()
                            ecop_grid_ax.cla()
                            leg_ax.cla()
            plt.close()
            return
        except:
            self._full_tb(exc_info())

    def do_infill(self):
        '''Perform the infilling based on given data
        '''
        if self.plot_diag_flag:
            self.compare_infill_flag = True
            self.force_infill_flag = True
            self.plot_step_cdf_pdf_flag = True
            self.flag_susp_flag = True

        if not self._dist_cmptd:
            self.cmpt_plot_nrst_stns()
            assert self._dist_cmptd, 'Call \'cmpt_plot_nrst_stns\' first!'

        if self.nrst_stns_type == 'rank':
            if not self._rank_corr_cmptd:
                self.cmpt_plot_rank_corr_stns()
                assert self._rank_corr_cmptd, \
                       'Call \'cmpt_plot_rank_corr_stns\' first!'
        elif self.nrst_stns_type == 'dist':
            pass
        else:
            assert False, 'Incorrect \'nrst_stns_type\': (%s)' % \
                   repr(self.nrst_stns_type)

        self.cmpt_conf_ser()
        assert self._conf_ser_cmptd, 'Call \'cmpt_conf_ser\' first!'

        if self.infill_type == u'precipitation':
            assert self.var_le_trs <= self.var_ge_trs, '\'var_le_trs\' > \'var_ge_trs\'!'
        else:
            self.var_le_trs, self.var_ge_trs, self.ge_le_trs_n = 3*[None]

        if self.flag_susp_flag:
            assert len(self._flag_probs) == 2, \
            'Only two values allowed inside \'_flag_probs\'!'
            assert self._flag_probs[0] < self._flag_probs[1], \
            '\'_flags_probs\': first value should be smaller than the last!'

            for _flag_val in self._flag_probs:
                assert _flag_val in self._conf_probs, '\'_flag_probs\' value not in \'_conf_probs\'!'
                assert isinstance(_flag_val, float), '\'_flag_probs\' can only be floats!'

            self.flag_df = DataFrame(columns=self.infill_stns,
                                     index=self.infill_dates)
            self.compare_infill_flag = True

        self._get_ncpus()

        if self.plot_diag_flag:
            self.plot_step_cdf_pdf_flag = True

        self.out_var_df = self.in_var_df_orig.copy()

        if self.verbose:
            print 'INFO: Flag conditions:'
            print '  \a compare_infill:', self.compare_infill_flag
            print '  \a plot_step_cdf_pdf:', self.plot_step_cdf_pdf_flag
            print '  \a flag_susp:', self.flag_susp_flag
            print '  \a plot_diag:', self.plot_diag_flag
            print '  \a plot_neighbors:', self.plot_neighbors_flag
            print '  \a overwrite:', self.overwrite_flag

        if self.verbose:
            print 'INFO: Infilling...'
            print 'INFO: using \'%s\' to get nearest stations' % \
                  self.nrst_stns_type
            print 'INFO: infilling type is:', self.infill_type

            print 'INFO: Maximum records to process per station: %d' % \
                  self.infill_dates.shape[0]
            print 'INFO: Total number of stations to infill: %d\n' % \
                  len(self.infill_stns)

        for ii, infill_stn in enumerate(self.infill_stns):
            if self.verbose:
                print '  \a Going through station: %d, %s' % (ii, infill_stn)

            self.curr_infill_stn = infill_stn
            self.stn_out_dir = os_join(self.out_dir, infill_stn)
            out_conf_df_file = os_join(self.stn_out_dir,
                                       'stn_%s_infill_conf_vals_df.csv' % infill_stn)
            no_out = True
            if not self.overwrite_flag:
                if os_exists(out_conf_df_file):
                    if self.verbose:
                        print '    \a Output exists already. Not overwriting.'
                    try:
                        out_conf_df = read_csv(out_conf_df_file, sep=str(self.sep), encoding='utf-8', index_col=0)
                        out_conf_df.index = to_datetime(out_conf_df.index, format=self.time_fmt)
                        out_stn_ser = self.in_var_df_orig.loc[self.infill_dates, self.curr_infill_stn].where(
                                      logical_not(isnan(self.in_var_df_orig.loc[self.infill_dates, self.curr_infill_stn])),
                                      out_conf_df[self._fin_conf_head], axis=0)
                        self.out_var_df.loc[out_conf_df.index, infill_stn] = out_stn_ser
                        no_out = False
                    except Exception as msg:
                        print 'Error while trying to read and update values from the existing dataframe:', msg
                        raise Exception
                else:
                    no_out = True
            if self.overwrite_flag or no_out:
                if not self.compare_infill_flag:
                    nan_idxs = where(isnan(self.in_var_df.loc[self.infill_dates, self.curr_infill_stn].values))[0]
                else:
                    nan_idxs = range(self.infill_dates.shape[0])

                n_nan_idxs = len(nan_idxs)
                if self.verbose:
                    print '    \a %d steps to infill' % n_nan_idxs

                if (n_nan_idxs == 0) and (not self.compare_infill_flag):
                    continue

                if self.nrst_stns_type == 'rank':
                    self.curr_nrst_stns = self.rank_corr_stns_dict[infill_stn]
                elif self.nrst_stns_type == 'dist':
                    self.curr_nrst_stns = self.nrst_stns_dict[infill_stn]
                else:
                    assert False, 'Incorrect \'nrst_stns_type\'!'

                dir_list = [self.stn_out_dir]
                if self.plot_step_cdf_pdf_flag:
                    self.stn_infill_cdfs_dir = os_join(self.stn_out_dir, 'stn_infill_cdfs')
                    self.stn_infill_pdfs_dir = os_join(self.stn_out_dir, 'stn_infill_pdfs')
                    dir_list.extend([self.stn_infill_cdfs_dir, self.stn_infill_pdfs_dir])

                if self.plot_diag_flag:
                    self.stn_step_cdfs_dir = os_join(self.stn_out_dir, 'stn_step_cdfs')
                    self.stn_step_corrs_dir = os_join(self.stn_out_dir, 'stn_step_corrs')
                    dir_list.extend([self.stn_step_cdfs_dir, self.stn_step_corrs_dir])

                for _dir in dir_list:
                    if not os_exists(_dir):
                        os_mkdir(_dir)

                idxs = linspace(0, n_nan_idxs, self.ncpus + 1,
                                endpoint=True, dtype='int64')

                infill_start = timeit.default_timer()

                if (idxs.shape[0] == 1) or (self.ncpus == 1) or self.debug_mode_flag:
                    out_conf_df = self._infill(self.infill_dates)
                    out_stn_ser = self.in_var_df_orig.loc[self.infill_dates, self.curr_infill_stn].where(
                                  logical_not(isnan(self.in_var_df_orig.loc[self.infill_dates, self.curr_infill_stn])),
                                  out_conf_df[self._fin_conf_head], axis=0)
                    self.out_var_df.loc[out_conf_df.index, infill_stn] = out_stn_ser

                else:
                    n_sub_dates = 0
                    sub_infill_dates_list = []
                    for idx in range(self.ncpus):
                        sub_dates = self.infill_dates[nan_idxs[idxs[idx]:idxs[idx + 1]]]
                        sub_infill_dates_list.append(sub_dates)
                        n_sub_dates += sub_dates.shape[0]

                    assert n_sub_dates == n_nan_idxs, \
                    '\'n_sub_dates\' (%d) and \'self.infill_dates\' (%d) of unequal length!' % \
                    (n_sub_dates, self.infill_dates.shape[0])

                    try:
                        sub_conf_dfs = list(self._norm_cop_pool.uimap(self._infill, sub_infill_dates_list))
                    except:
                        self._norm_cop_pool.close()
                        self._norm_cop_pool.join()
                        return

                    self._norm_cop_pool.clear()
                    out_conf_df = DataFrame(index=self.infill_dates,
                                            columns=self.conf_ser.index)

                    for sub_conf_df in sub_conf_dfs:
                        sub_stn_ser = self.in_var_df_orig.loc[sub_conf_df.index,
                                      self.curr_infill_stn].where(
                                      logical_not(isnan(self.in_var_df_orig.loc[sub_conf_df.index,
                                                                                self.curr_infill_stn])),
                                      sub_conf_df[self._fin_conf_head], axis=0)
                        self.out_var_df.loc[sub_conf_df.index, infill_stn] = sub_stn_ser
                        out_conf_df.update(sub_conf_df)

                if self.verbose:
                    print '    \a %d steps infilled' % out_conf_df.dropna().shape[0]

                    infill_stop = timeit.default_timer()
                    fin_secs = infill_stop - infill_start
                    print '    \a Took %0.3f secs, %0.3e secs per step' % (fin_secs, fin_secs / n_nan_idxs)

                out_conf_df = out_conf_df.apply(lambda x: to_numeric(x))
                out_conf_df.to_csv(out_conf_df_file, sep=str(self.sep),
                                   encoding='utf-8')

            # plot the infilled series
            lw = 0.8
            alpha = 0.7
            plt.figure(figsize=self.fig_size_long)
            act_var = self.in_var_df_orig[infill_stn].loc[self.infill_dates].values

            if self.infill_dates.shape[0] <= 100 and \
               (not os_exists(os_join(self.stn_out_dir,
                                      'missing_infill_%s.png' % infill_stn))):
                # plot type 1
                infill_ax = plt.subplot(111)
                full_data_idxs = isnan(act_var)
                for _conf_head in self.conf_ser.index:
                    conf_var_vals = where(full_data_idxs,
                    out_conf_df[_conf_head].loc[self.infill_dates], act_var)
                    infill_ax.plot(self.infill_dates, conf_var_vals,
                                   label=_conf_head, alpha=alpha, lw=lw, ls='-',
                                   marker='o', ms=lw+0.5)
                infill_ax.plot(self.infill_dates, act_var, label='actual', c='k',
                               ls='-', marker='o', alpha=1.0, lw=lw+0.5, ms=lw+1)

                infill_ax.set_xlabel('Time')
                infill_ax.set_ylabel('var_val')
                infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
                plt.grid()
                plt.legend(framealpha=0.5, loc=0)
                plt.savefig(os_join(self.stn_out_dir,
                                    'missing_infill_%s.png' % infill_stn),
                                    dpi=600, bbox='tight')
                plt.clf()

            if self.compare_infill_flag or \
               (not os_exists(os_join(self.stn_out_dir,
                                              'compare_infill_%s.png' % infill_stn))):
                # plot type 2
                infill_ax = plt.subplot(111)

                interp_data_idxs = logical_or(isnan(act_var),
                isnan(out_conf_df[self._fin_conf_head].loc[self.infill_dates].values))

                for _conf_head in self.conf_ser.index:
                    conf_var_vals = where(interp_data_idxs, nan,
                    out_conf_df[_conf_head].loc[self.infill_dates])

                    infill_ax.plot(self.infill_dates, conf_var_vals,
                                   label=_conf_head, alpha=alpha, lw=lw,
                                   ls='-', marker='o', ms=lw+0.5)

                infill_ax.plot(self.infill_dates, where(interp_data_idxs, nan, act_var),
                               label='actual', c='k', ls='-', marker='o',
                               alpha=1.0, lw=lw+0.5, ms=lw+1)

                infill_ax.set_xlabel('Time')
                infill_ax.set_ylabel('var_val')
                infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
                plt.grid()
                plt.legend(framealpha=0.5, loc=0)
                plt.savefig(os_join(self.stn_out_dir,
                                    'compare_infill_%s.png' % infill_stn),
                                    dpi=600, bbox='tight')
                plt.clf()

            if self.flag_susp_flag:
                # plot type 3
                infill_ax = plt.subplot(111)

                interp_data_idxs = logical_or(isnan(act_var),
                isnan(out_conf_df[self._fin_conf_head].loc[self.infill_dates].values))

                _conf_head_list = []
                for _conf_head in self.conf_ser.index:
                    if self.conf_ser[_conf_head] in self._flag_probs:
                        _conf_head_list.append(_conf_head)

                conf_var_vals_lo = where(interp_data_idxs, nan,
                out_conf_df[_conf_head_list[0]].loc[self.infill_dates])
                conf_var_vals_hi = where(interp_data_idxs, nan,
                out_conf_df[_conf_head_list[1]].loc[self.infill_dates])

                not_interp_data_idxs = logical_not(interp_data_idxs)
                flag_ser = full(act_var.shape[0], nan)
                flag_ser[not_interp_data_idxs] = where(act_var[not_interp_data_idxs] < conf_var_vals_lo[not_interp_data_idxs], -1, 0)
                flag_ser[not_interp_data_idxs] = where(act_var[not_interp_data_idxs] > conf_var_vals_hi[not_interp_data_idxs], 1, flag_ser[not_interp_data_idxs])

                self.flag_df[infill_stn] = flag_ser

                if not os_exists(os_join(os_join(self.stn_out_dir,
                                                 'flag_infill_%s.png' % infill_stn))):

                    infill_ax.plot(self.infill_dates[not_interp_data_idxs], flag_ser[not_interp_data_idxs],
                                   label='flag', alpha=alpha, lw=lw+0.5, ls='-')

                    infill_ax.set_xlabel('Time')
                    infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
                    infill_ax.set_ylabel('Flag')
                    infill_ax.set_yticks([-1, 0, 1])
                    infill_ax.set_yticklabels(['below_%0.3fP' % self._flag_probs[0], \
                                               'within\n%0.3fP_&_%0.3fP' % (self._flag_probs[0],
                                                                            self._flag_probs[1]),
                                               'above_%0.3fP' % self._flag_probs[1]])

                    plt.suptitle('Data quality flags')
                    infill_ax.set_ylim(-2, 2)
                    plt.grid()
                    plt.legend(framealpha=0.5, loc=0)
                    plt.savefig(os_join(self.stn_out_dir, 'flag_infill_%s.png' % infill_stn),
                                dpi=600, bbox='tight')
                    plt.clf()

            plt.close()

        self.out_var_df.to_csv(self.out_var_file, sep=str(self.sep), encoding='utf-8')
        if self.flag_susp_flag:
            self.flag_df.to_csv(self.out_flag_file, sep=str(self.sep), encoding='utf-8')
        self._infilled = True
        print '\n'
        return

    def cmpt_plot_avail_stns(self):
        '''To compare the number of stations before and after infilling
        '''

        if self.verbose:
            print 'INFO: Computing and plotting number of stations available per step...'

        assert self._infilled, 'Call \'do_infill\' first!'

        self.avail_nrst_stns_orig_ser = self.in_var_df_orig.count(axis=1)
        self.avail_nrst_stns_ser = self.out_var_df.count(axis=1)

        assert self.avail_nrst_stns_orig_ser.sum() > 0, 'in_var_df is empty!'
        assert self.avail_nrst_stns_ser.sum() > 0, 'out_var_df is empty!'

        plt.figure(figsize=self.fig_size_long)
        plt.plot(self.avail_nrst_stns_orig_ser.index,
                 self.avail_nrst_stns_orig_ser.values,
                 alpha=0.8, label='Original')
        plt.plot(self.avail_nrst_stns_ser.index,
                 self.avail_nrst_stns_ser.values,
                 alpha=0.8, label='Infilled')
        plt.xlabel('Time')
        plt.ylabel('Number of stations with valid values')
        plt.legend(framealpha=0.5)
        plt.grid()
        plt.savefig(self.out_stns_avail_fig, dpi=300, bbox='tight')
        plt.close()

        out_index = self.avail_nrst_stns_ser.index.union(self.avail_nrst_stns_orig_ser.index)
        fin_df = DataFrame(index=out_index, columns=['original', 'infill'])
        fin_df['original'] = self.avail_nrst_stns_orig_ser
        fin_df['infill'] = self.avail_nrst_stns_ser
        fin_df.to_csv(self.out_stns_avail_file,
                      sep=str(self.sep), encoding='utf-8')
        return

    def _full_tb(self, sys_info):
        exc_type, exc_value, exc_traceback = sys_info
        tb_fmt_obj = format_exception(exc_type, exc_value, exc_traceback)
        for trc in tb_fmt_obj:
            print trc
        raise RuntimeError
        return

    def _infill(self, infill_dates):
        try:
            out_conf_df = DataFrame(index=infill_dates, columns=self.conf_ser.index)
            _probs_str = 'probs'
            _norm_str = 'norm_vals'
            _vals_str = 'vals'

            pre_avail_stns = [self.curr_infill_stn]

            if self.plot_diag_flag:
                ax_1 = plt.subplot(111)
                ax_2 = ax_1.twiny()

            for infill_date in infill_dates:
                if not isnan(self.in_var_df.loc[infill_date, self.curr_infill_stn]):
                    if not self.compare_infill_flag:
                        continue

                if self.infill_type == u'precipitation':
                    if np_all(self.in_var_df.loc[infill_date].dropna().values == self.var_le_trs):
                        out_conf_df.loc[infill_date] = self.var_le_trs
                        continue

                date_pref = '%0.4d%0.2d%0.2d%0.2d%0.2d' % (infill_date.year,
                                                           infill_date.month,
                                                           infill_date.day,
                                                           infill_date.hour,
                                                           infill_date.minute)

                # see which stns are available at the given step
                avail_cols = self.in_var_df.loc[infill_date,
                                                self.curr_nrst_stns].dropna().index
                if self.take_min_stns_flag:
                    avail_cols = avail_cols[:self.n_nrst_stns_min]

                if avail_cols.shape[0] < self.n_nrst_stns_min:
                    if (not self.force_infill_flag) or (avail_cols.shape[0] < 1):
                        continue

                if self.plot_diag_flag:
                    ax_1 = plt.subplot(111)
                    ax_2 = ax_1.twiny()

                if pre_avail_stns[1:] != avail_cols.tolist():
                    curr_val_cdf_ftns_dict = {}
                    curr_py_zeros_dict = {}
                    curr_py_dels_dict = {}
                    # put all the station values and the infill station in a new df
                        # drop cols with too few values
                    curr_var_df = self.in_var_df[[self.curr_infill_stn] + list(avail_cols)].copy()
                    curr_var_df.dropna(axis=0, inplace=True)
                    curr_drop_le_count_cols = curr_var_df.columns[curr_var_df.count() < self.min_valid_vals]
                    curr_var_df.drop(labels=curr_drop_le_count_cols, axis=1, inplace=True)
                    try:
                        avail_cols = curr_var_df.columns.drop(self.curr_infill_stn)
                    except:
                        pre_avail_stns = ['x', 'y'] # a dummy
                        continue

                    if (curr_var_df.shape[1] - 1) < self.n_nrst_stns_min:
                        if (not self.force_infill_flag) or ((curr_var_df.shape[1] - 1) < 1):
                            pre_avail_stns = ['x', 'y'] # a dummy
                            continue

                    # create probability and standard normal value dfs for this time step
                    probs_df = DataFrame(index=curr_var_df.index, columns=curr_var_df.columns)
                    norms_df = DataFrame(index=curr_var_df.index, columns=curr_var_df.columns)

                    # the vector to hold the standard normal values of the discharge at
                        # neighbouring stations. These are calculated from an interpolation
                        # function that is based on common values
                    u_t = full((curr_var_df.shape[1] - 1), nan)

                    if self.infill_type == u'precipitation':
                        py_del = nan
                        py_zero = nan

                    assert curr_var_df.columns.shape[0] > 1, '\'curr_var_df\' has no neighboring stations in it!'

                    for i, col in enumerate(curr_var_df.columns):
                        # CDFs
                        var_ser = curr_var_df[col].copy()

                        if self.infill_type == u'precipitation':
                            # get probability of zero and below threshold
                            zero_idxs = where(var_ser.values == self.var_le_trs)[0]
                            zero_prob = float(zero_idxs.shape[0]) / var_ser.shape[0]
                            thresh_idxs = where(logical_and(var_ser.values > self.var_le_trs,
                                                var_ser.values <= self.var_ge_trs))[0]
                            thresh_prob = zero_prob + float(thresh_idxs.shape[0]) / var_ser.shape[0]
                            thresh_prob_orig = thresh_prob
                            thresh_prob = zero_prob + (0.5 * (thresh_prob - zero_prob))

                            assert zero_prob <= thresh_prob, '\'zero_prob\' > \'thresh_prob\'!'

                            curr_py_zeros_dict[col] = zero_prob * 0.5
                            curr_py_dels_dict[col] = thresh_prob

                            var_ser_copy = var_ser.copy()
                            var_ser_copy[var_ser_copy <= self.var_ge_trs] = nan

                            probs_ser = var_ser_copy.rank() / (var_ser_copy.count() + 1.)
                            probs_ser = thresh_prob_orig + ((1.0 - thresh_prob_orig) * probs_ser)

                            probs_ser.iloc[zero_idxs] = (0.5 * zero_prob)

                            probs_ser.iloc[thresh_idxs] = thresh_prob

                            assert thresh_prob <= probs_ser.max(), '\'thresh_prob\' > \'probs_ser.max()\'!'
                        else:
                            probs_ser = var_ser.rank() / (var_ser.count() + 1.)

                        assert np_all(isfinite(probs_ser.values)), 'NaNs in \'probs_ser\'!'

                        probs_df[col] = probs_ser
                        norms_df[col] = norm_ppf_py_arr(probs_ser.values)

                        if (col == self.curr_infill_stn) and \
                            (self.infill_type == u'precipitation'):
                            py_del = thresh_prob
                            py_zero = zero_prob * 0.5

                        curr_val_cdf_df = DataFrame(index=curr_var_df.index,
                                                    columns=[_probs_str, _vals_str])
                        curr_val_cdf_df[_probs_str] = probs_df[col].copy()
                        curr_val_cdf_df[_vals_str] = var_ser.copy()

                        curr_val_cdf_df.sort_values(by=_vals_str, inplace=True)

                        curr_max_prob = curr_val_cdf_df[_probs_str].values.max()
                        curr_min_prob = curr_val_cdf_df[_probs_str].values.min()

                        curr_val_cdf_ftn = interp1d(curr_val_cdf_df[_vals_str].values,
                                                    curr_val_cdf_df[_probs_str].values,
                                                    bounds_error=False,
                                                    fill_value=(curr_min_prob,
                                                                curr_max_prob))

                        curr_val_cdf_ftns_dict[col] = curr_val_cdf_ftn

                        if self.plot_diag_flag:
                            curr_norm_ppf_df = DataFrame(index=curr_var_df.index,
                                                         columns=[_probs_str, _norm_str])
                            curr_norm_ppf_df[_probs_str] = probs_df[col].copy()
                            curr_norm_ppf_df[_norm_str] = norms_df[col].copy()

                            curr_norm_ppf_df.sort_values(by=_probs_str, inplace=True)

                            # plot currently used stns CDFs
                            lg_1 = ax_1.scatter(curr_val_cdf_df[_vals_str].values,
                                                curr_val_cdf_df[_probs_str].values,
                                                label='CDF variable',
                                                alpha=0.5, color='r', s=0.5)

                            lg_2 = ax_2.scatter(curr_norm_ppf_df[_norm_str].values,
                                                curr_norm_ppf_df[_probs_str].values,
                                                label='CDF ui',
                                                alpha=0.9, color='b', s=0.5)

                            lgs = (lg_1, lg_2)
                            labs = [l.get_label() for l in lgs]
                            ax_1.legend(lgs, labs, loc=4, framealpha=0.5)

                            ax_1.grid()

                            ax_1.set_xlabel('variable x')
                            ax_2.set_xlabel('transformed variable x (ui)')
                            ax_1.set_ylabel('probability')
                            ax_1.set_ylim(0, 1)
                            ax_2.set_ylim(0, 1)

                            if self.infill_type == u'precipitation':
                                plt.suptitle('Actual and normalized value CDFs (n=%d)\n stn: %s, date: %s\npy_zero: %0.2f, py_del: %0.2f' % \
                                (curr_val_cdf_df.shape[0], col, date_pref, zero_prob, thresh_prob))
                            else:
                                plt.suptitle('Actual and normalized value CDFs (n=%d)\n stn: %s, date: %s' % \
                                (curr_val_cdf_df.shape[0], col, date_pref))

                            plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.8)

                            out_cdf_fig_loc = os_join(self.stn_step_cdfs_dir,
                                                           'CDF_%s_%s.%s' % (date_pref, col, self.out_fig_fmt))
                            plt.savefig(out_cdf_fig_loc, dpi=self.out_fig_dpi,
                                        bbox='tight')
                            ax_1.cla()
                            ax_2.cla()

                    # get corrs
                    full_corrs_arr = fill_correl_mat(norms_df.values)

                    # get vals from copulas
                    norm_cov_mat = full_corrs_arr[1:, 1:]
                    inv_norm_cov_mat = linalg.inv(norm_cov_mat)
                    cov_vec = full_corrs_arr[1:, 0]

                    siq_sq_v = var(norms_df[self.curr_infill_stn].values)
                    sig_sq_t = siq_sq_v - matmul(cov_vec.T, matmul(inv_norm_cov_mat, cov_vec))
    #                sig_sq_t = 1.0 - matmul(cov_vec.T, matmul(inv_norm_cov_mat, cov_vec))

                    # adjust if less than zero, this happens due to the transformation
                        # that we use while making cdfs
                        # not sure if it is the right thing to do
                    if sig_sq_t < 0:
                        sig_sq_t = 1 - matmul(cov_vec.T, matmul(inv_norm_cov_mat, cov_vec))
                    if sig_sq_t <= 0:
                        print '\'sig_sq_t (%0.6f)\' is still less than zero!' % sig_sq_t
                        print '\'infill_date\':', infill_date
                        print '\'cov_vec\':\n', cov_vec
                        print '\n'
                        continue

                    assert sig_sq_t > 0, '\'sig_sq_t (%0.2f)\' <= zero!' % sig_sq_t

                    # back transform
                    curr_max_var_val = curr_var_df[self.curr_infill_stn].max()
                    curr_min_var_val = curr_var_df[self.curr_infill_stn].min()

                if self.infill_type == u'precipitation':
                    assert not isnan(py_zero), '\'py_zero\' is nan!'
                    assert not isnan(py_del), '\'py_del\' is nan!'

                pre_avail_stns = [self.curr_infill_stn] + avail_cols.tolist()
                assert len(curr_val_cdf_ftns_dict.keys()) == len(avail_cols) + 1, \
                ('\'curr_val_cdf_ftns_dict\' has incorrect number of keys!', curr_val_cdf_ftns_dict, avail_cols)

                for i, col in enumerate(curr_var_df.columns):
                    # get u_t values or the interp ftns in case of infill_stn
                    if i > 0:
                        _curr_var_val = self.in_var_df.loc[infill_date, col]

                        if (self.infill_type == u'precipitation'):
                            if _curr_var_val == self.var_le_trs:
                                values_arr = self.in_var_df.loc[infill_date, curr_var_df.columns[1:]].dropna().values
                                if len(values_arr) > 0:
                                    n_wet = (values_arr > self.var_le_trs).sum()
                                    wt = n_wet / float(values_arr.shape[0])
                                else:
                                    wt = 0.0
                                u_t[i - 1] = norm_ppf_py(curr_py_zeros_dict[col] * (1.0 + wt))
                            elif (_curr_var_val > self.var_le_trs) and (_curr_var_val <= self.var_ge_trs):
                                u_t[i - 1] = norm_ppf_py(curr_py_dels_dict[col])
                            else:
                                u_t[i - 1] = norm_ppf_py(curr_val_cdf_ftns_dict[col](_curr_var_val))
                        else:
                            u_t[i - 1] = norm_ppf_py(curr_val_cdf_ftns_dict[col](_curr_var_val))
                    else:
                        val_cdf_ftn = curr_val_cdf_ftns_dict[col]

                assert np_all(isfinite(u_t)), 'NaNs in \'u_t\'!'

                mu_t = matmul(cov_vec.T, matmul(inv_norm_cov_mat, u_t))

                if self.infill_type == u'precipitation':
                    if curr_max_var_val > self.var_ge_trs:
                        val_arr = linspace(self.var_ge_trs, curr_max_var_val, self.n_discret)
                        val_arr = append(linspace(self.var_le_trs, self.var_ge_trs, \
                                         self.ge_le_trs_n, endpoint=False), val_arr)
                    else:
                        val_arr = linspace(curr_min_var_val, curr_max_var_val, self.n_discret)

                    gy_arr = full(val_arr.shape, nan)

                    for i, val in enumerate(val_arr):
                        if val > self.var_ge_trs:
                            gy_arr[i] = norm_cdf_py((norm_ppf_py(val_cdf_ftn(val)) - mu_t) / sig_sq_t**0.5)
                        elif (val > self.var_le_trs) and (val <= self.var_ge_trs):
                            gy_arr[i] = norm_cdf_py((norm_ppf_py(py_del) - mu_t) / sig_sq_t**0.5)
                        else:
                            values_arr = self.in_var_df.loc[infill_date, curr_var_df.columns[1:]].dropna().values
                            if len(values_arr) > 0:
                                n_wet = (values_arr > self.var_le_trs).sum()
                                wt = n_wet / float(values_arr.shape[0])
                            else:
                                wt = 0.0
                            gy_arr[i] = norm_cdf_py((norm_ppf_py(py_zero * (1.0 + wt)) - mu_t) / sig_sq_t**0.5)

                        assert not isnan(gy_arr[i]), '\'gy\' is nan (val: %0.2e)!' % val

                    # all of this to get rid of repeating zeros and ones in probs
                    gy_arr = np_round(gy_arr, self.n_round)
                    min_idx = 0
                    repeat_cond = False
                    for i in range(gy_arr.shape[0]):
                        if gy_arr[i] == 0.0:
                            repeat_cond = True
                            min_idx = i
                        elif gy_arr[i] > 0.0:
                            break
                    if repeat_cond:
                        gy_arr = delete(gy_arr, range(0, min_idx))
                        val_arr = delete(val_arr, range(0, min_idx))

                    assert gy_arr.shape[0] == val_arr.shape[0], 'unequal shapes of probs and vals!'

                    max_idx = 0
                    repeat_cond = False
                    for i in reversed(range(gy_arr.shape[0])):
                        if gy_arr[i] == 1.0:
                            repeat_cond = True
                            max_idx = i
                        elif gy_arr[i] < 1.0:
                            break

                    if repeat_cond:
                        gy_arr = delete(gy_arr, range(max_idx + 1, gy_arr.shape[0]))
                        val_arr = delete(val_arr, range(max_idx + 1, val_arr.shape[0]))

                    assert gy_arr.shape[0] == val_arr.shape[0], 'unequal shapes of probs and vals!'

                    if len(gy_arr) == 1:
                        # all probs are zero, hope so
                        fin_val_ppf_ftn_adj = interp1d(linspace(0, 1.0, 10), [0]*10,
                                                       bounds_error=False,
                                                       fill_value=(self.var_le_trs, self.var_le_trs))
                    else:
                        # final plots prep
                        fin_val_ppf_ftn = interp1d(gy_arr, val_arr,
                                                   bounds_error=False,
                                                   fill_value=(self.var_le_trs, curr_max_var_val))


                        curr_min_var_val_adj, curr_max_var_val_adj = \
                        fin_val_ppf_ftn([self._adj_prob_bounds[0], self._adj_prob_bounds[1]])

                        # do the interpolation again with adjusted bounds
                        if curr_max_var_val_adj > self.var_ge_trs:
                            val_arr_adj = linspace(self.var_ge_trs, curr_max_var_val_adj,
                                                   self.n_discret)
                            val_arr_adj = append(linspace(self.var_le_trs, self.var_ge_trs, \
                                                          self.ge_le_trs_n, endpoint=False),
                                                 val_arr_adj)
                        else:
                            val_arr_adj = linspace(curr_min_var_val_adj, curr_max_var_val_adj,
                                                   self.n_discret)

                        gy_arr_adj = full(val_arr_adj.shape, nan)
                        pdf_arr_adj = gy_arr_adj.copy()

                        for i, val_adj in enumerate(val_arr_adj):
                            if val_adj > self.var_ge_trs:
                                z_scor = (norm_ppf_py(val_cdf_ftn(val_adj)) - mu_t) / sig_sq_t**0.5
                                gy_arr_adj[i] = norm_cdf_py(z_scor)
                                pdf_arr_adj[i] = norm_pdf_py(z_scor)
                            elif (val_adj > self.var_le_trs) and (val_adj <= self.var_ge_trs):
                                z_scor = (norm_ppf_py(py_del) - mu_t) / sig_sq_t**0.5
                                gy_arr_adj[i] = norm_cdf_py(z_scor)
                                pdf_arr_adj[i] = norm_pdf_py(z_scor)
                            else:
                                values_arr = self.in_var_df.loc[infill_date, curr_var_df.columns[1:]].dropna().values
                                if len(values_arr) > 0:
                                    n_wet = (values_arr > self.var_le_trs).sum()
                                    wt = n_wet / float(values_arr.shape[0])
                                else:
                                    wt = 0.0
                                z_scor = (norm_ppf_py(py_zero * (1.0 + wt)) - mu_t) / sig_sq_t**0.5
                                gy_arr_adj[i] = norm_cdf_py(z_scor)
                                pdf_arr_adj[i] = norm_pdf_py(z_scor)

                            assert not isnan(gy_arr_adj[i]), '\'gy\' is nan (val: %0.2e)!' % val_adj
                            assert not isnan(pdf_arr_adj[i]), '\'pdf\' is nan (val: %0.2e)!' % val_adj

                        gy_arr_adj = np_round(gy_arr_adj, self.n_round)
                        pdf_arr_adj = np_round(pdf_arr_adj, self.n_round)
                        min_idx = 0
                        repeat_cond = False
                        for i in range(gy_arr_adj.shape[0]):
                            if gy_arr_adj[i] == 0.0:
                                repeat_cond = True
                                min_idx = i
                            elif gy_arr_adj[i] > 0.0:
                                break

                        if repeat_cond:
                            gy_arr_adj = delete(gy_arr_adj, range(0, min_idx))
                            pdf_arr_adj = delete(pdf_arr_adj, range(0, min_idx))
                            val_arr_adj = delete(val_arr_adj, range(0, min_idx))

                        assert gy_arr_adj.shape[0] == val_arr_adj.shape[0], 'unequal shapes of probs and vals!'
                        assert pdf_arr_adj.shape[0] == val_arr_adj.shape[0], 'unequal shapes of densities and vals!'

                        max_idx = 0
                        repeat_cond = False
                        for i in reversed(range(gy_arr_adj.shape[0])):
                            if gy_arr_adj[i] == 1.0:
                                repeat_cond = True
                                max_idx = i
                            elif gy_arr_adj[i] < 1.0:
                                break

                        if repeat_cond:
                            gy_arr_adj = delete(gy_arr_adj, range(max_idx + 1, gy_arr_adj.shape[0]))
                            pdf_arr_adj = delete(pdf_arr_adj, range(max_idx + 1, pdf_arr_adj.shape[0]))
                            val_arr_adj = delete(val_arr_adj, range(max_idx + 1, val_arr_adj.shape[0]))

                        assert gy_arr_adj.shape[0] == val_arr_adj.shape[0], 'unequal shapes of probs and vals!'
                        assert pdf_arr_adj.shape[0] == val_arr_adj.shape[0], 'unequal shapes of densities and vals!'

                        fin_val_ppf_ftn_adj = interp1d(gy_arr_adj, val_arr_adj,
                                                       bounds_error=False,
                                                       fill_value=(self.var_le_trs, curr_max_var_val_adj))
                        fin_val_grad_ftn = interp1d(val_arr_adj, pdf_arr_adj,
                                                    bounds_error=False,
                                                    fill_value=(0, 0))

                else:
                    val_arr = linspace(curr_min_var_val, curr_max_var_val, self.n_discret)
                    gy_arr = full(val_arr.shape, nan)
                    for i, val in enumerate(val_arr):
                        gy_arr[i] = norm_cdf_py((norm_ppf_py(val_cdf_ftn(val)) - mu_t) / sig_sq_t**0.5)
                        assert not isnan(gy_arr[i]), '\'gy\' is nan (val:%0.2e)!' % val

                    fin_val_ppf_ftn = interp1d(gy_arr, val_arr,
                                               bounds_error=False,
                                               fill_value=(curr_min_var_val,
                                                           curr_max_var_val))

                    curr_min_var_val_adj, curr_max_var_val_adj = \
                    fin_val_ppf_ftn([self._adj_prob_bounds[0], self._adj_prob_bounds[1]])

                    # do the interpolation again with adjusted bounds
                    val_arr_adj = linspace(curr_min_var_val_adj, curr_max_var_val_adj,
                                           self.n_discret)
                    gy_arr_adj = full(val_arr_adj.shape, nan)
                    pdf_arr_adj = gy_arr_adj.copy()

                    for i, val_adj in enumerate(val_arr_adj):
                        z_scor = (norm_ppf_py(val_cdf_ftn(val_adj)) - mu_t) / sig_sq_t**0.5
                        gy_arr_adj[i] = norm_cdf_py(z_scor)
                        pdf_arr_adj[i] = norm_pdf_py(z_scor)

                    fin_val_ppf_ftn_adj = interp1d(gy_arr_adj, val_arr_adj,
                                                   bounds_error=False,
                                                   fill_value=(curr_min_var_val_adj, curr_max_var_val_adj))
                    fin_val_grad_ftn = interp1d(val_arr_adj, pdf_arr_adj,
                                                bounds_error=False,
                                                fill_value=(0, 0))

                conf_probs = self.conf_ser.values
                conf_vals = fin_val_ppf_ftn_adj(conf_probs)
                out_conf_df.loc[infill_date] = np_round(conf_vals, self.n_round)

                if np_any(where(ediff1d(conf_vals) < 0, 1, 0)):
                    print 'Interpolated var_vals on %s at station: %s not in ascending order!' % \
                          (repr(infill_date), repr(self.curr_infill_stn))
                    print 'var_0.05', 'var_0.25', 'var_0.5', 'var_0.75', 'var_0.95:', conf_vals
                    print 'gy:\n', gy_arr_adj
                    print 'theoretical_var_vals:\n', val_arr_adj
                    assert False, \
                           'Interpolated var_vals on %s not in ascending order!' % \
                           repr(infill_date)

                if self.plot_step_cdf_pdf_flag:
                    # plot infill cdf
                    plt.clf()
                    plt.plot(val_arr_adj, gy_arr_adj)
                    plt.scatter(conf_vals, conf_probs)
                    if self.infill_type == u'precipitation':
                        plt.title('infill CDF\n stn: %s, date: %s\npy_zero: %0.2f, py_del: %0.2f' % \
                                 (self.curr_infill_stn, date_pref, py_zero, py_del))
                    else:
                        plt.title('infill CDF\n stn: %s, date: %s' % \
                                 (self.curr_infill_stn, date_pref))
                    plt.grid()

                    plt_texts = []
                    for i in range(conf_probs.shape[0]):
                        plt_texts.append(plt.text(conf_vals[i],
                                                  conf_probs[i],
                                                  'var_%0.2f: %0.2f' % \
                                                  (conf_probs[i],
                                                   conf_vals[i]),
                                                  va='top',
                                                  ha='left'))

                    adjust_text(plt_texts)

                    out_val_cdf_loc = os_join(self.stn_infill_cdfs_dir,
                    'infill_CDF_%s_%s.%s' % (self.curr_infill_stn, date_pref, self.out_fig_fmt))
                    plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.85)
                    plt.savefig(out_val_cdf_loc, dpi=self.out_fig_dpi,
                                bbox='tight')
                    plt.clf()

                    # plot infill pdf
                    conf_grads = fin_val_grad_ftn(conf_vals)

                    plt.plot(val_arr_adj, pdf_arr_adj)
                    plt.scatter(conf_vals, conf_grads)
                    if self.infill_type == u'precipitation':
                        plt.title('infill PDF\n stn: %s, date: %s\npy_zero: %0.2f, py_del: %0.2f' % \
                                 (self.curr_infill_stn, date_pref, py_zero, py_del))
                    else:
                        plt.title('infill PDF\n stn: %s, date: %s' % \
                                 (self.curr_infill_stn, date_pref))
                    plt.grid()

                    plt_texts = []
                    for i in range(conf_probs.shape[0]):
                        plt_texts.append(plt.text(conf_vals[i],
                                                  conf_grads[i],
                                                  'var_%0.2f: %0.2e' % \
                                                  (conf_probs[i],
                                                   conf_grads[i]),
                                                  va='top',
                                                  ha='left'))

                    adjust_text(plt_texts)

                    out_val_pdf_loc = os_join(self.stn_infill_pdfs_dir,
                    'infill_PDF_%s_%s.%s' % (self.curr_infill_stn, date_pref, self.out_fig_fmt))
                    plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.85)
                    plt.savefig(out_val_pdf_loc, dpi=self.out_fig_dpi,
                                bbox='tight')
                    plt.clf()

                if self.plot_diag_flag:
                    # plot corrs
                    tick_font_size = 3
                    n_stns = full_corrs_arr.shape[0]

                    corrs_ax = plt.subplot(111)
                    corrs_ax.matshow(full_corrs_arr, vmin=0, vmax=1,
                                     cmap=cmaps.Blues, origin='lower')
                    for s in zip(repeat(range(n_stns), n_stns),
                                 tile(range(n_stns), n_stns)):
                        corrs_ax.text(s[1], s[0],
                                      '%0.2f' % (full_corrs_arr[s[0], s[1]]),
                                      va='center', ha='center',
                                      fontsize=tick_font_size)

                    corrs_ax.set_xticks(range(0, n_stns))
                    corrs_ax.set_xticklabels(curr_var_df.columns)
                    corrs_ax.set_yticks(range(0, n_stns))
                    corrs_ax.set_yticklabels(curr_var_df.columns)

                    corrs_ax.spines['left'].set_position(('outward', 10))
                    corrs_ax.spines['right'].set_position(('outward', 10))
                    corrs_ax.spines['top'].set_position(('outward', 10))
                    corrs_ax.spines['bottom'].set_position(('outward', 10))

                    corrs_ax.tick_params(labelleft=True,
                                         labelbottom=True,
                                         labeltop=True,
                                         labelright=True)

                    plt.setp(corrs_ax.get_xticklabels(), size=tick_font_size,
                             rotation=45)
                    plt.setp(corrs_ax.get_yticklabels(), size=tick_font_size)

                    out_corrs_fig_loc = os_join(self.stn_step_corrs_dir,
                    'stn_corrs_%s.%s' % (date_pref, self.out_fig_fmt))
                    plt.savefig(out_corrs_fig_loc,
                                dpi=min(self.out_fig_dpi*4, 500), bbox='tight')
                    plt.clf()
                plt.close()
            return out_conf_df
        except:
            self._full_tb(exc_info())

if __name__ == '__main__':
    pass