import os
import sys
import numpy as np
import multiprocessing as mp
from glob import glob

from tqdm import tqdm

from . import ErrorComputer
from facebenchmark import compute_landmark_base_vertex_weights


class PoolParams:
    def __init__(self, reporter, subj_id, rec_method_full, ec):
        # Store parameters for each processing unit (subject, reconstruction method, error computer)
        self.reporter = reporter
        self.subj_id = subj_id
        self.rec_method_full = rec_method_full
        self.ec = ec


def func(pp):
    # Function to be executed in parallel:
    # For each PoolParams object, compute the per-vertex error.
    pp.reporter.compute_pervertex_error(pp.subj_id, pp.rec_method_full, pp.ec)


class BaseReporter:
    def __init__(self, db_info, error_computer_recipes,
                 rec_methods, mms_info, data_root_dir,
                 opts={}, use_cache=True, verbosity_level=1,
                 num_processes=1, subj_ids=None):
        # Retrieve the names of the morphable models
        mm_names = list(mms_info.keys())
        self.opts = opts
        self.db_info = db_info
        self.mms_info = mms_info
        self.use_cache = use_cache
        self.num_processes = num_processes
        self.verbosity_level = verbosity_level

        self.vertex_maps = None
        # If a weight threshold is specified, compute a boolean map of vertices above the threshold.
        if 'weight_threshold' in self.opts:
            self.vertex_maps = self.compute_vertex_maps_to_report_results_on()

        # Create error computer instances for each model using the provided recipes.
        self.error_computers = {
            mm_name:
                [ErrorComputer(r, mm=self.mms_info[mm_name])
                 for r in error_computer_recipes]
            for mm_name in mm_names
        }
        self.error_computer_names = [r['name'] for r in error_computer_recipes]

        # Store reconstruction methods and the data root directory.
        self.rec_methods = rec_methods
        self.DATA_ROOT = data_root_dir

        # Count the number of subjects in the dataset.
        self.Nsubjs = self.count_dataset_subjects()
        if 'num_subjects' in self.opts:
            self.Nsubjs = self.opts['num_subjects']

        # Determine which subjects to process (those with valid data).
        self.subj_ids = self.get_subjs_to_process() if subj_ids is None else subj_ids

    def compute_all_pervertex_errors(self):
        """
        Compute per-vertex errors in parallel over all subject/rec_method/error computer combinations.
        Only the combinations without a valid cache file are processed.
        """
        # Set multiprocessing start method to 'spawn' to ensure compatibility (especially on macOS/Windows)
        mp.set_start_method('spawn', force=True)

        # Build a list of parameter sets for each combination that requires error computation.
        param_sets = [
            PoolParams(self, subj_id, rec_method_full, ec)
            for rec_method_full in self.rec_methods
            for mm_name in [rec_method_full.split('/')[0]]
            for ec in self.error_computers[mm_name]
            for subj_id in self.subj_ids
            if not os.path.exists(self.construct_cache_filepath(subj_id, rec_method_full, ec))
        ]

        # Create a pool of workers and map the computation function to each parameter set.
        with mp.Pool(processes=self.num_processes) as pool:
            # .map has a too big overhead, so we use imap_unordered
            pool.imap_unordered(func, param_sets)

    def count_dataset_subjects(self):
        """
        Count the number of subjects in the dataset by counting the available G meshes.
        """
        Gdir = os.path.join(self.DATA_ROOT, self.db_info['name'], 'Gmeshes')
        return len(glob(f'{Gdir}/*.txt'))

    def get_subjs_to_process(self):
        """
        Determine which subject IDs have valid R meshes for all reconstruction methods.
        """
        # Create a dictionary mapping each reconstruction method to a set of subject IDs that have valid data.
        ids_per_rec_method = {
            rec_method: {
                subj_id for subj_id in range(self.Nsubjs)
                if os.path.exists(self.construct_Rfilepath(subj_id, rec_method)) and
                   np.sum(np.isnan(self.get_Rdata(subj_id, rec_method)[0])) == 0
            }
            for rec_method in self.rec_methods
        }
        all_subjs = set(range(self.Nsubjs))
        # Subjects to process must have valid data for all methods.
        selected_subjs = set.intersection(*ids_per_rec_method.values())
        # Warn if some methods have missing or NaN R meshes.
        if len(selected_subjs) < len(all_subjs) and self.verbosity_level > 0:
            print("R meshes are missing or have NaN values for some methods:", file=sys.stderr)
            for rec_method in self.rec_methods:
                gap = len(all_subjs) - len(ids_per_rec_method[rec_method])
                if gap > 0:
                    plural = "meshes are" if gap > 1 else "mesh is"
                    print(f"\t{gap} R {plural} missing for {rec_method} method.", file=sys.stderr)
        return selected_subjs

    def compute_pervertex_error(self, subj_id, rec_method_full, ec):
        """
        Compute the per-vertex error for a specific subject, reconstruction method, and error computer.
        Uses cached results if available and valid.

        Parameters
        ----------
        subj_id : int
            The subject ID.
        rec_method_full : str
            The reconstruction method (formatted as "MM/Method").
        ec : object
            An error computer object with a 'compute' method.

        Returns
        -------
        err : ndarray
            The computed per-vertex error, possibly filtered by vertex_maps.
        """
        # Construct cache filepath and determine the morphable model key.
        cache_fpath = self.construct_cache_filepath(subj_id, rec_method_full, ec)
        mm_key = rec_method_full.split('/')[0]

        # Check for an existing cache.
        valid_cache = os.path.exists(cache_fpath)
        err = None

        if valid_cache:
            # Load error from cache.
            err = np.loadtxt(cache_fpath)
            # Validate the cache by checking its length and that it has no NaN entries.
            if len(err) != self.mms_info[mm_key]['Npoints'] or np.any(np.isnan(err)):
                print('Cache is invalid -- recomputing error from scratch')
                valid_cache = False

        # If no valid cache is found or caching is disabled, compute the error.
        if not valid_cache or not self.use_cache:
            # Load R mesh data and corresponding landmarks.
            R, Rlmks = self.get_Rdata(subj_id, rec_method_full)
            # Load G mesh data and corresponding landmarks.
            G, Glmks = self.get_Gdata(subj_id)
            # Optionally, scale the G data.
            if 'divide_gmeshes_by' in self.opts:
                factor = self.opts['divide_gmeshes_by']
                G /= factor
                Glmks /= factor
            try:
                # Compute the per-vertex error using the error computer.
                err = ec.compute(R, G, Rlmks, Glmks)
            except Exception as ex:
                print("Error computing per-vertex error:", ex)
                err = np.full((R.shape[0], 1), np.nan)
            # Save computed error to cache if caching is enabled.
            if self.use_cache:
                np.savetxt(cache_fpath, err)

        # If vertex_maps are defined, filter the error accordingly.
        return err if self.vertex_maps is None else err[self.vertex_maps[mm_key]]

    def compute_pervertex_error_all_computers(self, subj_id, rec_method_full):
        """
        Compute the per-vertex error for all error computer methods for a given subject and reconstruction method.
        """
        mm0 = rec_method_full.split('/')[0]
        errs = {ec.name: self.compute_pervertex_error(subj_id, rec_method_full, ec)
                for ec in self.error_computers[mm0]}
        return errs

    def construct_cache_filepath(self, subj_id, rec_method_full, ec):
        """
        Construct the full cache file path for a given subject, reconstruction method, and error computer.
        """
        db_name = self.db_info['name']
        mm_name = rec_method_full.split('/')[0]
        rec_method = rec_method_full.split('/')[1]
        mm_version = self.mms_info[mm_name]['version']
        cache_fpath = os.path.join(self.DATA_ROOT, 'cache', db_name, 'errs', mm_name, mm_version, rec_method,
                                   f'id{subj_id:04d}' + ec.name + ec.key() + '.err')
        cache_dir = os.path.dirname(cache_fpath)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return cache_fpath

    def construct_Rfilepath(self, subj_id, rec_method_full):
        """
        Construct the file path for the R (reconstruction) mesh for a given subject and reconstruction method.
        """
        db_name = self.db_info['name']
        mm_name = rec_method_full.split('/')[0]
        rec_method = rec_method_full.split('/')[1]
        mm_version = self.mms_info[mm_name]['version']
        return os.path.join(self.DATA_ROOT, db_name, 'Rmeshes', mm_name, mm_version, rec_method, f'id{subj_id:04d}.txt')

    def get_Rdata(self, subj_id, rec_method_full):
        """
        Load the R mesh and extract landmark data for a given subject and reconstruction method.
        """
        mm_name = rec_method_full.split('/')[0]
        lmk_indices = self.mms_info[mm_name]['lmk_indices']
        Rpath = self.construct_Rfilepath(subj_id, rec_method_full)
        R = np.loadtxt(Rpath)
        Rlmks = R[lmk_indices, :]
        return R, Rlmks

    def get_Gdata(self, subj_id):
        """
        Load the G mesh and landmark data for a given subject.
        """
        bname = os.path.join(self.DATA_ROOT, self.db_info['name'], 'Gmeshes', f'id{subj_id:04d}')
        Gpath = bname + '.txt'
        Glmks_path = bname + '.lmks'
        return np.loadtxt(Gpath), np.loadtxt(Glmks_path)

    def compute_vertex_maps_to_report_results_on(self):
        """
        Compute a boolean map for each morphable model indicating which vertices have weights above a threshold.
        This is used later to filter the per-vertex errors.
        """
        vertex_maps = {}
        for mm_key in self.mms_info:
            w = compute_landmark_base_vertex_weights(self.mms_info[mm_key], 'sqrt', 'mixed')
            vertex_maps[mm_key] = w > self.opts['weight_threshold']
        return vertex_maps

    def produce(self, R, G):
        raise NotImplementedError("This class is not implemented (a virtual method has been called)")


class MetaEvaluator(BaseReporter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure that we are dealing with synthetic data and that the error computer names are as expected.
        assert self.db_info['is_synthetic']
        assert len(self.error_computer_names) == 2
        assert self.error_computer_names[0].lower() == 'true'
