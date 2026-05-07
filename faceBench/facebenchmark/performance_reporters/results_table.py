import itertools
import numpy as np
from facebenchmark import BaseReporter


class ResultsTable(BaseReporter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default options and update with any provided options.
        default_opts = {
            "overall_statistic": "mean",      # Overall statistic across subjects ("mean" or "median")
            "persubj_statistic": "mean",      # Per-subject statistic ("mean" or "median")
            "layout": "rec_methods/err_computers",  # Table layout: "rec_methods/err_computers" or reversed
            "show_row_headers": True,         # Whether to print row headers
            "show_col_headers": True,         # Whether to print column headers
            "round_digits": 2,                # Number of digits to round the errors
            "divide_error_by": 1.0,           # Normalization factor for errors
            "delimiter": "\t"                 # Delimiter for printing the table
        }
        self.opts = {**default_opts, **self.opts}

    def produce(self):
        # Compute per-vertex errors for all subjects and methods.
        self.compute_all_pervertex_errors()

        # Dictionary to store overall errors for each reconstruction method and error computer.
        all_errs = {}

        # Extract model name from the first reconstruction method.
        # (We assume that all rec_methods belong to the same morphable model.)
        mm0 = self.rec_methods[0].split('/')[0]
        # Headers for reconstruction methods are the part after '/'.
        headers_rec_method = [r.split('/')[1] for r in self.rec_methods]
        # Headers for error computers are their names.
        headers_ec = [ec.name for ec in self.error_computers[mm0]]
        # We assume the same model name for all rec_methods.
        mm_name = mm0

        # Loop over each reconstruction method.
        for rec_method_full in self.rec_methods:
            rec_method_name = rec_method_full.split('/')[1]
            all_errs[rec_method_name] = {}
            # Loop over each error computer for the given morphable model.
            for ec in self.error_computers[mm_name]:
                subj_errs = []
                # Loop over all subject IDs.
                for subj_id in self.subj_ids:
                    ptwise_err = self.compute_pervertex_error(subj_id, rec_method_full, ec)
                    # Aggregate per-vertex error per subject according to the specified statistic.
                    if self.opts["persubj_statistic"] == "mean":
                        subj_errs.append(ptwise_err.mean())
                    elif self.opts["persubj_statistic"] == "median":
                        subj_errs.append(np.median(ptwise_err))
                # Aggregate across subjects to get the overall error.
                if self.opts["overall_statistic"] == "mean":
                    all_errs[rec_method_name][ec.name] = np.mean(subj_errs)
                elif self.opts["overall_statistic"] == "median":
                    all_errs[rec_method_name][ec.name] = np.median(subj_errs)

        # Determine the number of reconstruction methods and error computers.
        Nrec_methods = len(headers_rec_method)
        Nerr_computers = len(headers_ec)

        # Initialize an error table (2D NumPy array) to store the overall errors.
        err_table = np.zeros((Nrec_methods, Nerr_computers))

        # Fill the error table using the values in the dictionary.
        for i, j in itertools.product(range(Nrec_methods), range(Nerr_computers)):
            err_table[i, j] = all_errs[headers_rec_method[i]][headers_ec[j]]

        # Format the reconstruction method headers by padding them to the same length.
        max_len = max(len(x) for x in headers_rec_method)
        headers_rec_method = [f'{x:{max_len}}' for x in headers_rec_method]

        # Normalize and round the error table.
        err_table /= self.opts['divide_error_by']
        err_table = err_table.round(self.opts['round_digits'])
        headers_columns = headers_ec
        headers_rows = headers_rec_method

        # If the layout is "err_computers/rec_methods", transpose the table.
        if self.opts['layout'] == "err_computers/rec_methods":
            err_table = err_table.T
            headers_rows = headers_ec
            headers_columns = headers_rec_method

        # Add row headers if requested.
        if self.opts['show_row_headers']:
            # Prepend column headers as the first row.
            err_table = np.concatenate(([headers_columns], err_table))
            headers_rows = ['\t'] + headers_rows

        # Add column headers if requested.
        if self.opts['show_col_headers']:
            # Prepend row headers as the first column.
            err_table = np.concatenate((np.array([headers_rows]).reshape(-1, 1), err_table), axis=1)

        # Print the error table row by row, joining entries using the specified delimiter.
        for row in err_table:
            # Ensure all entries are strings.
            print(self.opts['delimiter'].join(map(str, row)))