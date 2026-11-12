import numpy as np
import pandas as pd

class Scoring:
    def __init__(self, config):
        self.normalize_objective = config['dataset']['normalize_objective']
        self.switch_after = config['bo']['switch_after']
        self.n_iter = config['bo']['n_iter']
        self.batch_size = config['bo']['batch_size']
        self.top_perc_cov = config['scoring']['top_perc_cov']
        self.reorder_auc = config['scoring']['reorder_auc']
        self.if_auc = config['scoring']['auc']
    
    def get_score(self, results:pd.DataFrame, lookup:pd.DataFrame, objective_name, F_BEST=None):
        if not self.normalize_objective and F_BEST is None:
            raise ValueError(f"When objective is not normalized into [0,100], F_BEST needs to be specified.")
            
        if self.if_auc:
            auc = self.score_AUC(results, objective_name, F_BEST, self.switch_after, self.reorder_auc)
        else: 
            auc = None
        
        top_cov= self.top_coverage(lookup, results, objective_name, self.top_perc_cov)
        
        scores = {'AUC':auc, 'top_coverage':top_cov}
        
        return scores

    def score_AUC(self, results:pd.DataFrame, objective_name, F_BEST=100, switch_after=None, reorder_auc=True):
        """
        Convert curve into a scalar score.
        Score is normalized according to max(metric_x) and F_BEST (default=100 for yield_CumBest).
        
        switch_after: re-order yields before switch.
        NOTE switch_after=0 includes the case where init_data is not None (e.g., transfer learning). 
        """
        
        metric_y=f"{objective_name}_CumBest"
        metric_x='Num_Experiments'

        # mean-agg over all Monte Carlo runs or repetitions
        grouped = results.groupby(metric_x)[metric_y]
        xs = grouped.mean().index.values.astype(int)
        ys = grouped.mean().values
        stds = grouped.std().fillna(0.0).values

        # Ensure sorted by x
        idx_sorted = np.argsort(xs)
        xs, ys, stds = xs[idx_sorted], ys[idx_sorted], stds[idx_sorted]

        # AUC scoring
        if switch_after > 0 and reorder_auc: # re-order yields before switch
            SCORE_AUC = self._AUC_re_order(results, switch_after, objective_name, F_BEST)
        elif switch_after == 0 or not reorder_auc: # NOTE including the case where init_data is not None (transfer learning and test_metric).
            try:
                auc = np.trapz(ys, xs)
            except: # Numpy >= 2.0
                auc = np.trapezoid(ys, xs)
            SCORE_AUC =  auc / (F_BEST * xs[-1]) # higher is better
        else:
            SCORE_AUC = None

        return SCORE_AUC

    def top_coverage(self,
        lookup:pd.DataFrame, 
        results:pd.DataFrame, 
        objective_name:str,
        top_perc_list=None, 
    ):
        '''
        divided by the smaller one between k (number of all top reactions) and the number of measurements. So this value is normalized to [0%, 100%] and 100% is always reachable.
        
        NOTE implementation could be sensitive: measured_top / (n_repeat * k), while n_repeat is given by results["Iteration"].eq(0).sum().
        
        Re-written.
        
        Re-defined for batch_size=1 or larger.
        
        Re-defined for minimization.
        '''
        
        metric_y=f"{objective_name}_Measurements"
            
        if top_perc_list is None:
            top_perc_list = [0.05] # a common metric
            
        measurements = (
            results[metric_y]
            .explode(ignore_index=True)
            .dropna()
            .astype(float)
        )
        
        n_repeats = results["Iteration"].eq(0).sum() # NOTE currently rely on Iteration col.
        if n_repeats == 0:
            raise ValueError("No experiment start with Iteration == 0 was found.")
            
        SCORE_COVERAGE = []
        for top_perc in top_perc_list:
            lookup_len = len(lookup[objective_name])
            top_k = max(1, int(top_perc * lookup_len))
            
            # def: the highest possible value of top_k reactions found in one campaign of limited budget.
            denominator = min(top_k, self.n_iter*self.batch_size-self.switch_after) * n_repeats
            # NOTE effective measurements = n_iter * batch_size - switch_after, condition that every MC_run or repeat finished successfully. Before switch_after, it's random initialization stage.
            
            # NOTE logic for maximizing the objective
            threshold = lookup[objective_name].nlargest(top_k).min()
            top_k_cov = (measurements >= threshold).sum() / denominator
            # TODO: for some special benchmarks, e.g., Vapdiff, too many candidates share the same objective, so top-1% coverage > 1.0 is possible. We can hard-code this and clip it into [0,1].
            
            SCORE_COVERAGE.append(top_k_cov)
        return SCORE_COVERAGE

    def _AUC_re_order(self, df:pd.DataFrame, switch_after, objective_name, F_BEST=100):
        '''
        Deepcopy the result df, only return AUC value and don't change the original df.
        '''
        metric_y = f"{objective_name}_CumBest"
        
        metric_x='Num_Experiments'
        
        df = df.copy(deep=True)
        df = self._reorder_random_phase(df, switch_after, objective_name)
        # df.to_csv('exp.csv')
        
        # mean-agg over all Monte Carlo runs or repetitions
        grouped = df.groupby(metric_x)[metric_y]
        xs = grouped.mean().index.values.astype(int)
        ys = grouped.mean().values
        stds = grouped.std().fillna(0.0).values

        # Ensure sorted by x
        idx_sorted = np.argsort(xs)
        xs, ys, stds = xs[idx_sorted], ys[idx_sorted], stds[idx_sorted]

        try:
            auc = np.trapz(ys, xs)
        except: # Numpy >= 2.0
            auc = np.trapezoid(ys, xs)
        SCORE_AUC =  auc / (F_BEST * xs[-1]) # higher is better
        
        return SCORE_AUC

    def _reorder_random_phase(self, df, switch_after, objective_name):

        batch_size = len(df.iloc[0][f"{objective_name}_Measurements"])

        n_full_iters = switch_after // batch_size
        remainder = switch_after % batch_size

        for run_id in df["Monte_Carlo_Run"].unique():
            mask = df["Monte_Carlo_Run"] == run_id
            df_run = df[mask]

            collected = []

            # full batches
            for i in range(n_full_iters):
                collected.extend(df_run.iloc[i][f"{objective_name}_Measurements"])

            # partial batch
            if remainder > 0:
                collected.extend(df_run.iloc[n_full_iters][f"{objective_name}_Measurements"][:remainder])

            sorted_vals = sorted(collected, reverse=False) # logic for maximizing the objective

            ptr = 0

            # full batches
            for i in range(n_full_iters):
                row_idx = df_run.index[i]
                df.at[row_idx, f"{objective_name}_Measurements"] = sorted_vals[ptr:ptr+batch_size]
                ptr += batch_size

            # partial batch (only modify the first remainder)
            if remainder > 0:
                row_idx = df_run.index[n_full_iters]
                original = df.at[row_idx, f"{objective_name}_Measurements"]

                new_batch = (
                    sorted_vals[ptr:ptr+remainder] +
                    original[remainder:]   # keep the real BO part
                )
                df.at[row_idx, f"{objective_name}_Measurements"] = new_batch

            # --- update IterBest / CumBest  ---
            affected_rows = list(df_run.index[:n_full_iters + (1 if remainder > 0 else 0)])

            # logic for maximizing the objective
            iter_best = [max(df.at[idx, f"{objective_name}_Measurements"])
                            for idx in affected_rows]
            cum_best = np.maximum.accumulate(iter_best)

            for i, idx in enumerate(affected_rows):
                df.at[idx, f"{objective_name}_IterBest"] = iter_best[i]
                df.at[idx, f"{objective_name}_CumBest"] = cum_best[i]

        return df