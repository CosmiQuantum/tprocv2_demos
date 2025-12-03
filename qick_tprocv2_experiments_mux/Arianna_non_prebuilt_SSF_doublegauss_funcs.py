import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import datetime
import os

class non_prebuilt_ssf_analysis_class:
    def gauss_pdf(self, x: np.ndarray, mu: float, sigma: float):
        """
        x = the values where you want to evaluate the Gaussian curve
        mu = gaussian mean
        sigma = gaussian sigma (standard deviation)

        returns gaussian probability density function (PDF)
        """
        sigma = max(float(sigma), 1e-12)
        z = (x - mu) / sigma
        return np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sigma)


    def nll_2gauss_single(self, params: np.ndarray, x: np.ndarray):
        """
        Calculates negative log-likelihood (NLL) of a dataset (x) assuming
        that the dataset comes from a mixture of two Gaussian distributions.

        mu1, sig1 = mean and standard deviation of the first Gaussian
        mu2, sig2 = mean and standard deviation of the second Gaussian

        p1 = probability density of x under Gaussian 1
        p2 = probability density of x under Gaussian 2
        mix = weighted sum of the PDFs of both Gaussians

        Not all data points come from each Gaussian equally. Weights represent that proportion.
        w1 = weight of the first Gaussian in the mixture (will be between 0 and 1)
        The second Gaussian has weight 1 - w1

        Since we will be using a minimizing function, minimizing -log(L) is the same as maximizing log(L). That is why we add
        the negative sign at the end.
        """
        mu1, sig1, mu2, sig2, w1 = params
        p1 = self.gauss_pdf(x, mu1, sig1)
        p2 = self.gauss_pdf(x, mu2, sig2)
        mix = w1 * p1 + (1.0 - w1) * p2  # gaussian mixture func (sum of both contributions)
        return -np.sum(np.log(np.clip(mix, 1e-300, None)))  # np.clip here replaces zero or neg values with a very small number


    def safe_std(self, a: np.ndarray, floor: float = 1e-6):
        """
        "safe" standard deviation calculator that avoids returning zero or extremely tiny values.

        If the array has 1 or fewer elements, it can’t have a mathematically meaningful std, so it
        just returns a provided floor value (or default = 1e-6).

        This prevents issues in the Gaussian fitting where sigma near-zero could cause division by zero or explosive behavior.
        """
        a = np.asarray(a)
        if a.size <= 1:
            return floor
        s = float(np.std(a))
        return max(s, floor)


    def fit_double_gaussian_on_ground_Arianna(self, ig_new: np.ndarray, numbins: int = 55, lo=None, hi=None):
        """
        Fits a 2-Gaussian mixture to ig_new (ground state ssf data).
        Returns params and pre-scaled curves for plotting over the ssf g-state histogram.

        ig_new must be a 1D array.
        """
        x = ig_new[np.isfinite(ig_new)]  # drops NaN or infinite values
        if x.size < 2:
            raise ValueError("Need at least 2 data points")
        X = x.reshape(-1,1)  # scikit-learn KMeans expects the input data to be 2D, so we reshape the 1D array into a 2D array

        # KMeans clustering method, which is what sklearn.mixture.GaussianMixture uses
        # finds centroids that minimize the distance to the points around it
        # n_clusters = 2 since we want two clusters (one for the ground Gaussian and one for the excited/leakage Gaussian)
        # KMeans++ method makes several trials at each sampling step and choosing the best centroid among them
        # KMeans++ picks the initial centroids by spreading them out instead of going with random guesses
        # n_init=10 is the num of trials
        km = KMeans(n_clusters=2, init="k-means++", n_init=10)
        km.fit(X)  # Runs KMeans on our reshaped ground-state data
        centers = km.cluster_centers_.flatten()  # gives us coordinates of the two centroids found and we use flatten to turn the results from [[val1], [val2]] to [val1, val2].
        labels = km.labels_  # gives an array the same length as our data X. Each entry is either 0 or 1 ( the index of the cluster each point was assigned to).

        # np.argsort(centers) gives us the indices of centers in ascending order
        # This ensures that the first Gaussian (mu1) will always corresponds to the smaller mean and the second (mu2) to the larger mean.
        order = np.argsort(centers)

        # reorder so smaller mean comes first
        mu_small, mu_large = centers[order]

        # masks for the two clusters
        # mask_small is a Boolean array that is True for points in the smaller cluster and False otherwise.
        # mask_large is the opposite.
        mask_small = (labels == order[0])  # order[0] is the index of the smaller-mean cluster.
        mask_large = (labels == order[1])  # order[1] is the index of the larger-mean cluster.

        # corresponding standard deviations
        # safe_std that avoids returning zero or extremely tiny values
        sigma_small = self.safe_std(x[mask_small])
        sigma_large = self.safe_std(x[mask_large])

        # weight of component 1 (smaller mean)
        # In NumPy, True = 1 and False = 0. So a boolean array such as [True, True, False, False, True] is equivalent to [1, 1, 0, 0, 1].
        # Using this approach, we can calculate the fraction of the total data points that belong to the smaller-mean cluster (true ground state cluster).
        weight_small = float(np.mean(mask_small))  # therefore weight_large = 1 - weight_small

        # Now we have clean initial guesses for our double Gaussian fit
        # so we do Maximum Likelihood Estimation (MLE) via Negative Log-Likelihood (NLL) minimization
        # p0 is the initial parameter vector (initial guess for the optimizer)
        p0 = np.array([mu_small, sigma_small, mu_large, sigma_large, weight_small], dtype=float)

        # standard deviation chosen to be > 1e-6 so it never is zero and is also positive.
        # The weight must stay between 0 and 1 (but not equal to them) If we allowed 0 or 1 exactly,
        # one Gaussian would vanish entirely and the optimizer could fail.
        bounds = [(None, None), (1e-6, None), (None, None), (1e-6, None), (1e-6, 1 - 1e-6)]

        # Now we find the parameter vector: params = [mu_1, sigma_1, mu_2, sigma_2, weight_1] that makes the given function as small as possible
        # nll_2gauss_single: the function we’re minimizing (negative log-likelihood for a 2-Gaussian mixture).
        # Smaller NLL = better fit.
        # Stops after at most 300 steps (maxiter)
        # If the relative change in the function is smaller than ftol, the optimizer assumes it has converged and stops
        res = minimize(self.nll_2gauss_single, p0, args=(x,), bounds=bounds,
                       options={"maxiter": 300, "ftol": 0.001})  # GMM uses 0.001, I originally chose 1e-7
        if not res.success:
            raise RuntimeError(f"Double-Gaussian fit failed: {res.message}")

        # if succesful, we can extract our parameter vector components:
        mu1, sig1, mu2, sig2, w1 = res.x

        # enforce ordering we established earlier: component 1 = smaller mean.
        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
            sig1, sig2 = sig2, sig1
            w1 = 1.0 - w1

        # For plotting:
        # plotted curve should start a bit before our smallest data point and after our max datapoint,
        # so the Gaussian tails don’t look cut off. User can set this manually but this is the default:
        if lo is None: lo = float(np.min(x)) - 0.05 * (np.max(x) - np.min(x))
        if hi is None: hi = float(np.max(x)) + 0.05 * (np.max(x) - np.min(x))
        xvals = np.linspace(lo, hi, 1000)

        # gauss_pdf gives the probability density function of a Gaussian at each point in xvals. we do this for both gaussians.
        pdf1 = self.gauss_pdf(xvals, mu1, sig1)  # the Gaussian centered at mu1 with spread sig1
        pdf2 = self.gauss_pdf(xvals, mu2, sig2)  # the Gaussian centered at mu2 with spread sig2
        pdf_sum = w1 * pdf1 + (1.0 - w1) * pdf2  # combines the two Gaussians into a 'mixture model'
        # Note: these are not scaled to the data yet, they’re just normalized PDFs (area under curve = 1)

        # A PDF integrates to 1, but a histogram integrates to N (total number of counts)
        # We need to rescale the PDFs to match histogram heights
        N = x.size  # number of data points in the dataset
        bin_width = (hi - lo) / float(numbins)  # width of each histogram bin
        ground_gaussian = N * w1 * pdf1 * bin_width  # this scales pdf1 so that its area = number of points in cluster 1, which is approx (N) x (w1). So this gives the expected histogram height
        excited_gaussian = N * (1.0 - w1) * pdf2 * bin_width  # we do the same for pdf2
        sum_gaussians = N * pdf_sum * bin_width  # and same for the mixture pdf

        # Now we determine the thermal population threshold: midpoint of the two gaussian means
        threshold = 0.5 * (mu1 + mu2)  # we use this threshold to find the leakage (e-state) population

        ground_data = ig_new[ig_new <= threshold]  # anything to the left of the threshold is g-state data
        excited_data = ig_new[ig_new > threshold]  # anything to the right of the threshold is leakage data

        ground_state_population = len(ground_data) / len(ig_new)  # ground state population
        excited_state_population_leakage = len(excited_data) / len(ig_new)  # excited state (leakage) population

        # chi^2 equals twice the neg log likelyhood
        # We feed the optimal paramters into nll_2gauss_single
        # x = the ig_new data
        chisq = 2 * self.nll_2gauss_single(([mu1, sig1, mu2, sig2, w1]), x)

        # parameters of the gaussian fits that can be used later on for plotting
        params = {
            "mu": (mu1, mu2),
            "sigma": (sig1, sig2),
            "weight": (w1, 1.0 - w1),  # w2 = 1-w1
            "threshold": (threshold),
            "chisq": (chisq)}

        return params, xvals, ground_gaussian, excited_gaussian, sum_gaussians, ground_state_population, excited_state_population_leakage, lo, hi

    def plot_Ariannas_doublegauss_func(self, ig_new, ie_new, params, numbins = 55, save_figs_path = "", title_ext = "", filename_ext = ""):
        t0 = time.perf_counter()
        lo = float(np.min(ig_new))
        hi = float(np.max(ie_new))

        params, xvals, ground_gaussian, excited_gaussian, sum_gaussians, Pg, Pe, lo, hi = self.fit_double_gaussian_on_ground_Arianna(ig_new, numbins=numbins, lo=lo, hi=hi)

        # extract params
        # note: component 1 = true ground state and component 2 = (e=state) leakage
        mu1, mu2 = params["mu"]
        sig1, sig2 = params["sigma"]
        w1,  w2 = params["weight"]
        thresh = params["threshold"]
        chisq = params["chisq"]

        fig, ax = plt.subplots(figsize=(6,4))
        counts, edges, patches = ax.hist(ig_new, bins=numbins, range=(lo, hi), alpha=0.5, color='blue', edgecolor='none', label='True g-state')

        # compute bin centers from edges
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # coloring thermal pop. data in red
        for c, p in zip(bin_centers, patches):
            if c >= thresh:
                p.set_facecolor('red')

        # plotting Gaussians and their sum
        ax.plot(xvals, ground_gaussian,  '--', linewidth=1.6, label='g Gaussian', color='blue', alpha=1.0)
        ax.plot(xvals, excited_gaussian,  '--', linewidth=1.6, label='e-leakage Gaussian', color='red',  alpha=1.0)
        # ax.plot(xvals, ysum, '-',  linewidth=2.0, label='Sum (double-Gauss)', color='black')

        # plotting the thermal pop. threshold
        ax.axvline(thresh, color='k', linestyle=':', linewidth=1.4, label=f"threshold = {thresh:.3f}")

        # Shade to the RIGHT of threshold under the sum curve
        # ax.fill_between(
        #     xvals, sum_gaussians, 0.0,
        #     where=(xvals >= thresh),
        #     interpolate=True,
        #     color='red', alpha=0.5, label='thermal pop.')

        ax.set_xlabel("Rotated I (a.u.)")
        ax.set_ylabel("Counts")
        ax.set_title(f"{title_ext} g-state double gauss fit, not-prebuilt fitting")
        ax.legend()
        plt.tight_layout()

        os.makedirs(save_figs_path, exist_ok=True)
        fname = os.path.join(
            save_figs_path,
            f"{filename_ext}_SSFgaussfits_notprebuilt_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")

        fig.savefig(fname)

        t1 = time.perf_counter()
        print(f"My function took {t1 - t0:.4f} seconds")
        print("Ground state population: ", Pg)
        print("Excited state population: ", Pe)
        print("Chi-squared val:", chisq)

    def nll_1gauss_single(self, params: np.ndarray, x: np.ndarray):
        """
        Negative log-likelihood for a single Gaussian.
        params = [mu, sigma]
        """
        mu, sig = params
        sig = max(float(sig), 1e-12)
        p = self.gauss_pdf(x, mu, sig)
        return -np.sum(np.log(np.clip(p, 1e-300, None)))

    def fit_single_gaussian_on_ground_Arianna(self, ig_new: np.ndarray, numbins: int = 55, lo=None, hi=None):
        """
        Fits a single Gaussian to ig_new (ground state ssf data).
        Returns params and a pre-scaled curve for plotting over the SSF histogram.

        ig_new must be a 1D array.
        """
        x = ig_new[np.isfinite(ig_new)]  # drops NaN or infinite values
        if x.size < 2:
            raise ValueError("Need at least 2 data points")

        # ----- initial guesses -----
        mu0 = float(np.mean(x))
        sig0 = self.safe_std(x)

        p0 = np.array([mu0, sig0], dtype=float)
        bounds = [(None, None), (1e-6, None)]  # sigma > 0

        # ----- minimize single-Gaussian NLL -----
        res = minimize(
            self.nll_1gauss_single,
            p0,
            args=(x,),
            bounds=bounds,
            options={"maxiter": 300, "ftol": 0.001},
        )
        if not res.success:
            raise RuntimeError(f"Single-Gaussian fit failed: {res.message}")

        mu, sig = res.x
        nll1 = res.fun  # best NLL for 1-Gaussian

        # ----- x-range for plotting -----
        if lo is None:
            lo = float(np.min(x)) - 0.05 * (np.max(x) - np.min(x))
        if hi is None:
            hi = float(np.max(x)) + 0.05 * (np.max(x) - np.min(x))
        xvals = np.linspace(lo, hi, 1000)

        # ----- PDF and scaled curve -----
        pdf = self.gauss_pdf(xvals, mu, sig)
        N = x.size
        bin_width = (hi - lo) / float(numbins)
        gauss_scaled = N * pdf * bin_width  # expected histogram heights

        # chi^2-like statistic: 2 * NLL
        chisq = 2 * nll1

        params = {
            "mu": mu,
            "sigma": sig,
            "chisq": chisq,
            "nll": nll1,
        }

        return params, xvals, gauss_scaled, lo, hi
