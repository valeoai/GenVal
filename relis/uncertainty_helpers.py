import numpy as np
import torch
from torch.nn import functional as F
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score

class UncertaintyOps:
    """
    Class with all uncertainty methods.
    """

    def __init__(self):
        pass

    def compute_output_entropy(self, predictions, avg_image=True, avg_batch=True):

        """
        Given output predictions, performs softmax and computes entropy

        Params:

            predictions : torch tensor (M,K,H,W)

                The output of the model: M batch size, K
                number of classes, (H,W) image size

            avg_image : bool

                If True, averages the different pixels'
                output predictions in a single number

            avg_batch : bool

                If True, averages everything (this is like
                reduce_mean + avg_image=True)

        Returns:

            output_entropy : torch.tensor / torch.float

                The entropy associated with the prediction
                provided.

        """

        if len(predictions.shape) < 4:
            raise ValueError("Innput must be in the format (M,K,H,W)")

        output_entropy = torch.sum(-(torch.log(predictions + 1e-10) * predictions),1)

        if avg_batch:
            return torch.mean(output_entropy)
        else:
            if avg_image:
                return output_entropy.mean(-1).mean(-1)
            else:
                return output_entropy
            
    @staticmethod
    def ECE(probs, labels, num_bins=15, binning_strategy='equal_size', class_wise=False):
        '''
        Function to compute the expected callibration error.
        See: "Measuring Calibration in Deep Learning" - https://arxiv.org/abs/1904.01685
             "Calibrating Deep Neural Networks using Focal Loss" - https://arxiv.org/abs/2002.09437
             
        Code adapted from: https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py
        
        Parameters
        ----------
        probs : torch 2D array (NxC) with probabilities for each sample (rows) and class (columns).
        
        labels : torch 1D array of length N with correct class per sample.
        
        num_bins: Integer indicating the number of bins to use.
        
        binning_strategy: String indicating the binning strategy.
            Possible options are
                'equal_size': Bins divide interval [0, 1] equally spaced.
                'equal_population': Choose bins such that they have the same amount 
                                    of samples inside.
        
        class_wise: Flag to compute class-wise ECE. That is, to compute callibration for 
                    all classes, as opposed to only the top1 predictions (used by default).

        Returns
        -------
        ece : Scalar with the expected callibration error.
        '''
        
        # Filter out background class
        probs = probs[labels != 255, :]
        labels = labels[labels != 255]
        
        if not class_wise: # Compute callibration only for predicted classes.
            probs, preds = torch.max(probs, dim=1)
            accs = preds.eq(labels)
            probs = probs.double()
        
            if binning_strategy == 'equal_size':
                # Divide [0,1] interval equally
                bin_boundaries = np.linspace(0, 1, num_bins + 1)

            elif binning_strategy == 'equal_population':
                # Find bins such that they have the same amount of points
                npt = len(probs)
                bin_boundaries = np.interp(np.linspace(0, npt, num_bins + 1),
                                             np.arange(npt),
                                             np.sort(probs))

            # Define lower and upper bin edges
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = torch.zeros(1)
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Calculated |prob - acc| in each bin
                in_bin = probs.gt(bin_lower.item()) * probs.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accs[in_bin].float().mean()
                    avg_confidence_in_bin = probs[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return ece
        
        else: # Compute callibration for all classes per sample (stricter than top1 ECE).
            num_classes = probs.shape[1]
            per_class_sce = None
            
            for i in range(num_classes):
                class_confidences = probs[:, i].double()
                class_sce = torch.zeros(1)
                labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

                if binning_strategy == 'equal_size':
                    # Divide [0,1] interval equally
                    bin_boundaries = np.linspace(0, 1, num_bins + 1)

                elif binning_strategy == 'equal_population':
                    # Find bins such that they have the same amount of points
                    npt = len(class_confidences)
                    bin_boundaries = np.interp(np.linspace(0, npt, num_bins + 1),
                                                 np.arange(npt),
                                                 np.sort(class_confidences))
                    
                # Define lower and upper bin edges
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                    prop_in_bin = in_bin.float().mean()
                    if prop_in_bin.item() > 0:
                        accuracy_in_bin = labels_in_class[in_bin].float().mean()
                        avg_confidence_in_bin = class_confidences[in_bin].mean()
                        class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                if (i == 0):
                    per_class_sce = class_sce
                else:
                    per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

#             sce = torch.mean(per_class_sce)
#             return sce
            return per_class_sce
        
    @staticmethod
    def ensure_numpy(a):
        if not isinstance(a, np.ndarray): a = a.numpy()
        return a
    
    @staticmethod
    def ks_test(probs, labels, class_wise=False):
        '''
        Function to compute the Kolmogorov-Smirnov test.
        See: "Calibration of Neural Networks using Splines" - https://arxiv.org/abs/2006.12800

        Code adapted from https://github.com/kartikgupta-at-anu/spline-calibration/blob/master/cal_metrics/KS.py
        
        Parameters
        ----------
        probs : torch 2D array (NxC) with probabilities for each sample (rows) and class (columns).

        labels : torch 1D array of length N with correct class per sample.

        class_wise: Flag to compute class-wise KS-test. That is, to compute callibration for 
                    all classes, as opposed to only the top1 predictions (used by default).

        Returns
        -------
        ks_score : Scalar with the KS test score.
        '''

        # Filter out background class
        probs = probs[labels != 255, :]
        labels = labels[labels != 255]
        
        if not class_wise: # Compute callibration only for predicted classes.
            probs, preds = torch.max(probs, dim=1)
            accs = preds.eq(labels)

            scores = UncertaintyOps.ensure_numpy(probs)
            accs = UncertaintyOps.ensure_numpy(accs)

            # Sort probabilities
            order = np.argsort(scores)
            scores = scores[order]
            accs = accs[order]

            # Compute cummulative distribution funcion (CDF) for predicted scores
            nsamples = len(scores)
            integrated_accuracy = np.cumsum(accs.astype('float64')) / nsamples
            integrated_scores   = np.cumsum(scores.astype('float64')) / nsamples

            # Compute the Kolmogorov-Smirnov error
            ks_score = np.amax(np.absolute (integrated_scores - integrated_accuracy))

            return ks_score

        else: # Compute callibration for all classes per sample (stricter than top1 callibration only).
            num_classes = probs.shape[1]
            class_ks_scores = np.zeros((num_classes))

            for i in range(num_classes):
                scores = probs[:, i]
                accs = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

                scores = UncertaintyOps.ensure_numpy(scores)
                accs = UncertaintyOps.ensure_numpy(accs)

                # Sort probabilities
                order = np.argsort(scores)
                scores = scores[order]
                accs = accs[order]

                # Compute cummulative distribution funcion (CDF) for predicted scores
                nsamples = len(scores)
                integrated_accuracy = np.cumsum(accs.astype('float64')) / nsamples
                integrated_scores   = np.cumsum(scores.astype('float64')) / nsamples

                # Compute the Kolmogorov-Smirnov error
                ks_score = np.amax(np.absolute (integrated_scores - integrated_accuracy))

                class_ks_scores[i] = ks_score

#             ks_score = np.mean(class_ks_scores)
#             return ks_score
            return class_ks_scores

    @staticmethod
    def logits_to_probs(logits, T=1.0):
        '''
        Normalize logits to pseudo-probs with softmax and temperature scaling.
        
        Parameters
        ----------
        logits : torch 2D array NxC with logits per sample (rows) and class (columns).

        T : Temperature parameter to be applied.

        Returns
        -------
        probs : torch 2D array NxC with pseudo-probs.
        '''
        
        probs = F.softmax(logits / T, dim=1)
        return probs
    
    
    def find_best_temperature_per_class_grid_search(self, logits, labels, num_classes, min_t=0.5,
                                                    max_t=10, step=0.01, cuda=True, device=0):
        
        # Filter out background class
        logits = logits[labels != 255, :]
        labels = labels[labels != 255].long()
        
        predicted_labels = torch.argmax(logits, dim=1)
        unique_labels = list(torch.unique(predicted_labels))
        
        temperatures = np.zeros(num_classes)
        for label_class in unique_labels:
            selected_labels = labels[predicted_labels == label_class]
            selected_logits = logits[predicted_labels == label_class]
            temperatures[label_class] = self.find_best_temperature_grid_search(selected_logits,
                                                                               selected_labels,
                                                                               step=step,
                                                                               max_t=max_t,
                                                                               min_t=min_t,
                                                                               cuda=cuda,
                                                                               device=device)
            
        # In case some class was not present in images of that cluster, take the mean temp for that class
        missing_classes = [x for x in range(num_classes) if x not in unique_labels]
        if len(missing_classes) > 0:
            temperatures[missing_classes] = 1.0 # Assign a temperature of 1 to missing classes
                                                # could use mean temp of present classes but there is
                                                # a lot of variability so it might end up being worse.
        return temperatures
    
    

    def find_best_temperature_grid_search(self, logits, labels, step=0.01, min_t=0.5, max_t=10, cuda=True, device=0):
        """
        Tune the temperature of the model (using the validation set).
        
        Parameters
        ----------
        logits : torch 2D array NxC with logits per sample (rows) and class (columns).

        labels : torch 1D array of length N with correct class per sample.

        Returns
        -------
        temperature : Scalar with the best temperature.
        """
        
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        # Filter out background class
        logits = logits[labels != 255, :].double()
        labels = labels[labels != 255].long()
        
        if cuda:
            nll_criterion = nll_criterion.to(device)
            ece_criterion = ece_criterion.to(device)
            logits = logits.to(device)
            labels = labels.to(device)
        
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        
        print('Before temperature - NLL: %.3f | ECE: %.3f' % (100*before_temperature_nll, 100*before_temperature_ece))
        
        best_ece = before_temperature_ece
        
        temperatures = np.arange(min_t, max_t, step)
        best_temp = 1.0
        for t in tqdm(temperatures):
            ece = ece_criterion(logits / t, labels)
            if ece.item() < best_ece:
                best_ece = ece.item()
                best_temp = t
        
        after_temperature_nll = nll_criterion(logits / best_temp, labels).item()
        after_temperature_ece = ece_criterion(logits / best_temp, labels).item()
            
        print('After temperature scaling - NLL: %.3f | ECE: %.3f' % (100*after_temperature_nll, 100*after_temperature_ece))
        print('Optimal temperature: %.3f' % best_temp)

        return best_temp
    
    
    def prediction_rejection_ratio(self, labels, logits, metric='prob', norm_logits=False):
        # Based on https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/assessment/rejection.py

        # compute area between base_error(1-x) and the rejection curve
        # compute area between base_error(1-x) and the oracle curve
        # take the ratio
        
        # Get class probabilities
        if not norm_logits: # Logits as input
            probs = self.logits_to_probs(logits)
        else:
            probs = logits # For maskformer we compute probs directly
        
        if metric == 'prob':
            confidence, preds = torch.max(probs, dim=1) # Take as confidence the probability of the predicted class
        elif metric == 'entropy':
            probs = probs + 1e-16
            confidence = torch.sum((torch.log(probs) * probs), axis=1) # Negative entropy
            preds = torch.argmax(probs, dim=1)
        
        
        # the rejection plots needs to reject to the right the most uncertain/less confident samples
        # if uncertainty metric, high means reject, sort in ascending uncertainty;
        # if confidence metric, low means reject, sort in descending confidence
        sorted_idx = torch.argsort(confidence, descending = True)

        # reverse cumulative errors function (rev = from all to first, instead from first error to all)
        rev_cum_errors = []
        # fraction of data rejected, to compute a certain value of rev_cum_errors
        fraction_data = []

        num_samples = preds.shape[0]
        
        errors = (labels[sorted_idx] != preds[sorted_idx]).float().numpy()
        rev_cum_errors = np.cumsum(errors) / num_samples
        fraction_data = np.array([float(i + 1) / float(num_samples) * 100.0 for i in range(num_samples)])
        
        base_error = rev_cum_errors[-1] # error when all data is taken into account

        # area under the rejection curve (used later to compute area between random and rejection curve)
        auc_uns = 1.0 - auc(fraction_data / 100.0, rev_cum_errors[::-1] / 100.0)

        # random rejection baseline, it's 1 - x line "scaled" and "shifted" to pass through base error and go to 100% rejection
        random_rejection = np.asarray(
                    [base_error * (1.0 - float(i) / float(num_samples)) for i in range(num_samples)],
                    dtype=np.float32)
        # area under random rejection, should be 0.5
        auc_rnd = 1.0 - auc(fraction_data / 100.0, random_rejection / 100.0)

        # oracle curve, the oracle is assumed to commit the base error
        # making the oracle curve commit the base error allows to remove the impact of the base error when computing
        # the ratio of areas
        # line passing through base error at perc_rej = 0, and crossing
        # the line goes from x=0 to x=base_error/100*num_samples <- this is when the line intersects the x axis
        # which means the oracle ONLY REJECTS THE SAMPLES THAT ARE MISCASSIFIED
        # afterwards the function is set to zero
        orc_rejection = np.asarray(
                    [base_error * (1.0 - float(i) / float(base_error / 100.0 * num_samples)) for i in
                     range(int(base_error / 100.0 * num_samples))], dtype=np.float32)
        orc = np.zeros_like(rev_cum_errors)
        orc[0:orc_rejection.shape[0]] = orc_rejection
        auc_orc = 1.0 - auc(fraction_data / 100.0, orc / 100.0)
            
        # reported from -100 to 100
        rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0

        return rejection_ratio

    
    def auroc(self, labels, confidence):
        
        y_true = self.ensure_numpy(labels)[:, np.newaxis]
        y_score = self.ensure_numpy(confidence)[:, np.newaxis]
        auroc = roc_auc_score(y_true, y_score)
        
        return auroc


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    
    Source:
    https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        logits = logits.double()
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
